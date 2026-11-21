"""GET /anomalies and POST /anomalies/{id}/review over a stand-in store."""

from datetime import UTC, date, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ledgerlens.api.main import create_app
from ledgerlens.api.routers import anomalies
from ledgerlens.api.routers.anomalies import FLAG_SCORE, AnomalyOut

NOW = datetime(2026, 11, 21, 9, 0, tzinfo=UTC)
REASON = {"source": "rule", "code": "unknown_vendor", "text": "vendor 'X' is not in the master"}


def flag(score_id: int = 7, score: float = 0.9, **fields: Any) -> AnomalyOut:
    values: dict[str, Any] = {
        "id": score_id,
        "invoice_id": "SINV-1",
        "invoice_number": "INV-1",
        "vendor_name": "Westgate Trading FZE",
        "invoice_date": date(2026, 3, 1),
        "currency": "USD",
        "total": Decimal("1200.00"),
        "model_version": "ae-1",
        "score": score,
        "flagged": score >= FLAG_SCORE,
        "reasons": [REASON],
        "scored_at": NOW,
    }
    return AnomalyOut.model_validate(values | fields)


class FakeStore:
    def __init__(self, *flags: AnomalyOut) -> None:
        self.flags = {f.id: f for f in flags}
        self.listed: dict[str, Any] = {}

    def list(self, **filters: Any) -> list[AnomalyOut]:
        self.listed = filters
        return list(self.flags.values())

    def review(self, score_id: int, verdict: str, note: str | None) -> AnomalyOut | None:
        found = self.flags.get(score_id)
        if found is None:
            return None
        update = {"review": verdict, "review_note": note, "reviewed_at": NOW}
        self.flags[score_id] = found.model_copy(update=update)
        return self.flags[score_id]


@pytest.fixture
def store() -> FakeStore:
    return FakeStore(flag(7, 0.93), flag(8, 0.31, invoice_id="SINV-2"))


@pytest.fixture
def client(store: FakeStore) -> TestClient:
    app = create_app()
    app.dependency_overrides[anomalies.get_anomaly_store] = lambda: store
    return TestClient(app)


def test_default_list_is_unreviewed_flags(client: TestClient, store: FakeStore) -> None:
    response = client.get("/anomalies")
    assert response.status_code == 200
    body = response.json()
    assert [a["id"] for a in body] == [7, 8]
    assert body[0]["reasons"] == [REASON] and body[0]["flagged"] is True
    assert store.listed == {
        "review": "unreviewed",
        "min_score": FLAG_SCORE,
        "vendor_id": None,
        "model_version": None,
        "limit": 50,
        "offset": 0,
    }


def test_list_filters_pass_through(client: TestClient, store: FakeStore) -> None:
    params: dict[str, Any] = {
        "review": "false_positive",
        "min_score": 0,
        "vendor_id": "V9",
        "limit": 5,
    }
    assert client.get("/anomalies", params=params).status_code == 200
    assert store.listed["review"] == "false_positive"
    assert store.listed["min_score"] == 0 and store.listed["vendor_id"] == "V9"


@pytest.mark.parametrize(
    "params", [{"review": "maybe"}, {"min_score": 1.5}, {"min_score": -0.1}, {"limit": 501}]
)
def test_list_rejects_bad_filters(client: TestClient, params: dict[str, Any]) -> None:
    response = client.get("/anomalies", params=params)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


@pytest.mark.parametrize("verdict", ["confirmed", "false_positive"])
def test_review_records_the_verdict(client: TestClient, verdict: str) -> None:
    response = client.post(
        "/anomalies/7/review", json={"verdict": verdict, "note": "checked the PO"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["review"] == verdict and body["review_note"] == "checked the PO"
    assert body["reviewed_at"] is not None


def test_a_later_review_replaces_the_first(client: TestClient, store: FakeStore) -> None:
    client.post("/anomalies/8/review", json={"verdict": "confirmed"})
    client.post("/anomalies/8/review", json={"verdict": "false_positive"})
    assert store.flags[8].review == "false_positive" and store.flags[8].review_note is None


def test_reviewing_an_unknown_score_is_404(client: TestClient) -> None:
    response = client.post("/anomalies/99/review", json={"verdict": "confirmed"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


@pytest.mark.parametrize(
    "body",
    [
        {"verdict": "unsure"},
        {},
        {"verdict": "confirmed", "note": "x" * 2001},
    ],
)
def test_review_rejects_bad_bodies(client: TestClient, body: dict[str, Any]) -> None:
    assert client.post("/anomalies/7/review", json=body).status_code == 422


def test_non_numeric_id_is_422(client: TestClient) -> None:
    assert client.post("/anomalies/SINV-1/review", json={"verdict": "confirmed"}).status_code == 422


@pytest.mark.parametrize(("score", "flagged"), [(0.24, False), (0.25, True), (0.5, True)])
def test_flagged_follows_the_scorer_bands(score: float, flagged: bool) -> None:
    row = SimpleNamespace(
        id=1,
        model_version="ae-1",
        score=score,
        reasons=[REASON],
        review=None,
        review_note=None,
        reviewed_at=None,
        created_at=NOW,
    )
    invoice = SimpleNamespace(
        id="SINV-1",
        invoice_number="INV-1",
        document_id=None,
        vendor_id="V1",
        vendor_name="Acme",
        invoice_date=date(2026, 3, 1),
        currency="USD",
        total=Decimal("10.00"),
    )
    assert AnomalyOut.of(row, invoice).flagged is flagged  # type: ignore[arg-type]
