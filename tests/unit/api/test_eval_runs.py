"""GET /eval/runs over a stand-in store: order, metrics, the suite filter and limits."""

from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ledgerlens.api.main import create_app
from ledgerlens.api.routers import eval as eval_runs
from ledgerlens.api.routers.eval import EvalRunOut

RUNS = [
    EvalRunOut(
        id="b",
        suite="full",
        git_sha="d33fce7",
        config={"retriever": "R4"},
        metrics={"recall@5": 0.954, "mrr": 0.929, "facts_found": None},
        created_at=datetime(2026, 11, 22, 15, 14, tzinfo=UTC),
    ),
    EvalRunOut(
        id="a",
        suite="ci",
        git_sha=None,
        config={},
        metrics={"recall@5": 0.87, "questions": 40},
        created_at=datetime(2026, 11, 22, 14, 30, tzinfo=UTC),
    ),
]


class FakeStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str | None, int]] = []

    def runs(self, suite: str | None, limit: int) -> list[EvalRunOut]:
        self.calls.append((suite, limit))
        return [r for r in RUNS if suite is None or r.suite == suite][:limit]


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def client(store: FakeStore) -> TestClient:
    app = create_app()
    app.dependency_overrides[eval_runs.get_eval_store] = lambda: store
    return TestClient(app)


def test_runs_newest_first_with_their_metrics(client: TestClient, store: FakeStore) -> None:
    response = client.get("/eval/runs")
    assert response.status_code == 200
    body: list[dict[str, Any]] = response.json()
    assert [r["id"] for r in body] == ["b", "a"]
    assert body[0]["metrics"] == {"recall@5": 0.954, "mrr": 0.929, "facts_found": None}
    assert body[0]["config"] == {"retriever": "R4"} and body[0]["git_sha"] == "d33fce7"
    assert body[1]["metrics"]["questions"] == 40
    assert store.calls == [(None, 50)]


def test_one_suite(client: TestClient, store: FakeStore) -> None:
    body = client.get("/eval/runs", params={"suite": "ci", "limit": 5}).json()
    assert [r["suite"] for r in body] == ["ci"]
    assert store.calls == [("ci", 5)]


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 501}, {"suite": "x" * 65}])
def test_bad_parameters_are_422(client: TestClient, params: dict[str, Any]) -> None:
    response = client.get("/eval/runs", params=params)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
