"""GET /stats/overview over a stand-in store, the per-currency and per-day helpers, and
CORS for the dashboard's origin."""

from datetime import date
from decimal import Decimal
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ledgerlens.api.main import create_app
from ledgerlens.api.routers import stats
from ledgerlens.api.routers.stats import (
    DailyCost,
    DocumentCounts,
    Overview,
    fill_days,
    sum_unpaid,
)

AS_OF = date(2026, 11, 22)


class FakeStore:
    def __init__(self) -> None:
        self.calls: list[tuple[date, int]] = []

    def overview(self, as_of: date, days: int) -> Overview:
        self.calls.append((as_of, days))
        unpaid = sum_unpaid([("GBP", 2, Decimal("150.00"), 1, Decimal("100.00"))])
        return Overview(
            as_of=as_of,
            documents=DocumentCounts(total=3, ready=2, queued=1),
            unpaid=unpaid,
            open_invoices=2,
            overdue_invoices=1,
            flagged_invoices=4,
            pending_reminders=1,
            llm_cost_by_day=fill_days({as_of: (3, 0.012)}, as_of, days),
            llm_costs_recorded=True,
        )


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def client(store: FakeStore) -> TestClient:
    app = create_app()
    app.dependency_overrides[stats.get_stats_store] = lambda: store
    return TestClient(app)


def test_overview(client: TestClient, store: FakeStore) -> None:
    response = client.get("/stats/overview", params={"as_of": "2026-11-22", "days": 3})

    assert response.status_code == 200
    body: dict[str, Any] = response.json()
    assert store.calls == [(AS_OF, 3)]
    assert body["documents"] == {"total": 3, "queued": 1, "processing": 0, "ready": 2, "failed": 0}
    assert body["unpaid"] == [
        {
            "currency": "GBP",
            "open_invoices": 2,
            "outstanding": "150.00",
            "overdue_invoices": 1,
            "overdue": "100.00",
        }
    ]
    assert (body["flagged_invoices"], body["pending_reminders"]) == (4, 1)
    assert [d["day"] for d in body["llm_cost_by_day"]] == ["2026-11-20", "2026-11-21", "2026-11-22"]
    assert body["llm_cost_by_day"][-1] == {"day": "2026-11-22", "calls": 3, "cost_usd": 0.012}


def test_overview_defaults_to_today_and_a_week(client: TestClient, store: FakeStore) -> None:
    assert client.get("/stats/overview").status_code == 200
    assert store.calls == [(date.today(), 7)]


@pytest.mark.parametrize("days", [0, 91])
def test_days_is_bounded(client: TestClient, days: int) -> None:
    response = client.get("/stats/overview", params={"days": days})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_unpaid_is_summed_per_currency_most_invoices_first() -> None:
    rows = [
        ("GBP", 1, Decimal("10.00"), 0, Decimal("0")),
        ("USD", 2, Decimal("500.00"), 2, Decimal("500.00")),
        ("GBP", 3, Decimal("30.50"), 1, Decimal("5.00")),
    ]

    totals = sum_unpaid(rows)

    # GBP first: more open invoices, although USD 500 is "more" than GBP 40.50
    assert [(u.currency, u.open_invoices, u.outstanding, u.overdue) for u in totals] == [
        ("GBP", 4, Decimal("40.50"), Decimal("5.00")),
        ("USD", 2, Decimal("500.00"), Decimal("500.00")),
    ]


def test_days_without_calls_are_zero() -> None:
    days = fill_days({date(2026, 11, 21): (2, 0.5)}, AS_OF, 3)

    assert days == [
        DailyCost(day=date(2026, 11, 20), calls=0, cost_usd=0.0),
        DailyCost(day=date(2026, 11, 21), calls=2, cost_usd=0.5),
        DailyCost(day=date(2026, 11, 22), calls=0, cost_usd=0.0),
    ]


# --------------------------------------------------------------------------- CORS


def test_the_dashboard_origin_may_call_the_api(client: TestClient) -> None:
    preflight = client.options(
        "/stats/overview",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "X-Request-ID",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:3000"

    response = client.get("/stats/overview", headers={"Origin": "http://localhost:3000"})
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    exposed = response.headers["access-control-expose-headers"].lower()
    assert "x-request-id" in exposed and "x-thread-id" in exposed


def test_other_origins_are_not_allowed(client: TestClient) -> None:
    preflight = client.options(
        "/stats/overview",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"},
    )
    assert preflight.status_code == 400
    response = client.get("/stats/overview", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in response.headers


def test_origins_come_from_the_settings() -> None:
    from ledgerlens.core.config import Settings

    settings = Settings(cors_origins=" https://dash.example/ , http://localhost:3000 ,")
    assert settings.cors_origin_list == ["https://dash.example", "http://localhost:3000"]
