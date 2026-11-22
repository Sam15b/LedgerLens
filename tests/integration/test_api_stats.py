"""The overview's queries against Postgres (throwaway database).

docker compose up -d --wait postgres
pytest -m integration tests/integration/test_api_stats.py
"""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy.orm import Session, sessionmaker

from ledgerlens.api.routers.stats import StatsStore
from ledgerlens.core.schemas import DocType, Document, DocumentStatus, Invoice, LineItem
from ledgerlens.storage.postgres.db import session_scope
from ledgerlens.storage.postgres.models import LLMCallRow
from ledgerlens.storage.postgres.repositories import ReminderRepository, Repositories

pytestmark = pytest.mark.integration

AS_OF = date(2026, 11, 22)
DOCS = [f"{n}2c3a8e-4b7d-4c1e-9a3f-0d2b5e7c9a10" for n in ("1a", "2a", "3a")]


def invoice(number: str, total: str, due: date, currency: str = "GBP") -> Invoice:
    return Invoice(
        invoice_number=number,
        vendor_name="Acme Supplies Ltd",
        invoice_date=date(2026, 10, 1),
        due_date=due,
        currency=currency,
        total=Decimal(total),
        line_items=[LineItem(description="Widgets", amount=Decimal(total))],
    )


@pytest.fixture
def ledger(sessions: sessionmaker[Session]) -> sessionmaker[Session]:
    with session_scope(sessions) as session:
        repos = Repositories(session)
        repos.vendors.upsert("V001", "Acme Supplies Ltd")
        statuses = (DocumentStatus.READY, DocumentStatus.READY, DocumentStatus.QUEUED)
        for i, (doc_id, status) in enumerate(zip(DOCS, statuses, strict=True)):
            repos.documents.upsert(
                Document(
                    id=doc_id,
                    sha256=f"{i}" * 64,
                    filename=f"doc{i}.pdf",
                    doc_type=DocType.INVOICE,
                    status=status,
                )
            )
        repos.invoices.upsert(
            invoice("INV-1", "100.00", date(2026, 11, 1)), id="I-1", vendor_id="V001"
        )  # overdue
        repos.invoices.upsert(
            invoice("INV-2", "50.00", date(2026, 12, 1)), id="I-2", vendor_id="V001"
        )  # open, not due
        repos.invoices.upsert(
            invoice("INV-3", "70.00", date(2026, 11, 2), currency="USD"), id="I-3", vendor_id="V001"
        )
        scores = repos.anomaly_scores
        scores.upsert("I-1", "ae-1", 0.80, [])  # flagged, unreviewed
        scores.upsert("I-2", "ae-1", 0.10, [])  # not a flag
        scores.upsert("I-3", "ae-1", 0.90, [])  # flagged, then reviewed below
        session.flush()
        flagged_i3 = scores.for_invoice("I-3")[0]
        scores.review(flagged_i3.id, "false_positive", None)
        reminders = ReminderRepository(session)
        reminders.create("I-1", "Please pay INV-1.", "firm")
        decided = reminders.create("I-3", "Please pay INV-3.", "polite")
        reminders.decide(decided.id, "rejected")
        noon = datetime(2026, 11, 22, 12, tzinfo=UTC)
        for ts, cost in ((noon, "0.010"), (noon, "0.002"), (noon - timedelta(days=2), None)):
            session.add(
                LLMCallRow(
                    ts=ts,
                    request_id="r",
                    alias="smart",
                    provider="openai",
                    model="gpt-4o",
                    tokens_in=10,
                    tokens_out=1,
                    cost_usd=None if cost is None else Decimal(cost),
                    latency_ms=5.0,
                    streamed=False,
                )
            )
    return sessions


def test_overview_counts_and_totals(ledger: sessionmaker[Session]) -> None:
    overview = StatsStore(ledger).overview(AS_OF, days=3)

    assert overview.documents.model_dump() == {
        "total": 3,
        "queued": 1,
        "processing": 0,
        "ready": 2,
        "failed": 0,
    }
    by_currency = {u.currency: u for u in overview.unpaid}
    assert by_currency["GBP"].outstanding == Decimal("150.00")
    assert (by_currency["GBP"].open_invoices, by_currency["GBP"].overdue_invoices) == (2, 1)
    assert by_currency["GBP"].overdue == Decimal("100.00")
    assert by_currency["USD"].overdue_invoices == 1
    assert overview.unpaid[0].currency == "GBP"  # most open invoices first
    assert (overview.open_invoices, overview.overdue_invoices) == (3, 2)
    assert overview.flagged_invoices == 1  # I-3's flag was reviewed
    assert overview.pending_reminders == 1
    assert [(d.day.isoformat(), d.calls, d.cost_usd) for d in overview.llm_cost_by_day] == [
        ("2026-11-20", 1, 0.0),
        ("2026-11-21", 0, 0.0),
        ("2026-11-22", 2, 0.012),
    ]
    assert overview.llm_costs_recorded


def test_an_empty_ledger(sessions: sessionmaker[Session]) -> None:
    overview = StatsStore(sessions).overview(AS_OF, days=1)

    assert overview.documents.total == 0 and overview.unpaid == []
    assert overview.flagged_invoices == 0 and not overview.llm_costs_recorded
    assert [d.calls for d in overview.llm_cost_by_day] == [0]
