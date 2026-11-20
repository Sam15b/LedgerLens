"""reconcile, score_anomaly and draft_reminder against Postgres (throwaway database).

One vendor with a contract (Net 30), a PO for 4 rolls, an invoice for 2 that matches, and
a copy of it billed again under another number, so the PO is over-invoiced.

    docker compose up -d --wait postgres
    pytest -m integration tests/integration/test_mcp_reconcile_db.py
"""

from datetime import date
from decimal import Decimal

import pytest
from mcp.server.fastmcp.exceptions import ToolError
from sqlalchemy.orm import Session, sessionmaker

from ledgerlens.anomaly.scorer import AnomalyScorer
from ledgerlens.core.schemas import Contract, Invoice, LineItem, PurchaseOrder, RateCardEntry
from ledgerlens.mcp_server.services import Services
from ledgerlens.mcp_server.tools.anomaly import score_anomaly
from ledgerlens.mcp_server.tools.reconcile import reconcile
from ledgerlens.mcp_server.tools.reminders import draft_reminder
from ledgerlens.retrieval.analyze import VendorFinder
from ledgerlens.storage.neo4j.linker import VendorCandidate
from ledgerlens.storage.postgres.db import session_scope
from ledgerlens.storage.postgres.repositories import ReminderRepository, Repositories

pytestmark = pytest.mark.integration

Sessions = sessionmaker[Session]
VENDOR = "Acme Supplies Ltd"
WRAP = "Pallet wrap 500m"


def invoice(number: str, rolls: int, price: str = "42.41") -> Invoice:
    amount = Decimal(rolls) * Decimal(price)
    return Invoice(
        invoice_number=number,
        vendor_name=VENDOR,
        invoice_date=date(2026, 2, 1),
        due_date=date(2026, 3, 3),
        po_number="PO-1",
        currency="GBP",
        subtotal=amount,
        tax=Decimal(0),
        total=amount,
        line_items=[
            LineItem(
                description=WRAP, quantity=Decimal(rolls), unit_price=Decimal(price), amount=amount
            )
        ],
    )


@pytest.fixture
def services(sessions: Sessions) -> Services:
    with session_scope(sessions) as session:
        repos = Repositories(session)
        repos.vendors.upsert("V001", VENDOR, tax_id="GB123456789")
        repos.contracts.upsert(
            Contract(
                contract_number="CT-1",
                vendor_name=VENDOR,
                start_date=date(2026, 1, 1),
                payment_terms_days=30,
                currency="GBP",
                rate_card=[RateCardEntry(item=WRAP, unit="roll", unit_price=Decimal("42.41"))],
            ),
            id="CT-1",
            vendor_id="V001",
        )
        repos.purchase_orders.upsert(
            PurchaseOrder(
                po_number="PO-1",
                vendor_name=VENDOR,
                order_date=date(2026, 1, 20),
                contract_number="CT-1",
                currency="GBP",
                total=Decimal("169.64"),
                line_items=[
                    LineItem(
                        description=WRAP,
                        quantity=Decimal(4),
                        unit_price=Decimal("42.41"),
                        amount=Decimal("169.64"),
                    )
                ],
            ),
            id="PO-1",
            vendor_id="V001",
        )
        repos.invoices.upsert(invoice("INV-1", 2), id="I-1", vendor_id="V001")
    return Services(
        sessions=sessions,
        vendors=VendorFinder([VendorCandidate("V001", VENDOR)]),
        scorer=AnomalyScorer(None),
        today=date(2026, 2, 10),
    )


def test_matching_invoice(services: Services) -> None:
    report = reconcile(services, "INV-1")
    assert report.status == "match", [c.message for c in report.checks]
    assert report.contract_number == "CT-1" and report.payment_status == "unpaid"
    statuses = {c.check: c.status for c in report.checks}
    assert statuses == dict.fromkeys(statuses, "pass")


def test_over_invoiced_po_and_terms(sessions: Sessions, services: Services) -> None:
    with session_scope(sessions) as session:
        rush = invoice("INV-2", 3, price="45.00").model_copy(update={"due_date": date(2026, 2, 2)})
        Repositories(session).invoices.upsert(rush, id="I-2", vendor_id="V001")
    report = reconcile(services, "INV-2", vendor="Acme")
    assert report.status == "mismatch"
    assert report.mismatches == ["line_qty", "unit_price", "payment", "terms"]  # due 02-02
    by_check = {c.check: c for c in report.checks}
    assert by_check["line_qty"].message == "Pallet wrap 500m: 5 invoiced vs 4 ordered"
    # the first invoice now shares the over-invoiced PO too
    assert reconcile(services, "INV-1").mismatches == ["line_qty"]


def test_wrong_vendor_po(sessions: Sessions, services: Services) -> None:
    with session_scope(sessions) as session:
        repos = Repositories(session)
        repos.vendors.upsert("V002", "Brightline Packaging Ltd")
        other = invoice("B-1", 1).model_copy(
            update={"vendor_name": "Brightline Packaging Ltd", "po_number": "PO-1"}
        )
        repos.invoices.upsert(other, id="I-B1", vendor_id="V002")
    report = reconcile(services, "I-B1")
    assert "vendor_match" in report.mismatches


def test_score_anomaly_stores_the_score(services: Services) -> None:
    result = score_anomaly(services, "INV-1")
    assert (result.model_version, result.flagged) == ("rules-only", False)
    assert [r.code for r in result.reasons] == ["nothing_unusual"]


def test_draft_reminder_is_pending_and_checks_facts(sessions: Sessions, services: Services) -> None:
    services._today = date(2026, 3, 25)  # 22 days past due
    body = "Invoice INV-1 for GBP 84.82 under PO-1 was due on 3 March 2026."
    draft = draft_reminder(services, "INV-1", body, "firm")
    assert (draft.status, draft.sent, draft.suggested_tone) == ("pending_approval", False, "firm")
    assert draft.days_overdue == 22 and draft.missing_facts == []
    thin = draft_reminder(services, "INV-1", "Please pay INV-1.", "friendly")
    assert thin.missing_facts == ["amount", "due_date", "po_number"]
    with sessions() as session:
        stored = ReminderRepository(session).find(invoice_id="I-1")
    assert {r.status for r in stored} == {"pending_approval"} and len(stored) == 2
    with pytest.raises(ToolError, match="body is empty"):
        draft_reminder(services, "INV-1", "  ", "firm")
