"""The MCP record tools against Postgres (throwaway database): typed records, payments,
overdue invoices, analytics, vendor contact, document text and save_extraction.

The ledger is one vendor with a contract, a PO, two invoices (one partly paid) and a
payment; the invoice and the contract come from stored documents with chunks.

    docker compose up -d --wait postgres
    pytest -m integration tests/integration/test_mcp_records_db.py
"""

import hashlib
import json
from datetime import date
from decimal import Decimal
from typing import Any

import pytest
from mcp.server.fastmcp.exceptions import ToolError
from sqlalchemy.orm import Session, sessionmaker

from ledgerlens.anomaly.scorer import AnomalyScorer
from ledgerlens.core.schemas import (
    Contract,
    DocType,
    Document,
    DocumentStatus,
    Invoice,
    LineItem,
    Payment,
    PaymentAllocation,
    PurchaseOrder,
    RateCardEntry,
)
from ledgerlens.ingestion.chunking import Chunk
from ledgerlens.ingestion.sinks import StoresSink
from ledgerlens.mcp_server.services import Services
from ledgerlens.mcp_server.tools import records
from ledgerlens.retrieval.analyze import VendorFinder
from ledgerlens.storage.neo4j.linker import VendorCandidate, VendorMatch
from ledgerlens.storage.postgres.db import session_scope
from ledgerlens.storage.postgres.repositories import Repositories

pytestmark = pytest.mark.integration

Sessions = sessionmaker[Session]
VENDOR = "Acme Supplies Ltd"
TODAY = date(2026, 5, 20)


def document(name: str, doc_type: DocType) -> Document:
    return Document(
        sha256=hashlib.sha256(name.encode()).hexdigest(),
        filename=f"{name}.pdf",
        doc_type=doc_type,
        pages=1,
    ).with_status(DocumentStatus.READY)


INVOICE_DOC = document("inv-100", DocType.INVOICE)
CONTRACT_DOC = document("ct-1", DocType.CONTRACT)


def invoice(number: str, day: date, total: str, *, lines: list[LineItem] | None = None) -> Invoice:
    return Invoice(
        invoice_number=number,
        vendor_name=VENDOR,
        vendor_tax_id="GB123456789",
        invoice_date=day,
        due_date=date(day.year, day.month + 1, day.day),
        po_number="PO-1",
        currency="GBP",
        total=Decimal(total),
        line_items=lines or [],
    )


@pytest.fixture
def services(sessions: Sessions) -> Services:
    with session_scope(sessions) as session:
        repos = Repositories(session)
        repos.vendors.upsert("V001", VENDOR, tax_id="GB123456789")
        repos.vendors.get("V001").email = "ap@acme.example"  # type: ignore[union-attr]
        repos.documents.upsert(INVOICE_DOC)
        repos.documents.upsert(CONTRACT_DOC)
        repos.chunks.replace(
            INVOICE_DOC.id,
            [
                Chunk(
                    index=0,
                    doc_type=DocType.INVOICE,
                    kind="header",
                    text="[INVOICE INV-100 | Vendor: Acme]\nINVOICE INV-100\nTotal 120.00",
                    body="INVOICE INV-100\nTotal 120.00",
                    pages=[1],
                )
            ],
        )
        repos.contracts.upsert(
            Contract(
                contract_number="CT-1",
                vendor_name=VENDOR,
                start_date=date(2026, 1, 1),
                payment_terms_days=30,
                currency="GBP",
                rate_card=[
                    RateCardEntry(item="Pallet wrap 500m", unit="roll", unit_price=Decimal("42.41"))
                ],
            ),
            id="CT-1",
            document_id=CONTRACT_DOC.id,
            vendor_id="V001",
        )
        repos.purchase_orders.upsert(
            PurchaseOrder(
                po_number="PO-1",
                vendor_name=VENDOR,
                order_date=date(2026, 1, 2),
                contract_number="CT-1",
                currency="GBP",
                total=Decimal("400.00"),
            ),
            id="PO-1",
            vendor_id="V001",
        )
        line = LineItem(
            description="Pallet wrap 500m",
            quantity=Decimal(2),
            unit_price=Decimal("42.41"),
            amount=Decimal("84.82"),
        )
        repos.invoices.upsert(
            invoice("INV-100", date(2026, 2, 1), "120.00", lines=[line]),
            id="I-100",
            document_id=INVOICE_DOC.id,
            vendor_id="V001",
        )
        repos.invoices.upsert(
            invoice("INV-101", date(2026, 3, 1), "50.00"), id="I-101", vendor_id="V001"
        )
        repos.payments.upsert(
            Payment(
                payment_reference="PAY-1",
                vendor_name=VENDOR,
                paid_at=date(2026, 3, 1),
                currency="GBP",
                amount=Decimal("100.00"),
                allocations=[PaymentAllocation(invoice_number="INV-100", amount=Decimal("100.00"))],
            ),
            id="PAY-1",
            vendor_id="V001",
        )

    class MasterSink(StoresSink):
        def resolve_vendor(self, result: Any, record: Any) -> VendorMatch:
            return VendorMatch("V001", "tax_id", 100.0)

    return Services(
        sessions=sessions,
        vendors=VendorFinder([VendorCandidate("V001", VENDOR)]),
        record_sink=MasterSink(sessions=sessions, scorer=AnomalyScorer(None)),
        today=TODAY,
    )


def test_get_invoice_by_number(services: Services) -> None:
    found = records.get_invoice(services, "INV-100", vendor="Acme")
    assert found.id == "I-100" and found.citation == "doc:INV-100"
    assert (found.paid, found.outstanding, found.status) == (
        Decimal("100.00"),
        Decimal("20.00"),
        "partially_paid",
    )
    (line,) = found.line_items
    assert line.model_dump(mode="json") == {
        "description": "Pallet wrap 500m",
        "quantity": "2",
        "unit_price": "42.41",
        "amount": "84.82",
    }
    assert records.get_invoice(services, "I-101").citation is None  # no document


def test_get_po_and_contract(services: Services) -> None:
    po = records.get_po(services, "PO-1")
    assert po.invoiced == Decimal("170.00") and po.invoice_ids == ["I-100", "I-101"]
    contract = records.get_contract(services, "CT-1")
    assert contract.citation == "doc:CT-1" and contract.payment_terms_days == 30
    assert contract.rate_card[0].unit_price == Decimal("42.41")


def test_unknown_record_is_a_tool_error(services: Services) -> None:
    with pytest.raises(ToolError, match="no invoice 'INV-999'"):
        records.get_invoice(services, "INV-999")


def test_get_payments(services: Services) -> None:
    paid = records.get_payments(services, "INV-100")
    assert (paid.paid, paid.outstanding) == (Decimal("100.00"), Decimal("20.00"))
    (payment,) = paid.payments
    assert (payment.payment_reference, payment.applied) == ("PAY-1", Decimal("100.00"))


def test_list_overdue(services: Services) -> None:
    overdue = records.list_overdue(services, min_days=1)
    assert overdue.as_of == TODAY and overdue.count == 2
    # most overdue first: INV-100 was due 2026-03-01, INV-101 on 2026-04-01
    assert [(i.invoice_number, i.days_overdue, i.outstanding) for i in overdue.invoices] == [
        ("INV-100", 80, Decimal("20.00")),
        ("INV-101", 49, Decimal("50.00")),
    ]
    assert overdue.outstanding_by_currency == {"GBP": Decimal("70.00")}
    assert records.list_overdue(services, min_days=60).count == 1


def test_run_analytics(services: Services) -> None:
    aging = records.run_analytics(services, "aging")
    assert aging.params == {"as_of": "2026-05-20"}
    buckets = {row["bucket"]: row["outstanding"] for row in aging.rows}
    assert buckets["61-90"] == "20.00" and buckets["31-60"] == "50.00"
    unpaid = records.run_analytics(services, "unpaid", {"vendor": "Acme"})
    assert unpaid.rows[0]["vendor_id"] == "V001" and unpaid.rows[0]["outstanding"] == "70.00"
    json.dumps(unpaid.model_dump(mode="json"))  # JSON all the way down


def test_vendor_contact_and_document_text(services: Services) -> None:
    contact = records.get_vendor_contact(services, "Acme")
    assert (contact.email, contact.contract_number, contact.payment_terms_days) == (
        "ap@acme.example",
        "CT-1",
        30,
    )
    text = records.get_document_text(services, "INV-100")
    assert text.document_id == INVOICE_DOC.id and text.record_number == "INV-100"
    assert text.text == "INVOICE INV-100\nTotal 120.00"  # context header removed
    with pytest.raises(ToolError, match="no stored document"):
        records.get_document_text(services, "INV-404")


def test_save_extraction_updates_the_documents_record(services: Services) -> None:
    corrected = invoice("INV-100", date(2026, 2, 1), "130.00").model_dump(mode="json")
    saved = records.save_extraction(services, INVOICE_DOC.id, corrected)
    # the record the document already holds is updated, not duplicated
    assert (saved.record_id, saved.vendor_id, saved.citation) == ("I-100", "V001", "doc:INV-100")
    assert saved.anomaly is not None and saved.anomaly.model_version == "rules-only"
    assert records.get_invoice(services, "INV-100").total == Decimal("130.00")
    assert records.get_po(services, "PO-1").invoice_ids == ["I-100", "I-101"]
    again = records.save_extraction(services, INVOICE_DOC.id, corrected)
    assert again == saved


def test_save_extraction_rejects_bad_input(services: Services) -> None:
    with pytest.raises(ToolError, match="invoice_number: Field required"):
        records.save_extraction(services, INVOICE_DOC.id, {"doc_type": "invoice"})
    good = invoice("INV-7", date(2026, 2, 1), "1.00").model_dump(mode="json")
    with pytest.raises(ToolError, match="no stored document"):
        records.save_extraction(services, "00000000-0000-4000-8000-000000000000", good)
