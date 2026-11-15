"""Repository upserts against a real PostgreSQL (``docker compose up -d postgres``).

The day's exit criterion: upserting the same record twice leaves one row -- and, since the
update only fires when a value changed, leaves that row untouched.
"""

import hashlib
from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from ledgerlens.core.schemas import (
    Contract,
    DocType,
    Document,
    DocumentStatus,
    Invoice,
    Payment,
    PaymentAllocation,
    PurchaseOrder,
    RateCardEntry,
    Receipt,
)
from ledgerlens.ingestion.chunking import Chunk
from ledgerlens.storage.postgres.db import session_scope
from ledgerlens.storage.postgres.models import (
    Base,
    ChunkRow,
    ContractRow,
    DocumentRow,
    InvoiceLineItemRow,
    InvoiceRow,
    PaymentAllocationRow,
    PaymentRow,
    PurchaseOrderRow,
    VendorRow,
)
from ledgerlens.storage.postgres.repositories import (
    ChunkRepository,
    DocumentRepository,
    InvoiceRepository,
    PaymentRepository,
    Repositories,
    VendorRepository,
)

pytestmark = pytest.mark.integration

Sessions = sessionmaker[Session]
D = Decimal


def make_document(content: bytes = b"invoice", **changes: object) -> Document:
    return Document.model_validate(
        {"sha256": hashlib.sha256(content).hexdigest(), "filename": "inv.pdf", "pages": 1} | changes
    )


def make_chunk(index: int, text: str = "body") -> Chunk:
    return Chunk(
        index=index, doc_type=DocType.INVOICE, kind="body", text=text, body=text, pages=[1]
    )


def invoice(number: str = "INV-1", total: str = "120.00", **changes: object) -> Invoice:
    data = {
        "invoice_number": number,
        "vendor_name": "Acme Supplies Ltd",
        "invoice_date": date(2026, 9, 1),
        "due_date": date(2026, 10, 1),
        "po_number": "PO-7",
        "currency": "GBP",
        "subtotal": D(total) - D(20),
        "tax": D(20),
        "total": D(total),
        "line_items": [
            {"description": "Boxes", "quantity": 10, "unit_price": 5, "amount": 50},
            {"description": "Tape", "quantity": 10, "unit_price": 5, "amount": 50},
        ],
    }
    return Invoice.model_validate(data | changes)


def payment(*allocations: tuple[str, str], reference: str = "PAY-1") -> Payment:
    return Payment(
        payment_reference=reference,
        vendor_name="Acme Supplies Ltd",
        paid_at=date(2026, 10, 15),
        method="bank_transfer",
        currency="GBP",
        amount=sum((D(a) for _, a in allocations), D(0)),
        allocations=[PaymentAllocation(invoice_number=n, amount=D(a)) for n, a in allocations],
    )


def rows(sessions: Sessions, model: type[Base]) -> int:
    with sessions() as session:
        return session.scalar(select(func.count()).select_from(model)) or 0


def status(sessions: Sessions, invoice_id: str) -> str:
    with sessions() as session:
        row = InvoiceRepository(session).get(invoice_id)
        assert row is not None
        return row.status


@pytest.fixture
def vendor(sessions: Sessions) -> str:
    with session_scope(sessions) as session:
        VendorRepository(session).upsert(
            "V001", "Acme Supplies Ltd", aliases=["ACME SUPPLIES"], tax_id="GB123456789"
        )
    return "V001"


# --------------------------------------------------------------------------- documents


def test_document_upsert_twice_leaves_one_unchanged_row(sessions: Sessions) -> None:
    document = make_document()
    with session_scope(sessions) as session:
        DocumentRepository(session).upsert(document)
    with sessions() as session:
        first = session.get(DocumentRow, document.id)
        assert first is not None
        stamp = first.updated_at

    with session_scope(sessions) as session:
        DocumentRepository(session).upsert(document)

    assert rows(sessions, DocumentRow) == 1
    with sessions() as session:
        again = session.get(DocumentRow, document.id)
        assert again is not None and again.updated_at == stamp  # identical write: no-op
        assert DocumentRepository(session).get(document.id) == document


def test_claim_lets_exactly_one_document_have_the_bytes(sessions: Sessions) -> None:
    first, second = make_document(), make_document()  # same sha256, different ids
    with session_scope(sessions) as session:
        repo = DocumentRepository(session)
        assert repo.claim(first) is None
        assert repo.claim(second) == first.id

    assert rows(sessions, DocumentRow) == 1
    with sessions() as session:
        found = DocumentRepository(session).get_by_sha256(first.sha256.upper())
    assert found is not None and found.id == first.id


def test_status_transitions(sessions: Sessions) -> None:
    document = make_document()
    with session_scope(sessions) as session:
        repo = DocumentRepository(session)
        repo.upsert(document)
        assert repo.set_status(document.id, DocumentStatus.FAILED, error="ocr: boom")
        assert not repo.set_status("00000000-0000-4000-8000-000000000000", DocumentStatus.READY)
    with sessions() as session:
        repo = DocumentRepository(session)
        stored = repo.get(document.id)
        assert [d.id for d in repo.list_by_status(DocumentStatus.FAILED)] == [document.id]
    assert stored is not None and (stored.status, stored.error) == ("failed", "ocr: boom")

    with session_scope(sessions) as session:
        DocumentRepository(session).set_status(document.id, DocumentStatus.READY, error="stale")
    with sessions() as session:
        stored = DocumentRepository(session).get(document.id)
    assert stored is not None and (stored.status, stored.error) == ("ready", None)


def test_chunks_replace_is_idempotent_and_drops_leftovers(sessions: Sessions) -> None:
    document = make_document()
    with session_scope(sessions) as session:
        DocumentRepository(session).upsert(document)
        ChunkRepository(session).replace(document.id, [make_chunk(i) for i in range(3)])
        ChunkRepository(session).replace(document.id, [make_chunk(i) for i in range(3)])
    assert rows(sessions, ChunkRow) == 3

    with session_scope(sessions) as session:
        ChunkRepository(session).replace(document.id, [make_chunk(0, "new\x00 text")])

    with sessions() as session:
        repo = ChunkRepository(session)
        stored = repo.for_document(document.id)
        texts = repo.texts([f"{document.id}#c0", f"{document.id}#c1"])
    assert [(r.chunk_index, r.text) for r in stored] == [(0, "new text")]
    assert texts == {f"{document.id}#c0": "new text"}


def test_deleting_a_document_removes_its_chunks_and_records(sessions: Sessions) -> None:
    document = make_document()
    with session_scope(sessions) as session:
        repos = Repositories(session)
        repos.documents.upsert(document)
        repos.chunks.replace(document.id, [make_chunk(0)])
        repos.upsert_record(invoice(), id=document.id, document_id=document.id)
    with session_scope(sessions) as session:
        assert DocumentRepository(session).delete(document.id)

    assert (rows(sessions, ChunkRow), rows(sessions, InvoiceRow)) == (0, 0)
    assert rows(sessions, InvoiceLineItemRow) == 0


# --------------------------------------------------------------------------- records


def test_invoice_upsert_twice_leaves_one_invoice_and_one_set_of_lines(
    sessions: Sessions, vendor: str
) -> None:
    for _ in range(2):
        with session_scope(sessions) as session:
            InvoiceRepository(session).upsert(invoice(), id="inv-1", vendor_id=vendor)

    assert rows(sessions, InvoiceRow) == 1
    assert rows(sessions, InvoiceLineItemRow) == 2
    with sessions() as session:
        stored = InvoiceRepository(session).get("inv-1")
        assert stored is not None
        assert (stored.total, stored.status, stored.vendor_id) == (D("120"), "unpaid", vendor)
        assert [line.description for line in stored.line_items] == ["Boxes", "Tape"]


def test_reingest_with_fewer_lines_replaces_them(sessions: Sessions) -> None:
    with session_scope(sessions) as session:
        InvoiceRepository(session).upsert(invoice(), id="inv-1")
    shorter = invoice(
        total="70.00",
        line_items=[{"description": "Boxes", "quantity": 10, "unit_price": 5, "amount": 50}],
    )
    with session_scope(sessions) as session:
        InvoiceRepository(session).upsert(shorter, id="inv-1")

    with sessions() as session:
        stored = InvoiceRepository(session).get("inv-1")
        assert stored is not None
        assert stored.total == D("70")
        assert [line.description for line in stored.line_items] == ["Boxes"]


def test_reingesting_keeps_workflow_status(sessions: Sessions) -> None:
    with session_scope(sessions) as session:
        InvoiceRepository(session).upsert(invoice(), id="inv-1", status="disputed")
    with session_scope(sessions) as session:
        InvoiceRepository(session).upsert(invoice(), id="inv-1")

    assert status(sessions, "inv-1") == "disputed"


def test_payments_settle_invoices_and_reingest_is_a_no_op(sessions: Sessions, vendor: str) -> None:
    with session_scope(sessions) as session:
        repos = Repositories(session)
        repos.invoices.upsert(invoice("INV-1"), id="inv-1", vendor_id=vendor)
        repos.invoices.upsert(invoice("INV-2"), id="inv-2", vendor_id=vendor)
    for _ in range(2):
        with session_scope(sessions) as session:
            Repositories(session).upsert_record(
                payment(("INV-1", "120.00"), ("INV-2", "50.00")), id="pay-1", vendor_id=vendor
            )

    assert rows(sessions, PaymentRow) == 1
    assert rows(sessions, PaymentAllocationRow) == 2
    assert (status(sessions, "inv-1"), status(sessions, "inv-2")) == ("paid", "partially_paid")
    with sessions() as session:
        overdue = InvoiceRepository(session).list_overdue(date(2026, 11, 1))
    assert [(o.invoice.id, o.outstanding, o.days_overdue) for o in overdue] == [
        ("inv-2", D("70"), 31)
    ]


def test_correcting_a_payment_reopens_the_invoice(sessions: Sessions) -> None:
    with session_scope(sessions) as session:
        repos = Repositories(session)
        repos.invoices.upsert(invoice("INV-1"), id="inv-1")
        repos.payments.upsert(payment(("INV-1", "120.00")), id="pay-1")
    assert status(sessions, "inv-1") == "paid"

    with session_scope(sessions) as session:
        PaymentRepository(session).upsert(payment(), id="pay-1")  # allocation removed

    assert status(sessions, "inv-1") == "unpaid"
    assert rows(sessions, PaymentAllocationRow) == 0


def test_unmatched_allocations_keep_the_printed_number(sessions: Sessions) -> None:
    with session_scope(sessions) as session:
        PaymentRepository(session).upsert(payment(("INV-404", "10.00")), id="pay-1")

    with sessions() as session:
        stored = PaymentRepository(session).get("pay-1")
        assert stored is not None
        assert [(a.invoice_number, a.invoice_id) for a in stored.allocations] == [("INV-404", None)]
        assert [p.id for p in PaymentRepository(session).find_by_reference("PAY-1")] == ["pay-1"]


def test_list_overdue_filters_by_vendor_and_date(sessions: Sessions, vendor: str) -> None:
    with session_scope(sessions) as session:
        repos = Repositories(session)
        repos.invoices.upsert(invoice("INV-1"), id="a", vendor_id=vendor)
        repos.invoices.upsert(invoice("INV-2", due_date=date(2026, 12, 1)), id="b")
        repos.invoices.upsert(invoice("INV-3"), id="c", status="void")

    with sessions() as session:
        repo = InvoiceRepository(session)
        assert [o.invoice.id for o in repo.list_overdue(date(2026, 12, 2))] == ["a", "b"]
        assert [o.invoice.id for o in repo.list_overdue(date(2026, 12, 2), limit=1)] == ["a"]
        assert [o.invoice.id for o in repo.list_overdue(date(2026, 11, 1))] == ["a"]
        assert repo.list_overdue(date(2026, 12, 2), vendor_id="V999") == []
        assert repo.list_overdue(date(2026, 10, 1)) == []  # due today is not overdue


def test_find_by_number_filters_by_vendor(sessions: Sessions, vendor: str) -> None:
    with session_scope(sessions) as session:
        repos = Repositories(session)
        repos.invoices.upsert(invoice("INV-1"), id="a", vendor_id=vendor)
        repos.invoices.upsert(invoice("INV-1"), id="b")

    with sessions() as session:
        repo = InvoiceRepository(session)
        assert [r.id for r in repo.find_by_number("INV-1")] == ["a", "b"]
        assert [r.id for r in repo.find_by_number("INV-1", vendor_id=vendor)] == ["a"]


def test_purchase_orders_resolve_their_contract(sessions: Sessions, vendor: str) -> None:
    contract = Contract(
        contract_number="MSA-1",
        vendor_name="Acme Supplies Ltd",
        start_date=date(2026, 1, 1),
        payment_terms_days=30,
        rate_card=[RateCardEntry(item="Boxes", unit="each", unit_price=D("5"))],
    )
    po = PurchaseOrder(
        po_number="PO-7",
        vendor_name="Acme Supplies Ltd",
        order_date=date(2026, 8, 20),
        contract_number="MSA-1",
        total=D("120"),
    )
    for _ in range(2):
        with session_scope(sessions) as session:
            repos = Repositories(session)
            repos.upsert_record(contract, id="c-1", vendor_id=vendor)
            repos.upsert_record(po, id="po-1", vendor_id=vendor)

    assert (rows(sessions, ContractRow), rows(sessions, PurchaseOrderRow)) == (1, 1)
    with sessions() as session:
        repos = Repositories(session)
        stored = repos.purchase_orders.get("po-1")
        stored_contract = repos.contracts.get("c-1")
        assert stored is not None and stored.contract_id == "c-1"
        assert [r.id for r in repos.purchase_orders.find_by_number("PO-7")] == ["po-1"]
        assert [r.id for r in repos.contracts.find_by_number("MSA-1", vendor_id=vendor)] == ["c-1"]
    assert stored_contract is not None
    assert stored_contract.rate_card == [{"item": "Boxes", "unit": "each", "unit_price": "5"}]


def test_receipts_are_stored_as_paid_invoices(sessions: Sessions) -> None:
    receipt = Receipt(
        vendor_name="Kedai Kopi", receipt_date=date(2026, 3, 1), currency="MYR", total=D("9.50")
    )
    with session_scope(sessions) as session:
        Repositories(session).upsert_record(receipt, id="r-1")

    with sessions() as session:
        repo = InvoiceRepository(session)
        stored = repo.get("r-1")
        assert stored is not None
        assert (stored.invoice_number, stored.status, stored.total) == (
            "RECEIPT-r-1",
            "paid",
            D("9.5"),
        )
        assert repo.list_overdue(date(2027, 1, 1)) == []


def test_vendor_aliases_and_tax_id_lookup(sessions: Sessions, vendor: str) -> None:
    with session_scope(sessions) as session:
        repo = VendorRepository(session)
        assert repo.add_alias(vendor, "Acme Supplies") is True
        assert repo.add_alias(vendor, "Acme Supplies") is False
        assert repo.add_alias(vendor, "Acme Supplies Ltd") is False  # the canonical name
        stored = repo.get(vendor)
        assert stored is not None and stored.aliases == ["ACME SUPPLIES", "Acme Supplies"]
        with pytest.raises(KeyError):
            repo.add_alias("V999", "Nobody")

    with session_scope(sessions) as session:  # re-import the master data: aliases reset
        VendorRepository(session).upsert(
            vendor, "Acme Supplies Ltd", aliases=["ACME SUPPLIES"], tax_id="GB123456789"
        )

    with sessions() as session:
        repo = VendorRepository(session)
        found = repo.get_by_tax_id("GB123456789")
        assert found is not None and found.id == vendor
        assert found.aliases == ["ACME SUPPLIES"]
        assert [v.id for v in repo.all()] == [vendor]
    assert rows(sessions, VendorRow) == 1
