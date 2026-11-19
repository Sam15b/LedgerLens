"""Idempotent reads and writes on the PostgreSQL system of record.

Every write is an upsert -- ``INSERT ... ON CONFLICT DO UPDATE`` keyed by the row's id --
whose update only fires when a value actually changed. Writing the same record twice
leaves one row, untouched by the second write (``updated_at`` included), so a re-run of
ingestion is safe at any point.

Repositories wrap a :class:`~sqlalchemy.orm.Session` and never commit: the caller owns
the transaction, usually with :func:`~ledgerlens.storage.postgres.db.session_scope`::

    with session_scope() as session:
        documents = DocumentRepository(session)
        documents.upsert(document)
        ChunkRepository(session).replace(document.id, chunks)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy import Table, delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from ledgerlens.core.schemas import (
    ROUNDING_TOLERANCE,
    Contract,
    DocType,
    Document,
    DocumentStatus,
    ExtractedRecord,
    Invoice,
    LineItem,
    Payment,
    PaymentAllocation,
    PurchaseOrder,
    RateCardEntry,
    Receipt,
)
from ledgerlens.storage.postgres.models import (
    AnomalyScoreRow,
    Base,
    ChunkRow,
    ContractRow,
    DocumentRow,
    InvoiceLineItemRow,
    InvoiceRow,
    PaymentAllocationRow,
    PaymentRow,
    POLineItemRow,
    PurchaseOrderRow,
    VendorRow,
)

if TYPE_CHECKING:
    from ledgerlens.ingestion.chunking import Chunk


def upsert(
    session: Session,
    model: type[Base],
    rows: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    conflict: Sequence[str] = ("id",),
    keep: Iterable[str] = (),
) -> None:
    """``INSERT rows ON CONFLICT (conflict) DO UPDATE`` the other columns given.

    Columns in ``keep`` are written on insert but never overwritten (e.g. a workflow
    status). The update only runs for rows whose values differ, so an identical upsert
    is a no-op and does not bump ``updated_at``.
    """
    batch = [rows] if isinstance(rows, Mapping) else list(rows)
    if not batch:
        return
    table: Table = model.__table__  # type: ignore[assignment]
    stmt = pg_insert(table).values(batch)
    skip = {*conflict, *keep, "created_at"}
    columns = [c for c in batch[0] if c not in skip]
    if not columns:
        session.execute(stmt.on_conflict_do_nothing(index_elements=list(conflict)))
        _expire(session, model)
        return
    changes: dict[str, Any] = {c: stmt.excluded[c] for c in columns}
    if "updated_at" in table.c:
        changes["updated_at"] = func.now()
    changed = or_(*(table.c[c].is_distinct_from(stmt.excluded[c]) for c in columns))
    session.execute(
        stmt.on_conflict_do_update(index_elements=list(conflict), set_=changes, where=changed)
    )
    _expire(session, model)


def _expire(session: Session, model: type[Base]) -> None:
    """Core writes bypass the ORM: reload ``model`` objects this session already holds."""
    for obj in list(session.identity_map.values()):
        if isinstance(obj, model):
            session.expire(obj)


def clean_text(text: str) -> str:
    """PostgreSQL ``TEXT`` cannot hold NUL bytes, which OCR output occasionally contains."""
    return text.replace("\x00", "")


# --------------------------------------------------------------------------- documents


def document_values(document: Document) -> dict[str, Any]:
    return {
        "id": document.id,
        "sha256": document.sha256,
        "filename": document.filename,
        "mime_type": document.mime_type,
        "doc_type": document.doc_type.value,
        "status": document.status.value,
        "error": document.error,
        "pages": document.pages,
        "ocr_engine": document.ocr_engine,
        "created_at": document.created_at,
    }


def to_document(row: DocumentRow) -> Document:
    return Document(
        id=row.id,
        sha256=row.sha256,
        filename=row.filename,
        mime_type=row.mime_type,
        doc_type=DocType(row.doc_type),
        status=DocumentStatus(row.status),
        error=row.error,
        pages=row.pages,
        ocr_engine=row.ocr_engine,  # type: ignore[arg-type]  # CHECKed in the database
        created_at=row.created_at,
    )


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert(self, document: Document) -> None:
        """Insert or update the document by ``id``; ``created_at`` keeps its first value."""
        upsert(self.session, DocumentRow, document_values(document))

    def claim(self, document: Document) -> str | None:
        """Insert ``document`` unless its ``sha256`` is already stored.

        Returns ``None`` if this call inserted it, else the ``id`` of the document that
        already has these bytes. The unique index on ``sha256`` makes check-and-insert
        atomic, so two concurrent uploads of the same file cannot both win.
        """
        table: Table = DocumentRow.__table__  # type: ignore[assignment]
        stmt = (
            pg_insert(table)
            .values(document_values(document))
            .on_conflict_do_nothing(index_elements=["sha256"])
            .returning(table.c.id)
        )
        if self.session.execute(stmt).scalar_one_or_none() is not None:
            return None
        existing = self.id_for_sha256(document.sha256)
        if existing is None:  # pragma: no cover - the conflicting row was deleted meanwhile
            raise RuntimeError(f"sha256 {document.sha256} conflicted but no row has it")
        return existing

    def get(self, document_id: str) -> Document | None:
        row = self.session.get(DocumentRow, document_id, populate_existing=True)
        return None if row is None else to_document(row)

    def get_by_sha256(self, sha256: str) -> Document | None:
        row = self.session.scalars(
            select(DocumentRow).where(DocumentRow.sha256 == sha256.lower())
        ).one_or_none()
        return None if row is None else to_document(row)

    def id_for_sha256(self, sha256: str) -> str | None:
        return self.session.scalars(
            select(DocumentRow.id).where(DocumentRow.sha256 == sha256.lower())
        ).one_or_none()

    def set_status(
        self, document_id: str, status: DocumentStatus, *, error: str | None = None
    ) -> bool:
        """Move a document to ``status`` (``error`` is kept only for ``failed``).

        Returns whether the document exists.
        """
        stmt = (
            update(DocumentRow)
            .where(DocumentRow.id == document_id)
            .values(
                status=status.value,
                error=error if status is DocumentStatus.FAILED else None,
                updated_at=func.now(),
            )
        )
        return bool(self.session.execute(stmt).rowcount)  # type: ignore[attr-defined]

    def list_by_status(self, *statuses: DocumentStatus) -> list[Document]:
        rows = self.session.scalars(
            select(DocumentRow)
            .where(DocumentRow.status.in_([s.value for s in statuses]))
            .order_by(DocumentRow.created_at)
        )
        return [to_document(row) for row in rows]

    def delete(self, document_id: str) -> bool:
        """Delete a document; its chunks and extracted records cascade with it."""
        result = self.session.execute(delete(DocumentRow).where(DocumentRow.id == document_id))
        return bool(result.rowcount)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- chunks


class ChunkRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def replace(
        self,
        document_id: str,
        chunks: Sequence[Chunk],
        *,
        token_counts: Sequence[int | None] | None = None,
    ) -> int:
        """Make ``chunks`` the document's chunks: upsert each by its vector id and delete
        any left over from a previous, longer version. Returns how many there are now."""
        if token_counts is not None and len(token_counts) != len(chunks):
            raise ValueError("token_counts must have one entry per chunk")
        counts = token_counts if token_counts is not None else [None] * len(chunks)
        upsert(
            self.session,
            ChunkRow,
            [
                {
                    "id": chunk.vector_id(document_id),
                    "document_id": document_id,
                    "chunk_index": chunk.index,
                    "kind": chunk.kind,
                    "title": chunk.title,
                    "text": clean_text(chunk.text),
                    "token_count": count,
                    "pages": list(chunk.pages),
                }
                for chunk, count in zip(chunks, counts, strict=True)
            ],
        )
        keep = [chunk.vector_id(document_id) for chunk in chunks]
        self.session.execute(
            delete(ChunkRow).where(ChunkRow.document_id == document_id, ChunkRow.id.not_in(keep))
        )
        return len(chunks)

    def for_document(self, document_id: str) -> list[ChunkRow]:
        return list(
            self.session.scalars(
                select(ChunkRow)
                .where(ChunkRow.document_id == document_id)
                .order_by(ChunkRow.chunk_index)
            )
        )

    def texts(self, ids: Iterable[str]) -> dict[str, str]:
        """Full text by vector id -- Pinecone hits carry only a truncated copy."""
        wanted = list(dict.fromkeys(ids))
        if not wanted:
            return {}
        rows = self.session.execute(
            select(ChunkRow.id, ChunkRow.text).where(ChunkRow.id.in_(wanted))
        )
        return {row.id: row.text for row in rows}


# --------------------------------------------------------------------------- line items


def sync_lines(
    session: Session,
    model: type[InvoiceLineItemRow] | type[POLineItemRow],
    parent: str,
    parent_id: str,
    lines: Sequence[LineItem],
) -> None:
    """Make ``lines`` the parent's line items, keyed by position (``line_no``)."""
    upsert(
        session,
        model,
        [
            {
                parent: parent_id,
                "line_no": n,
                "description": line.description,
                "quantity": line.quantity,
                "unit_price": line.unit_price,
                "amount": line.amount,
            }
            for n, line in enumerate(lines)
        ],
        conflict=(parent, "line_no"),
    )
    table: Table = model.__table__  # type: ignore[assignment]
    session.execute(
        delete(table).where(table.c[parent] == parent_id, table.c.line_no >= len(lines))
    )


# --------------------------------------------------------------------------- vendors


class VendorRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert(
        self,
        vendor_id: str,
        canonical_name: str,
        *,
        aliases: Sequence[str] = (),
        tax_id: str | None = None,
        email: str | None = None,
    ) -> None:
        upsert(
            self.session,
            VendorRow,
            {
                "id": vendor_id,
                "canonical_name": canonical_name,
                "aliases": list(dict.fromkeys(aliases)),
                "tax_id": tax_id,
                "email": email,
            },
        )

    def ensure(self, vendor_id: str, name: str, *, tax_id: str | None = None) -> bool:
        """Insert the vendor unless its id exists (an existing row is left as it is).

        The tax id is only stored if no other vendor has it. Returns whether it was new.
        """
        if tax_id is not None:
            owner = self.get_by_tax_id(tax_id)
            if owner is not None and owner.id != vendor_id:
                tax_id = None
        table: Table = VendorRow.__table__  # type: ignore[assignment]
        stmt = (
            pg_insert(table)
            .values(id=vendor_id, canonical_name=name, tax_id=tax_id)
            .on_conflict_do_nothing(index_elements=["id"])
            .returning(table.c.id)
        )
        return self.session.execute(stmt).scalar_one_or_none() is not None

    def add_alias(self, vendor_id: str, alias: str) -> bool:
        """Remember another spelling of a vendor's name. Returns whether it was new."""
        row = self.get(vendor_id)
        if row is None:
            raise KeyError(f"unknown vendor {vendor_id!r}")
        if alias == row.canonical_name or alias in row.aliases:
            return False
        stmt = (
            update(VendorRow)
            .where(VendorRow.id == vendor_id, ~VendorRow.aliases.contains([alias]))
            .values(aliases=func.array_append(VendorRow.aliases, alias), updated_at=func.now())
        )
        self.session.execute(stmt)
        self.session.expire(row)
        return True

    def get(self, vendor_id: str) -> VendorRow | None:
        return self.session.get(VendorRow, vendor_id, populate_existing=True)

    def get_by_tax_id(self, tax_id: str) -> VendorRow | None:
        return self.session.scalars(
            select(VendorRow).where(VendorRow.tax_id == tax_id)
        ).one_or_none()

    def all(self) -> list[VendorRow]:
        return list(self.session.scalars(select(VendorRow).order_by(VendorRow.id)))


# --------------------------------------------------------------------------- records


def _amounts(record: Invoice | PurchaseOrder) -> dict[str, Any]:
    return {
        "currency": record.currency,
        "subtotal": record.subtotal,
        "tax": record.tax,
        "total": record.total,
    }


@dataclass(frozen=True)
class OverdueInvoice:
    invoice: InvoiceRow
    outstanding: Decimal
    days_overdue: int


class InvoiceRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert(
        self,
        record: Invoice,
        *,
        id: str,
        document_id: str | None = None,
        vendor_id: str | None = None,
        status: str | None = None,
    ) -> None:
        """Write the invoice and its line items.

        ``status`` is workflow state: it is set when given (or to ``unpaid`` on first
        insert) and otherwise left as it is, so re-ingesting a paid invoice keeps it paid.
        """
        values = {
            "id": id,
            "document_id": document_id,
            "vendor_id": vendor_id,
            "vendor_name": record.vendor_name,
            "vendor_tax_id": record.vendor_tax_id,
            "customer_name": record.customer_name,
            "invoice_number": record.invoice_number,
            "invoice_date": record.invoice_date,
            "due_date": record.due_date,
            "po_number": record.po_number,
            "status": status or "unpaid",
            **_amounts(record),
        }
        upsert(self.session, InvoiceRow, values, keep=() if status else ("status",))
        sync_lines(self.session, InvoiceLineItemRow, "invoice_id", id, record.line_items)

    def upsert_receipt(
        self,
        record: Receipt,
        *,
        id: str,
        document_id: str | None = None,
        vendor_id: str | None = None,
    ) -> None:
        """A till receipt is an invoice that was paid on the spot: stored as a ``paid``
        invoice due on its date, numbered by its receipt number when it has one."""
        self.upsert(
            record.as_invoice(id),
            id=id,
            document_id=document_id,
            vendor_id=vendor_id,
            status="paid",
        )

    def get(self, invoice_id: str) -> InvoiceRow | None:
        return self.session.get(InvoiceRow, invoice_id, populate_existing=True)

    def find_by_number(
        self, invoice_number: str, *, vendor_id: str | None = None
    ) -> list[InvoiceRow]:
        """Invoices printed with ``invoice_number`` (numbers are only unique per vendor)."""
        stmt = select(InvoiceRow).where(InvoiceRow.invoice_number == invoice_number)
        if vendor_id is not None:
            stmt = stmt.where(InvoiceRow.vendor_id == vendor_id)
        return list(self.session.scalars(stmt.order_by(InvoiceRow.invoice_date, InvoiceRow.id)))

    def paid_amounts(self, invoice_ids: Iterable[str]) -> dict[str, Decimal]:
        """Sum of payment allocations matched to each invoice."""
        wanted = list(dict.fromkeys(invoice_ids))
        if not wanted:
            return {}
        rows = self.session.execute(
            select(PaymentAllocationRow.invoice_id, func.sum(PaymentAllocationRow.amount))
            .where(PaymentAllocationRow.invoice_id.in_(wanted))
            .group_by(PaymentAllocationRow.invoice_id)
        )
        paid = {invoice_id: Decimal(total) for invoice_id, total in rows}
        return {invoice_id: paid.get(invoice_id, Decimal(0)) for invoice_id in wanted}

    def refresh_status(self, invoice_ids: Iterable[str]) -> None:
        """Set ``unpaid`` / ``partially_paid`` / ``paid`` from the matched payments.

        ``disputed`` and ``void`` are decisions, not arithmetic, and are left alone.
        """
        paid = self.paid_amounts(invoice_ids)
        for invoice_id, amount in paid.items():
            row = self.get(invoice_id)
            if row is None or row.status in ("disputed", "void"):
                continue
            if amount >= row.total - ROUNDING_TOLERANCE:
                status = "paid"
            elif amount > 0:
                status = "partially_paid"
            else:
                status = "unpaid"
            if row.status != status:
                row.status = status
        self.session.flush()

    def list_overdue(
        self, as_of: date, *, vendor_id: str | None = None, limit: int | None = None
    ) -> list[OverdueInvoice]:
        """Invoices past their due date on ``as_of`` with money still owed, oldest first."""
        paid = (
            select(
                PaymentAllocationRow.invoice_id,
                func.sum(PaymentAllocationRow.amount).label("paid"),
            )
            .group_by(PaymentAllocationRow.invoice_id)
            .subquery()
        )
        outstanding = (InvoiceRow.total - func.coalesce(paid.c.paid, 0)).label("outstanding")
        stmt = (
            select(InvoiceRow, outstanding)
            .outerjoin(paid, paid.c.invoice_id == InvoiceRow.id)
            .where(
                InvoiceRow.due_date < as_of,
                InvoiceRow.status.not_in(("paid", "void")),
                outstanding > ROUNDING_TOLERANCE,
            )
            .order_by(InvoiceRow.due_date, InvoiceRow.id)
        )
        if vendor_id is not None:
            stmt = stmt.where(InvoiceRow.vendor_id == vendor_id)
        if limit is not None:
            stmt = stmt.limit(limit)
        return [
            OverdueInvoice(invoice=row, outstanding=Decimal(owed), days_overdue=_days(row, as_of))
            for row, owed in self.session.execute(stmt)
        ]


def _days(row: InvoiceRow, as_of: date) -> int:
    assert row.due_date is not None  # filtered in the query
    return (as_of - row.due_date).days


class ContractRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert(
        self,
        record: Contract,
        *,
        id: str,
        document_id: str | None = None,
        vendor_id: str | None = None,
    ) -> None:
        upsert(
            self.session,
            ContractRow,
            {
                "id": id,
                "document_id": document_id,
                "vendor_id": vendor_id,
                "vendor_name": record.vendor_name,
                "customer_name": record.customer_name,
                "contract_number": record.contract_number,
                "title": record.title,
                "start_date": record.start_date,
                "end_date": record.end_date,
                "payment_terms_days": record.payment_terms_days,
                "currency": record.currency,
                "rate_card": [entry.model_dump(mode="json") for entry in record.rate_card],
            },
        )

    def get(self, contract_id: str) -> ContractRow | None:
        return self.session.get(ContractRow, contract_id, populate_existing=True)

    def find_by_number(
        self, contract_number: str, *, vendor_id: str | None = None
    ) -> list[ContractRow]:
        stmt = select(ContractRow).where(ContractRow.contract_number == contract_number)
        if vendor_id is not None:
            stmt = stmt.where(ContractRow.vendor_id == vendor_id)
        return list(self.session.scalars(stmt.order_by(ContractRow.start_date, ContractRow.id)))


class PurchaseOrderRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert(
        self,
        record: PurchaseOrder,
        *,
        id: str,
        document_id: str | None = None,
        vendor_id: str | None = None,
        contract_id: str | None = None,
    ) -> None:
        """Write the PO and its lines. Without ``contract_id``, the printed contract number
        is looked up (this vendor's contract first)."""
        if contract_id is None and record.contract_number:
            contract_id = _pick(
                ContractRepository(self.session).find_by_number(record.contract_number),
                vendor_id,
            )
        upsert(
            self.session,
            PurchaseOrderRow,
            {
                "id": id,
                "document_id": document_id,
                "vendor_id": vendor_id,
                "vendor_name": record.vendor_name,
                "vendor_tax_id": record.vendor_tax_id,
                "buyer_name": record.buyer_name,
                "po_number": record.po_number,
                "order_date": record.order_date,
                "delivery_date": record.delivery_date,
                "contract_id": contract_id,
                "contract_number": record.contract_number,
                **_amounts(record),
            },
        )
        sync_lines(self.session, POLineItemRow, "purchase_order_id", id, record.line_items)

    def get(self, purchase_order_id: str) -> PurchaseOrderRow | None:
        return self.session.get(PurchaseOrderRow, purchase_order_id, populate_existing=True)

    def find_by_number(
        self, po_number: str, *, vendor_id: str | None = None
    ) -> list[PurchaseOrderRow]:
        stmt = select(PurchaseOrderRow).where(PurchaseOrderRow.po_number == po_number)
        if vendor_id is not None:
            stmt = stmt.where(PurchaseOrderRow.vendor_id == vendor_id)
        return list(
            self.session.scalars(stmt.order_by(PurchaseOrderRow.order_date, PurchaseOrderRow.id))
        )


class PaymentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert(
        self,
        record: Payment,
        *,
        id: str,
        document_id: str | None = None,
        vendor_id: str | None = None,
    ) -> None:
        """Write the payment and its allocations, match each allocation's invoice number
        to an invoice (this vendor's first), and refresh the affected invoices' status."""
        invoices = InvoiceRepository(self.session)
        before = {
            invoice_id
            for invoice_id in self.session.scalars(
                select(PaymentAllocationRow.invoice_id).where(PaymentAllocationRow.payment_id == id)
            )
            if invoice_id is not None
        }
        upsert(
            self.session,
            PaymentRow,
            {
                "id": id,
                "document_id": document_id,
                "vendor_id": vendor_id,
                "vendor_name": record.vendor_name,
                "payer_name": record.payer_name,
                "payment_reference": record.payment_reference,
                "paid_at": record.paid_at,
                "method": record.method,
                "currency": record.currency,
                "amount": record.amount,
            },
        )
        matched = {
            a.invoice_number: _pick(invoices.find_by_number(a.invoice_number), vendor_id)
            for a in record.allocations
        }
        allocations = [
            {
                "payment_id": id,
                "invoice_number": a.invoice_number,
                "invoice_id": matched[a.invoice_number],
                "amount": a.amount,
            }
            for a in record.allocations
        ]
        upsert(
            self.session,
            PaymentAllocationRow,
            allocations,
            conflict=("payment_id", "invoice_number"),
        )
        numbers = [a.invoice_number for a in record.allocations]
        self.session.execute(
            delete(PaymentAllocationRow).where(
                PaymentAllocationRow.payment_id == id,
                PaymentAllocationRow.invoice_number.not_in(numbers),
            )
        )
        after = {invoice_id for invoice_id in matched.values() if invoice_id is not None}
        invoices.refresh_status(before | after)

    def get(self, payment_id: str) -> PaymentRow | None:
        return self.session.get(PaymentRow, payment_id, populate_existing=True)

    def find_by_reference(self, payment_reference: str) -> list[PaymentRow]:
        return list(
            self.session.scalars(
                select(PaymentRow)
                .where(PaymentRow.payment_reference == payment_reference)
                .order_by(PaymentRow.paid_at, PaymentRow.id)
            )
        )


class AnomalyScoreRepository:
    """One score per invoice and model version (re-scoring replaces it)."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert(
        self,
        invoice_id: str,
        model_version: str,
        score: float,
        reasons: Sequence[Mapping[str, Any]],
    ) -> None:
        upsert(
            self.session,
            AnomalyScoreRow,
            {
                "invoice_id": invoice_id,
                "model_version": model_version,
                "score": score,
                "reasons": list(reasons),
            },
            conflict=("invoice_id", "model_version"),
        )

    def for_invoice(self, invoice_id: str) -> list[AnomalyScoreRow]:
        """Every version's score, newest first."""
        stmt = (
            select(AnomalyScoreRow)
            .where(AnomalyScoreRow.invoice_id == invoice_id)
            .order_by(AnomalyScoreRow.created_at.desc(), AnomalyScoreRow.id.desc())
        )
        return list(self.session.scalars(stmt))

    def unscored(self, model_version: str) -> list[str]:
        """Invoices with no score from ``model_version``, oldest first."""
        scored = select(AnomalyScoreRow.invoice_id).where(
            AnomalyScoreRow.model_version == model_version
        )
        stmt = (
            select(InvoiceRow.id)
            .where(InvoiceRow.id.not_in(scored))
            .order_by(InvoiceRow.created_at, InvoiceRow.id)
        )
        return list(self.session.scalars(stmt))


def _pick(rows: Sequence[InvoiceRow] | Sequence[ContractRow], vendor_id: str | None) -> str | None:
    """The id of the one candidate, preferring ``vendor_id``'s; ``None`` if ambiguous."""
    if vendor_id is not None:
        own = [row for row in rows if row.vendor_id == vendor_id]
        if len(own) == 1:
            return own[0].id
    return rows[0].id if len(rows) == 1 else None


# --------------------------------------------------------------------------- rows -> records
#
# The inverse of the upserts above, for rebuilding the graph or vectors from Postgres.


def to_line_items(rows: Iterable[InvoiceLineItemRow | POLineItemRow]) -> list[LineItem]:
    return [
        LineItem(
            description=row.description,
            quantity=row.quantity,
            unit_price=row.unit_price,
            amount=row.amount,
        )
        for row in rows
    ]


def to_invoice(row: InvoiceRow) -> Invoice:
    return Invoice(
        invoice_number=row.invoice_number,
        vendor_name=row.vendor_name,
        vendor_tax_id=row.vendor_tax_id,
        customer_name=row.customer_name,
        invoice_date=row.invoice_date,
        due_date=row.due_date,
        po_number=row.po_number,
        currency=row.currency,
        subtotal=row.subtotal,
        tax=row.tax,
        total=row.total,
        line_items=to_line_items(row.line_items),
    )


def to_receipt(row: InvoiceRow) -> Receipt:
    """An invoice row written by ``upsert_receipt`` back as the receipt it came from
    (its invoice number is the receipt number, or the ``RECEIPT-...`` stand-in)."""
    return Receipt(
        vendor_name=row.vendor_name,
        receipt_number=row.invoice_number,
        receipt_date=row.invoice_date,
        currency=row.currency,
        tax=row.tax,
        total=row.total,
        line_items=to_line_items(row.line_items),
    )


def to_purchase_order(row: PurchaseOrderRow) -> PurchaseOrder:
    return PurchaseOrder(
        po_number=row.po_number,
        vendor_name=row.vendor_name,
        vendor_tax_id=row.vendor_tax_id,
        buyer_name=row.buyer_name,
        order_date=row.order_date,
        delivery_date=row.delivery_date,
        contract_number=row.contract_number,
        currency=row.currency,
        subtotal=row.subtotal,
        tax=row.tax,
        total=row.total,
        line_items=to_line_items(row.line_items),
    )


def to_payment(row: PaymentRow) -> Payment:
    return Payment(
        payment_reference=row.payment_reference,
        payer_name=row.payer_name,
        vendor_name=row.vendor_name,
        paid_at=row.paid_at,
        method=row.method,  # type: ignore[arg-type]  # CHECKed in the database
        currency=row.currency,
        amount=row.amount,
        allocations=[
            PaymentAllocation(invoice_number=a.invoice_number, amount=a.amount)
            for a in row.allocations
        ],
    )


def to_contract(row: ContractRow) -> Contract:
    return Contract(
        contract_number=row.contract_number,
        title=row.title,
        vendor_name=row.vendor_name,
        customer_name=row.customer_name,
        start_date=row.start_date,
        end_date=row.end_date,
        payment_terms_days=row.payment_terms_days,
        currency=row.currency,
        rate_card=[RateCardEntry.model_validate(entry) for entry in row.rate_card],
    )


# --------------------------------------------------------------------------- facade

RECORD_MODELS: tuple[type[Base], ...] = (InvoiceRow, PurchaseOrderRow, PaymentRow, ContractRow)


def record_model(record: ExtractedRecord) -> type[Base]:
    """The table a record is stored in (receipts are invoices)."""
    if isinstance(record, Invoice | Receipt):
        return InvoiceRow
    if isinstance(record, PurchaseOrder):
        return PurchaseOrderRow
    if isinstance(record, Payment):
        return PaymentRow
    return ContractRow


class Repositories:
    """Every repository on one session, plus :meth:`upsert_record` for any record type."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.documents = DocumentRepository(session)
        self.chunks = ChunkRepository(session)
        self.vendors = VendorRepository(session)
        self.invoices = InvoiceRepository(session)
        self.purchase_orders = PurchaseOrderRepository(session)
        self.payments = PaymentRepository(session)
        self.contracts = ContractRepository(session)
        self.anomaly_scores = AnomalyScoreRepository(session)

    def delete_records(self, document_id: str, *, keep: type[Base] | None = None) -> int:
        """Delete the business records a document produced, except in table ``keep`` --
        for a re-ingest whose extraction changed the document type. Returns how many."""
        removed = 0
        for model in RECORD_MODELS:
            if model is keep:
                continue
            table: Table = model.__table__  # type: ignore[assignment]
            result = self.session.execute(delete(table).where(table.c.document_id == document_id))
            removed += result.rowcount  # type: ignore[attr-defined]
        return removed

    def upsert_record(
        self,
        record: ExtractedRecord,
        *,
        id: str,
        document_id: str | None = None,
        vendor_id: str | None = None,
    ) -> None:
        """Write an extracted record to its table(s)."""
        if isinstance(record, Invoice):
            self.invoices.upsert(record, id=id, document_id=document_id, vendor_id=vendor_id)
        elif isinstance(record, Receipt):
            self.invoices.upsert_receipt(
                record, id=id, document_id=document_id, vendor_id=vendor_id
            )
        elif isinstance(record, PurchaseOrder):
            self.purchase_orders.upsert(record, id=id, document_id=document_id, vendor_id=vendor_id)
        elif isinstance(record, Payment):
            self.payments.upsert(record, id=id, document_id=document_id, vendor_id=vendor_id)
        elif isinstance(record, Contract):
            self.contracts.upsert(record, id=id, document_id=document_id, vendor_id=vendor_id)
        else:  # pragma: no cover - ExtractedRecord is a closed union
            raise TypeError(f"no table for {type(record).__name__}")
