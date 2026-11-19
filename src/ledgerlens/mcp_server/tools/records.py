"""Typed access to the ledger's records: agents read and write through these tools and
never query the databases themselves.

``save_extraction`` is the one tool that writes: a validated record goes through the same
record write path as ingestion (:meth:`~ledgerlens.ingestion.sinks.StoresSink.write_record`).

Every tool returns a pydantic model, so MCP clients get a JSON schema for the output and
JSON values back (amounts as exact decimal strings, dates as ISO strings). A record is
looked up by its id or by the number printed on it (``INV-889``, ``PO-1042``); numbers are
only unique per vendor, so an ambiguous number is an error that lists the candidates,
and ``vendor`` (an id or a name) narrows it down. Records that came from a document carry
its id and the citation tag answers use (``doc:INV-889``).
"""

from __future__ import annotations

import re
from dataclasses import asdict
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeVar

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import AfterValidator, BaseModel, Field

from ledgerlens.anomaly.scorer import RULE_BAND
from ledgerlens.core.schemas import ExtractedRecord
from ledgerlens.mcp_server.services import Services
from ledgerlens.mcp_server.tools import READ_ONLY, in_thread, jsonable
from ledgerlens.mcp_server.tools.search import resolve_vendor

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from ledgerlens.storage.postgres.models import (
        ContractRow,
        InvoiceLineItemRow,
        InvoiceRow,
        POLineItemRow,
        PurchaseOrderRow,
    )

Row = TypeVar("Row", "InvoiceRow", "PurchaseOrderRow", "ContractRow")
RecordKind = Literal["invoice", "purchase order", "contract"]


CENTS = Decimal("0.01")


def tidy_amount(value: Decimal) -> Decimal:
    """``783.0600`` (as Postgres returns it) -> ``783.06``; extra digits that mean
    something are kept."""
    cents = value.quantize(CENTS)
    return cents if cents == value else value.normalize()


Amount = Annotated[Decimal, AfterValidator(tidy_amount)]
"""A money amount: serialized as an exact decimal string, to the cent when that is exact."""


def tidy_quantity(value: Decimal) -> Decimal:
    """``17.0000`` -> ``17``, ``2.5000`` -> ``2.5`` (never ``1E+2``)."""
    whole = value.quantize(Decimal(1))
    return whole if whole == value else value.normalize()


Quantity = Annotated[Decimal, AfterValidator(tidy_quantity)]


def citation(number: str | None, document_id: str | None) -> str | None:
    """The tag an answer cites a record's document by (``None`` without a document)."""
    if not document_id:
        return None
    return f"doc:{number or document_id[:8]}"


class Line(BaseModel):
    description: str
    quantity: Quantity | None = None
    unit_price: Amount | None = None
    amount: Amount


def _lines(rows: list[InvoiceLineItemRow] | list[POLineItemRow]) -> list[Line]:
    return [
        Line(
            description=row.description,
            quantity=row.quantity,
            unit_price=row.unit_price,
            amount=row.amount,
        )
        for row in rows
    ]


class InvoiceRecord(BaseModel):
    id: str
    document_id: str | None
    citation: str | None
    vendor_id: str | None
    vendor_name: str
    vendor_tax_id: str | None
    invoice_number: str
    invoice_date: date
    due_date: date | None
    po_number: str | None
    currency: str
    subtotal: Amount | None
    tax: Amount | None
    total: Amount
    status: str
    """``unpaid``, ``partially_paid``, ``paid``, ``disputed`` or ``void``."""
    paid: Amount
    """Sum of the payments matched to it."""
    outstanding: Amount
    line_items: list[Line] = Field(default_factory=list)


class PurchaseOrderRecord(BaseModel):
    id: str
    document_id: str | None
    citation: str | None
    vendor_id: str | None
    vendor_name: str
    po_number: str
    order_date: date
    delivery_date: date | None
    contract_number: str | None
    currency: str
    subtotal: Amount | None
    tax: Amount | None
    total: Amount
    invoiced: Amount
    """What the invoices citing this PO add up to."""
    invoice_ids: list[str] = Field(default_factory=list)
    line_items: list[Line] = Field(default_factory=list)


class RateCardEntry(BaseModel):
    item: str
    unit: str | None = None
    unit_price: Amount


class ContractRecord(BaseModel):
    id: str
    document_id: str | None
    citation: str | None
    vendor_id: str | None
    vendor_name: str
    contract_number: str | None
    title: str | None
    start_date: date
    end_date: date | None
    payment_terms_days: int | None
    currency: str
    rate_card: list[RateCardEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- lookups


def _pick(rows: list[Row], kind: RecordKind, given: str) -> Row:
    if not rows:
        raise ToolError(f"no {kind} {given!r}")
    if len(rows) > 1:
        options = ", ".join(f"{row.id} ({row.vendor_name})" for row in rows[:10])
        raise ToolError(
            f"{len(rows)} {kind}s are numbered {given!r}: {options}. "
            "Give the id, or the vendor to narrow it down."
        )
    return rows[0]


def _vendor_id(services: Services, vendor: str | None) -> str | None:
    if not vendor:
        return None
    ids = resolve_vendor(services, vendor)
    if len(ids) > 1:
        raise ToolError(f"{vendor!r} matches several vendors: {', '.join(ids)}")
    return ids[0]


def find_invoice(
    session: Session, services: Services, invoice: str, vendor: str | None = None
) -> InvoiceRow:
    from ledgerlens.storage.postgres.repositories import InvoiceRepository

    repo = InvoiceRepository(session)
    row = repo.get(invoice)
    if row is not None:
        return row
    return _pick(
        repo.find_by_number(invoice, vendor_id=_vendor_id(services, vendor)), "invoice", invoice
    )


def find_po(
    session: Session, services: Services, po: str, vendor: str | None = None
) -> PurchaseOrderRow:
    from ledgerlens.storage.postgres.repositories import PurchaseOrderRepository

    repo = PurchaseOrderRepository(session)
    row = repo.get(po)
    if row is not None:
        return row
    rows = repo.find_by_number(po, vendor_id=_vendor_id(services, vendor))
    return _pick(rows, "purchase order", po)


def find_contract(
    session: Session, services: Services, contract: str, vendor: str | None = None
) -> ContractRow:
    from ledgerlens.storage.postgres.repositories import ContractRepository

    repo = ContractRepository(session)
    row = repo.get(contract)
    if row is not None:
        return row
    rows = repo.find_by_number(contract, vendor_id=_vendor_id(services, vendor))
    return _pick(rows, "contract", contract)


# --------------------------------------------------------------------------- records


def invoice_record(session: Session, row: InvoiceRow) -> InvoiceRecord:
    from ledgerlens.storage.postgres.repositories import InvoiceRepository

    paid = InvoiceRepository(session).paid_amounts([row.id])[row.id]
    return InvoiceRecord(
        id=row.id,
        document_id=row.document_id,
        citation=citation(row.invoice_number, row.document_id),
        vendor_id=row.vendor_id,
        vendor_name=row.vendor_name,
        vendor_tax_id=row.vendor_tax_id,
        invoice_number=row.invoice_number,
        invoice_date=row.invoice_date,
        due_date=row.due_date,
        po_number=row.po_number,
        currency=row.currency,
        subtotal=row.subtotal,
        tax=row.tax,
        total=row.total,
        status=row.status,
        paid=paid,
        outstanding=row.total - paid,
        line_items=_lines(row.line_items),
    )


def get_invoice(services: Services, invoice: str, vendor: str | None = None) -> InvoiceRecord:
    with services.sessions() as session:
        return invoice_record(session, find_invoice(session, services, invoice, vendor))


def get_po(services: Services, po: str, vendor: str | None = None) -> PurchaseOrderRecord:
    from sqlalchemy import select

    from ledgerlens.storage.postgres.models import InvoiceRow

    with services.sessions() as session:
        row = find_po(session, services, po, vendor)
        invoices = list(
            session.execute(
                select(InvoiceRow.id, InvoiceRow.total)
                .where(InvoiceRow.po_number == row.po_number, InvoiceRow.currency == row.currency)
                .where(
                    InvoiceRow.vendor_id == row.vendor_id
                    if row.vendor_id is not None
                    else InvoiceRow.vendor_id.is_(None)
                )
                .order_by(InvoiceRow.invoice_date, InvoiceRow.id)
            )
        )
        return PurchaseOrderRecord(
            id=row.id,
            document_id=row.document_id,
            citation=citation(row.po_number, row.document_id),
            vendor_id=row.vendor_id,
            vendor_name=row.vendor_name,
            po_number=row.po_number,
            order_date=row.order_date,
            delivery_date=row.delivery_date,
            contract_number=row.contract_number,
            currency=row.currency,
            subtotal=row.subtotal,
            tax=row.tax,
            total=row.total,
            invoiced=sum((Decimal(total) for _, total in invoices), Decimal(0)),
            invoice_ids=[invoice_id for invoice_id, _ in invoices],
            line_items=_lines(row.line_items),
        )


def get_contract(services: Services, contract: str, vendor: str | None = None) -> ContractRecord:
    with services.sessions() as session:
        row = find_contract(session, services, contract, vendor)
        return ContractRecord(
            id=row.id,
            document_id=row.document_id,
            citation=citation(row.contract_number, row.document_id),
            vendor_id=row.vendor_id,
            vendor_name=row.vendor_name,
            contract_number=row.contract_number,
            title=row.title,
            start_date=row.start_date,
            end_date=row.end_date,
            payment_terms_days=row.payment_terms_days,
            currency=row.currency,
            rate_card=[RateCardEntry.model_validate(entry) for entry in row.rate_card],
        )


# --------------------------------------------------------------------------- payments


class PaymentApplied(BaseModel):
    """A payment, and the part of it applied to one invoice."""

    payment_id: str
    payment_reference: str
    document_id: str | None
    citation: str | None
    paid_at: date
    method: str | None
    currency: str
    amount: Amount
    """The whole payment (it can settle several invoices)."""
    applied: Amount
    """The part applied to this invoice."""


class InvoicePayments(BaseModel):
    invoice_id: str
    invoice_number: str
    currency: str
    total: Amount
    paid: Amount
    outstanding: Amount
    status: str
    due_date: date | None
    payments: list[PaymentApplied] = Field(default_factory=list)


def get_payments(services: Services, invoice: str, vendor: str | None = None) -> InvoicePayments:
    from sqlalchemy import select

    from ledgerlens.storage.postgres.models import PaymentAllocationRow, PaymentRow

    with services.sessions() as session:
        row = find_invoice(session, services, invoice, vendor)
        applied = session.execute(
            select(PaymentRow, PaymentAllocationRow.amount)
            .join(PaymentAllocationRow, PaymentAllocationRow.payment_id == PaymentRow.id)
            .where(PaymentAllocationRow.invoice_id == row.id)
            .order_by(PaymentRow.paid_at, PaymentRow.id)
        ).all()
        payments = [
            PaymentApplied(
                payment_id=payment.id,
                payment_reference=payment.payment_reference,
                document_id=payment.document_id,
                citation=citation(payment.payment_reference, payment.document_id),
                paid_at=payment.paid_at,
                method=payment.method,
                currency=payment.currency,
                amount=payment.amount,
                applied=amount,
            )
            for payment, amount in applied
        ]
        paid = sum((p.applied for p in payments), Decimal(0))
        return InvoicePayments(
            invoice_id=row.id,
            invoice_number=row.invoice_number,
            currency=row.currency,
            total=row.total,
            paid=paid,
            outstanding=row.total - paid,
            status=row.status,
            due_date=row.due_date,
            payments=payments,
        )


class OverdueItem(BaseModel):
    invoice_id: str
    invoice_number: str
    document_id: str | None
    citation: str | None
    vendor_id: str | None
    vendor_name: str
    po_number: str | None
    due_date: date
    days_overdue: int
    currency: str
    total: Amount
    outstanding: Amount


class OverdueList(BaseModel):
    as_of: date
    min_days: int
    count: int
    """All overdue invoices matching, even past ``limit``."""
    outstanding_by_currency: dict[str, Amount] = Field(default_factory=dict)
    invoices: list[OverdueItem] = Field(default_factory=list)
    """Most overdue first, at most ``limit``."""


def list_overdue(
    services: Services,
    min_days: int = 1,
    vendor: str | None = None,
    *,
    as_of: date | None = None,
    limit: int = 50,
) -> OverdueList:
    from ledgerlens.storage.postgres.repositories import InvoiceRepository

    if min_days < 1:
        raise ToolError("min_days must be at least 1")
    day = as_of or services.today
    with services.sessions() as session:
        overdue = [
            item
            for item in InvoiceRepository(session).list_overdue(
                day, vendor_id=_vendor_id(services, vendor)
            )
            if item.days_overdue >= min_days
        ]
        by_currency: dict[str, Decimal] = {}
        for item in overdue:
            currency = item.invoice.currency
            by_currency[currency] = by_currency.get(currency, Decimal(0)) + item.outstanding
        return OverdueList(
            as_of=day,
            min_days=min_days,
            count=len(overdue),
            outstanding_by_currency=dict(sorted(by_currency.items())),
            invoices=[
                OverdueItem(
                    invoice_id=item.invoice.id,
                    invoice_number=item.invoice.invoice_number,
                    document_id=item.invoice.document_id,
                    citation=citation(item.invoice.invoice_number, item.invoice.document_id),
                    vendor_id=item.invoice.vendor_id,
                    vendor_name=item.invoice.vendor_name,
                    po_number=item.invoice.po_number,
                    due_date=item.invoice.due_date or day,
                    days_overdue=item.days_overdue,
                    currency=item.invoice.currency,
                    total=item.invoice.total,
                    outstanding=item.outstanding,
                )
                for item in overdue[:limit]  # oldest due date first = most overdue
            ],
        )


# --------------------------------------------------------------------------- analytics

Report = Literal["aging", "spend_by_vendor", "unpaid", "days_to_pay"]
REPORT_PARAMS: dict[str, tuple[str, ...]] = {
    "aging": ("as_of",),
    "spend_by_vendor": ("start", "end", "vendor"),
    "unpaid": ("as_of", "vendor"),
    "days_to_pay": ("start", "end", "vendor"),
}
"""The parameters each report takes (all optional: ``as_of`` defaults to today,
``start`` / ``end`` to unbounded, ``vendor`` to every vendor)."""
REPORT_DESCRIPTIONS = {
    "aging": "what is owed by days past due (current, 0-30, 31-60, 61-90, 90+), per currency",
    "spend_by_vendor": "invoiced spend per vendor per month",
    "unpaid": "per vendor: open invoices and outstanding, of which overdue, oldest due date",
    "days_to_pay": "per vendor: invoices paid, average and longest days to pay, days late",
}
MAX_ROWS = 500


class AnalyticsResult(BaseModel):
    report: Report
    params: dict[str, str | None] = Field(default_factory=dict)
    """The parameters used (defaults filled in)."""
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    total_rows: int
    truncated: bool = False


def _date_param(params: dict[str, Any], name: str) -> date | None:
    value = params.get(name)
    if value in (None, ""):
        return None
    try:
        return value if isinstance(value, date) else date.fromisoformat(str(value))
    except ValueError as exc:
        raise ToolError(f"{name} must be a date (YYYY-MM-DD), got {value!r}") from exc


def run_analytics(
    services: Services, report: str, params: dict[str, Any] | None = None, *, limit: int = 200
) -> AnalyticsResult:
    from ledgerlens.storage.postgres import analytics

    if report not in REPORT_PARAMS:
        raise ToolError(f"unknown report {report!r}; choose from {', '.join(REPORT_PARAMS)}")
    given = dict(params or {})
    unknown = sorted(set(given) - set(REPORT_PARAMS[report]))
    if unknown:
        allowed = ", ".join(REPORT_PARAMS[report])
        raise ToolError(f"{report} takes no {', '.join(unknown)} (it takes: {allowed})")
    as_of = _date_param(given, "as_of") or services.today
    start, end = _date_param(given, "start"), _date_param(given, "end")
    vendor_id = _vendor_id(services, given.get("vendor"))
    used: dict[str, str | None]
    with services.sessions() as session:
        rows: list[Any]
        if report == "aging":
            rows = analytics.ap_aging(session, as_of)
            used = {"as_of": as_of.isoformat()}
        elif report == "unpaid":
            rows = analytics.unpaid_totals(session, as_of)
            used = {"as_of": as_of.isoformat(), "vendor": vendor_id}
        elif report == "spend_by_vendor":
            rows = analytics.spend_by_vendor_month(session, start, end)
            used = {"start": _iso(start), "end": _iso(end), "vendor": vendor_id}
        else:
            rows = analytics.days_to_pay(session, start, end)
            used = {"start": _iso(start), "end": _iso(end), "vendor": vendor_id}
    if vendor_id is not None:
        rows = [row for row in rows if getattr(row, "vendor_id", None) == vendor_id]
    records = [jsonable(_rounded(asdict(row))) for row in rows]
    return AnalyticsResult(
        report=report,  # type: ignore[arg-type]  # checked against REPORT_PARAMS above
        params=used,
        columns=list(records[0]) if records else [],
        rows=records[:limit],
        total_rows=len(records),
        truncated=len(records) > limit,
    )


def _rounded(row: dict[str, Any]) -> dict[str, Any]:
    """Report decimals to the cent: amounts, and averages of days."""
    return {k: v.quantize(CENTS) if isinstance(v, Decimal) else v for k, v in row.items()}


def _iso(day: date | None) -> str | None:
    return day.isoformat() if day else None


# --------------------------------------------------------------------------- vendors, documents


class VendorContact(BaseModel):
    vendor_id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    tax_id: str | None = None
    email: str | None = None
    """``None``: no email on file; a reminder needs one added before it can be sent."""
    contract_number: str | None = None
    """The vendor's current contract (latest start date)."""
    payment_terms_days: int | None = None


def get_vendor_contact(services: Services, vendor: str) -> VendorContact:
    from sqlalchemy import select

    from ledgerlens.storage.postgres.models import ContractRow
    from ledgerlens.storage.postgres.repositories import VendorRepository

    vendor_id = _vendor_id(services, vendor)
    assert vendor_id is not None
    with services.sessions() as session:
        row = VendorRepository(session).get(vendor_id)
        if row is None:
            raise ToolError(f"vendor {vendor_id} is not stored")
        contract = session.scalars(
            select(ContractRow)
            .where(ContractRow.vendor_id == vendor_id)
            .order_by(ContractRow.start_date.desc(), ContractRow.id)
            .limit(1)
        ).first()
        return VendorContact(
            vendor_id=row.id,
            name=row.canonical_name,
            aliases=list(row.aliases),
            tax_id=row.tax_id,
            email=row.email,
            contract_number=contract.contract_number if contract else None,
            payment_terms_days=contract.payment_terms_days if contract else None,
        )


_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_CONTEXT_HEADER = re.compile(r"\A\[[^\]\n]*\]\n")
"""The one-line ``[INVOICE INV-889 | Vendor: ...]`` header chunking puts on each chunk."""
MAX_TEXT_CHARS = 50_000


class DocumentText(BaseModel):
    document_id: str
    filename: str
    doc_type: str
    status: str
    pages: int | None
    citation: str | None
    record_number: str | None
    """The number of the record extracted from it, if any."""
    text: str
    """The document's text, rebuilt from its stored chunks (context headers removed)."""
    truncated: bool = False
    chunk_count: int


def find_document_id(session: Session, document: str) -> str:
    """A document id as given, or the document of the record with that number."""
    from sqlalchemy import select

    from ledgerlens.storage.postgres.models import (
        ContractRow,
        InvoiceRow,
        PaymentRow,
        PurchaseOrderRow,
    )

    if _UUID.match(document.lower()):
        return document.lower()
    lookups = (
        (InvoiceRow.document_id, InvoiceRow.invoice_number),
        (PurchaseOrderRow.document_id, PurchaseOrderRow.po_number),
        (ContractRow.document_id, ContractRow.contract_number),
        (PaymentRow.document_id, PaymentRow.payment_reference),
    )
    found: list[str] = []
    for document_id, number in lookups:
        found += [
            str(d)
            for d in session.scalars(
                select(document_id).where(number == document, document_id.is_not(None))
            )
        ]
    found = list(dict.fromkeys(found))
    if not found:
        raise ToolError(f"no stored document {document!r} (give a document id or a number)")
    if len(found) > 1:
        raise ToolError(f"{len(found)} documents carry {document!r}: {', '.join(found)}")
    return found[0]


def _record_number(session: Session, document_id: str) -> str | None:
    from sqlalchemy import select

    from ledgerlens.storage.postgres.models import (
        ContractRow,
        InvoiceRow,
        PaymentRow,
        PurchaseOrderRow,
    )

    for number, owner in (
        (InvoiceRow.invoice_number, InvoiceRow.document_id),
        (PurchaseOrderRow.po_number, PurchaseOrderRow.document_id),
        (ContractRow.contract_number, ContractRow.document_id),
        (PaymentRow.payment_reference, PaymentRow.document_id),
    ):
        value = session.scalars(select(number).where(owner == document_id).limit(1)).first()
        if value:
            return str(value)
    return None


def get_document_text(
    services: Services, document: str, *, max_chars: int = 20_000
) -> DocumentText:
    from ledgerlens.storage.postgres.repositories import ChunkRepository, DocumentRepository

    with services.sessions() as session:
        document_id = find_document_id(session, document)
        stored = DocumentRepository(session).get(document_id)
        if stored is None:
            raise ToolError(f"no stored document {document_id}")
        chunks = ChunkRepository(session).for_document(document_id)
        number = _record_number(session, document_id)
    text = "\n\n".join(_CONTEXT_HEADER.sub("", chunk.text).strip() for chunk in chunks)
    return DocumentText(
        document_id=document_id,
        filename=stored.filename,
        doc_type=stored.doc_type.value,
        status=stored.status.value,
        pages=stored.pages,
        citation=citation(number, document_id),
        record_number=number,
        text=text[:max_chars],
        truncated=len(text) > max_chars,
        chunk_count=len(chunks),
    )


# --------------------------------------------------------------------------- save_extraction

FLAG_SCORE = RULE_BAND / 2
"""Combined anomaly scores at or above this are flagged (a hard hit scores >= 0.5, an
autoencoder flag >= 0.25: ``ledgerlens.anomaly.scorer``)."""

SAVE = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False
)
"""save_extraction writes, but saving the same record twice leaves the same state."""


class AnomalySummary(BaseModel):
    model_version: str
    score: float
    flagged: bool
    reasons: list[str] = Field(default_factory=list)


class SavedExtraction(BaseModel):
    document_id: str
    record_id: str
    doc_type: str
    record_number: str | None
    vendor_id: str | None
    vendor_match: str | None
    """How the vendor was resolved (``tax_id``, ``name``, ``fuzzy``, ``new``)."""
    citation: str | None
    anomaly: AnomalySummary | None = None
    """For an invoice: the anomaly score it got on saving."""


def existing_record_id(session: Session, document_id: str, record: ExtractedRecord) -> str | None:
    """The id of the record of ``record``'s type stored for the document, if any."""
    from sqlalchemy import select

    from ledgerlens.storage.postgres.repositories import record_model

    model: Any = record_model(record)
    return session.scalars(
        select(model.id).where(model.document_id == document_id).limit(1)
    ).first()


def _number(record: ExtractedRecord) -> str | None:
    for name in ("invoice_number", "po_number", "contract_number", "payment_reference"):
        value = getattr(record, name, None)
        if value:
            return str(value)
    return None


def save_extraction(
    services: Services, document_id: str, record: dict[str, Any]
) -> SavedExtraction:
    from pydantic import ValidationError

    from ledgerlens.core.schemas import parse_extracted
    from ledgerlens.ingestion.sinks import StoreWriteError
    from ledgerlens.storage.postgres.repositories import (
        AnomalyScoreRepository,
        DocumentRepository,
    )

    try:
        parsed = parse_extracted(record)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()[:8]
        )
        raise ToolError(f"the record does not validate: {problems}") from exc
    if not _UUID.match(document_id):
        raise ToolError(f"document_id must be a stored document's id (a UUID), not {document_id!r}")
    with services.sessions() as session:
        stored = DocumentRepository(session).get(document_id)
        if stored is None:
            raise ToolError(f"no stored document {document_id}")
        # the record this document already holds (bulk-loaded records keep their ledger ids)
        record_id = existing_record_id(session, document_id, parsed) or document_id
    try:
        vendor = services.record_sink.write_record(stored, parsed, record_id=record_id)
    except StoreWriteError as exc:
        raise ToolError(f"saving failed at {exc}") from exc
    anomaly = None
    if parsed.doc_type == "invoice":
        with services.sessions() as session:
            scores = AnomalyScoreRepository(session).for_invoice(record_id)
        if scores:
            row = scores[0]
            anomaly = AnomalySummary(
                model_version=row.model_version,
                score=round(row.score, 4),
                flagged=row.score >= FLAG_SCORE,
                reasons=[str(reason.get("text", "")) for reason in row.reasons],
            )
    number = _number(parsed)
    return SavedExtraction(
        document_id=document_id,
        record_id=record_id,
        doc_type=parsed.doc_type,
        record_number=number,
        vendor_id=vendor.vendor_id if vendor else None,
        vendor_match=vendor.method if vendor else None,
        citation=citation(number, document_id),
        anomaly=anomaly,
    )


# --------------------------------------------------------------------------- tools

Vendor = Annotated[
    str | None, Field(description="Narrow a number down to this vendor (an id or a name)")
]


def register(server: FastMCP, services: Services) -> None:
    @server.tool(name="get_invoice", annotations=READ_ONLY)
    async def get_invoice_tool(
        invoice: Annotated[str, Field(description="Invoice id, or the number printed on it")],
        vendor: Vendor = None,
    ) -> InvoiceRecord:
        """One invoice with its line items, payment status, amount paid and outstanding."""
        return await in_thread(get_invoice, services, invoice, vendor)

    @server.tool(name="get_po", annotations=READ_ONLY)
    async def get_po_tool(
        po: Annotated[str, Field(description="Purchase order id, or its PO number")],
        vendor: Vendor = None,
    ) -> PurchaseOrderRecord:
        """One purchase order with its lines, and what the invoices citing it add up to."""
        return await in_thread(get_po, services, po, vendor)

    @server.tool(name="get_contract", annotations=READ_ONLY)
    async def get_contract_tool(
        contract: Annotated[str, Field(description="Contract id, or its contract number")],
        vendor: Vendor = None,
    ) -> ContractRecord:
        """One contract: dates, payment terms and the rate card (agreed unit prices)."""
        return await in_thread(get_contract, services, contract, vendor)

    @server.tool(name="get_payments", annotations=READ_ONLY)
    async def get_payments_tool(
        invoice: Annotated[str, Field(description="Invoice id, or the number printed on it")],
        vendor: Vendor = None,
    ) -> InvoicePayments:
        """The payments applied to an invoice (reference, date, method, amount applied),
        with what is paid and still outstanding."""
        return await in_thread(get_payments, services, invoice, vendor)

    @server.tool(name="list_overdue", annotations=READ_ONLY)
    async def list_overdue_tool(
        min_days: Annotated[
            int, Field(ge=1, description="At least this many days past the due date")
        ] = 1,
        vendor: Annotated[str | None, Field(description="Only this vendor (id or name)")] = None,
        as_of: Annotated[
            date | None, Field(description="Count days overdue at this date (default today)")
        ] = None,
        limit: Annotated[int, Field(ge=1, le=200, description="Most invoices to list")] = 50,
    ) -> OverdueList:
        """Invoices past their due date with money still owed, most overdue first, with
        the outstanding total per currency."""
        return await in_thread(list_overdue, services, min_days, vendor, as_of=as_of, limit=limit)

    @server.tool(
        name="run_analytics",
        annotations=READ_ONLY,
        description=(
            "Accounts-payable reports from the ledger (amounts per currency, never added "
            "across currencies). Reports and their optional params:\n"
            + "\n".join(
                f"- {name}({', '.join(REPORT_PARAMS[name])}): {text}"
                for name, text in REPORT_DESCRIPTIONS.items()
            )
            + "\nDates are YYYY-MM-DD; vendor is an id or a name."
        ),
    )
    async def run_analytics_tool(
        report: Annotated[Report, Field(description="Which report")],
        params: Annotated[
            dict[str, Any] | None, Field(description="The report's parameters")
        ] = None,
        limit: Annotated[int, Field(ge=1, le=MAX_ROWS, description="Most rows to return")] = 200,
    ) -> AnalyticsResult:
        return await in_thread(run_analytics, services, report, params, limit=limit)

    @server.tool(name="get_vendor_contact", annotations=READ_ONLY)
    async def get_vendor_contact_tool(
        vendor: Annotated[str, Field(description="A vendor id (V001) or name")],
    ) -> VendorContact:
        """A vendor's name, aliases, tax id, email (if on file) and current contract terms."""
        return await in_thread(get_vendor_contact, services, vendor)

    @server.tool(name="get_document_text", annotations=READ_ONLY)
    async def get_document_text_tool(
        document: Annotated[
            str, Field(description="A document id, or the number of the record it holds")
        ],
        max_chars: Annotated[int, Field(ge=500, le=MAX_TEXT_CHARS)] = 20_000,
    ) -> DocumentText:
        """The text of a stored document (as read at ingestion), for extraction or
        quoting, with its type, status and citation tag."""
        return await in_thread(get_document_text, services, document, max_chars=max_chars)

    @server.tool(name="save_extraction", annotations=SAVE)
    async def save_extraction_tool(
        document_id: Annotated[str, Field(description="The stored document it was read from")],
        record: Annotated[
            dict[str, Any],
            Field(
                description="The extracted record as JSON, with doc_type (invoice, "
                "purchase_order, contract, payment or receipt) and that type's fields"
            ),
        ],
    ) -> SavedExtraction:
        """Validate an extracted record and store it for its document: Postgres rows, the
        vendor and graph links, and for an invoice an anomaly score. Saving the same
        record again changes nothing."""
        return await in_thread(save_extraction, services, document_id, record)
