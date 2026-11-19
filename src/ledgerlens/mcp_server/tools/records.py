"""Typed access to the ledger's records: agents read and write through these tools and
never query the databases themselves.

Every tool returns a pydantic model, so MCP clients get a JSON schema for the output and
JSON values back (amounts as exact decimal strings, dates as ISO strings). A record is
looked up by its id or by the number printed on it (``INV-889``, ``PO-1042``); numbers are
only unique per vendor, so an ambiguous number is an error that lists the candidates,
and ``vendor`` (an id or a name) narrows it down. Records that came from a document carry
its id and the citation tag answers use (``doc:INV-889``).
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeVar

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import AfterValidator, BaseModel, Field

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
