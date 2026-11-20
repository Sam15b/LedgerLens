"""``reconcile``: the 3-way match of an invoice against its purchase order and contract.

The match is deterministic code; the reconciler agent's LLM only explains the report it
returns (docs/layers/04-agents-mcp.md, "Reconciler logic"). Each check compares records
and returns a :class:`Check` with a status and a one-line message:

* ``pass``: the invoice agrees; ``fail``: a mismatch to act on; ``warn``: worth a look
  but not a mismatch; ``skipped``: nothing to compare (no PO lines, no contract, ...).

* ``po_exists``: the invoice cites a PO number and that PO is stored;
* ``vendor_match``: the invoice's vendor is the PO's vendor;
* ``line_qty``: per PO line, the quantity invoiced (this invoice and the PO's others) is
  at most the quantity ordered; invoice lines that are not on the PO fail;
* ``unit_price``: every unit price is within ``price_tolerance_pct`` of the contract rate
  for its item;
* ``total``: the lines add up to the subtotal, and subtotal + tax to the total, within
  ``total_tolerance``;
* ``payment``: what the matched payments add up to: paid in full passes, overpaid fails,
  unpaid or partly paid fails once overdue (before that: unpaid passes, partly paid warns);
* ``terms``: the due date is the contract's ``payment_terms_days`` after the invoice date,
  within ``terms_tolerance_days``.

Tolerances come from ``configs/agents.yaml`` (``agents.reconciler``): see
:class:`Tolerances`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Annotated, Any, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from ledgerlens.anomaly.rules import money, rate_for
from ledgerlens.core.config import AgentsConfig, get_config
from ledgerlens.core.schemas import Contract, Invoice, LineItem, PurchaseOrder
from ledgerlens.core.schemas.invoice import totals_issues
from ledgerlens.extraction.normalize import canonical_vendor
from ledgerlens.mcp_server.services import Services
from ledgerlens.mcp_server.tools import READ_ONLY, in_thread
from ledgerlens.mcp_server.tools.records import tidy_quantity

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

CheckName = Literal[
    "po_exists", "vendor_match", "line_qty", "unit_price", "total", "payment", "terms"
]
PaymentStatus = Literal["paid", "partial", "unpaid", "overpaid"]
CheckStatus = Literal["pass", "fail", "warn", "skipped"]

ITEM_MATCH = 90.0
"""rapidfuzz token-sort score (0-100) for an invoice line to bill a PO line."""


class Tolerances(BaseModel):
    """How far an invoice may be off before a check fails."""

    price_pct: Decimal = Decimal("2.0")
    """Unit price vs contract rate, in percent of the rate."""
    total: Decimal = Decimal("0.01")
    """Per line and for the totals, in the invoice's currency."""
    qty: Decimal = Decimal(0)
    """Units invoiced beyond the PO quantity."""
    payment: Decimal = Decimal("0.01")
    """Paid vs total, in the invoice's currency."""
    terms_days: int = 0
    """Due date vs invoice date + contract terms, in days."""

    @classmethod
    def from_config(cls, config: AgentsConfig | None = None) -> Tolerances:
        """``agents.reconciler`` in ``configs/agents.yaml``: ``price_tolerance_pct``,
        ``total_tolerance``, ``qty_tolerance`` (defaults for any not set)."""
        config = config or get_config().agents
        reconciler = config.agents.get("reconciler")
        extra: dict[str, Any] = dict(reconciler.model_extra or {}) if reconciler else {}
        names = {
            "price_tolerance_pct": "price_pct",
            "total_tolerance": "total",
            "qty_tolerance": "qty",
            "payment_tolerance": "payment",
            "terms_tolerance_days": "terms_days",
        }
        return cls(**{field: extra[key] for key, field in names.items() if key in extra})


class Check(BaseModel):
    """One check's verdict."""

    check: CheckName
    status: CheckStatus
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


@dataclass
class MatchInputs:
    """Everything the checks compare, as records."""

    invoice: Invoice
    invoice_id: str
    vendor_id: str | None = None
    po: PurchaseOrder | None = None
    po_vendor_id: str | None = None
    contract: Contract | None = None
    other_invoices_on_po: Sequence[Invoice] = field(default_factory=list)
    """The PO's other invoices (void ones left out): their quantities count against it."""
    paid: Decimal = Decimal(0)
    """What the payments matched to this invoice add up to."""
    payments: int = 0
    today: date = field(default_factory=date.today)


def _check(name: CheckName, status: CheckStatus, message: str, **details: Any) -> Check:
    return Check(check=name, status=status, message=message, details=details)


def check_po_exists(m: MatchInputs) -> Check:
    number = m.invoice.po_number
    if not number:
        return _check("po_exists", "fail", "the invoice cites no purchase order")
    if m.po is None:
        return _check(
            "po_exists", "fail", f"{number} is not a stored purchase order", po_number=number
        )
    return _check("po_exists", "pass", f"{number} is on file", po_number=number)


def check_vendor_match(m: MatchInputs) -> Check:
    if m.po is None:
        return _check("vendor_match", "skipped", "no purchase order to compare with")
    if m.vendor_id and m.po_vendor_id:
        same = m.vendor_id == m.po_vendor_id
    else:  # not both resolved: compare the printed names
        same = canonical_vendor(m.invoice.vendor_name) == canonical_vendor(m.po.vendor_name)
    if same:
        return _check("vendor_match", "pass", f"{m.invoice.vendor_name} on both")
    return _check(
        "vendor_match",
        "fail",
        f"the invoice is from {m.invoice.vendor_name} but {m.po.po_number} was issued to "
        f"{m.po.vendor_name}",
        invoice_vendor=m.vendor_id or m.invoice.vendor_name,
        po_vendor=m.po_vendor_id or m.po.vendor_name,
    )


def _key(text: str) -> str:
    return " ".join(re.findall(r"[0-9a-z]+", text.lower()))


def match_line(line: LineItem, po_lines: Sequence[LineItem]) -> int | None:
    """The index of the PO line an invoice line bills (exact item, else the closest above
    :data:`ITEM_MATCH`), or ``None``."""
    key = _key(line.description)
    best: tuple[float, int] | None = None
    for index, po_line in enumerate(po_lines):
        score = (
            100.0
            if _key(po_line.description) == key
            else fuzz.token_sort_ratio(key, _key(po_line.description))
        )
        if score >= ITEM_MATCH and (best is None or score > best[0]):
            best = (score, index)
    return best[1] if best else None


def check_line_qty(m: MatchInputs, tol: Tolerances) -> Check:
    if m.po is None or not m.po.line_items:
        return _check("line_qty", "skipped", "no purchase order lines to compare with")
    po_lines = m.po.line_items
    invoiced = [Decimal(0)] * len(po_lines)
    unordered: list[str] = []
    for invoice in [m.invoice, *m.other_invoices_on_po]:
        for line in invoice.line_items:
            index = match_line(line, po_lines)
            if index is None:
                if invoice is m.invoice:
                    unordered.append(line.description)
                continue
            invoiced[index] += line.quantity if line.quantity is not None else Decimal(0)
    over = [
        {
            "item": po_line.description,
            "ordered": str(tidy_quantity(po_line.quantity)),
            "invoiced": str(tidy_quantity(invoiced[i])),
        }
        for i, po_line in enumerate(po_lines)
        if po_line.quantity is not None and invoiced[i] > po_line.quantity + tol.qty
    ]
    if not over and not unordered:
        others = len(m.other_invoices_on_po)
        scope = f" (with {others} other invoice(s) on it)" if others else ""
        return _check("line_qty", "pass", f"quantities within {m.po.po_number}{scope}")
    parts = [f"{o['item']}: {o['invoiced']} invoiced vs {o['ordered']} ordered" for o in over]
    parts += [f"{d!r} is not on {m.po.po_number}" for d in unordered]
    return _check("line_qty", "fail", "; ".join(parts), over=over, not_on_po=unordered)


def check_unit_price(m: MatchInputs, tol: Tolerances) -> Check:
    contract = m.contract
    if contract is None:
        return _check("unit_price", "skipped", "no contract with a rate card")
    if contract.currency != m.invoice.currency:
        return _check(
            "unit_price",
            "skipped",
            f"the contract is in {contract.currency}, the invoice in {m.invoice.currency}",
        )
    off: list[dict[str, str]] = []
    compared = 0
    for line in m.invoice.line_items:
        if line.unit_price is None:
            continue
        rate = rate_for(line.description, contract)
        if rate is None:
            continue
        compared += 1
        item, agreed = rate
        gap = (line.unit_price - agreed) / agreed * 100 if agreed else Decimal(100)
        if abs(gap) > tol.price_pct:
            off.append(
                {
                    "item": item,
                    "unit_price": money(line.unit_price),
                    "rate": money(agreed),
                    "difference_pct": f"{gap:+.1f}",
                }
            )
    number = contract.contract_number or "the contract"
    if not compared:
        return _check("unit_price", "skipped", f"no invoice line is priced on {number}'s rate card")
    if not off:
        return _check(
            "unit_price", "pass", f"{compared} line(s) within {tol.price_pct}% of {number} rates"
        )
    parts = [
        f"{o['item']} at {o['unit_price']} vs {o['rate']} ({o['difference_pct']}%)" for o in off
    ]
    return _check(
        "unit_price",
        "fail",
        f"{len(off)} line(s) off {number} rates by more than {tol.price_pct}%: " + "; ".join(parts),
        lines=off,
    )


def check_total(m: MatchInputs, tol: Tolerances) -> Check:
    invoice = m.invoice
    if not invoice.line_items and invoice.subtotal is None:
        return _check("total", "skipped", "no lines or subtotal to add up")
    issues = totals_issues(
        invoice.line_items, invoice.subtotal, invoice.tax, invoice.total, tolerance=tol.total
    )
    if issues:
        return _check("total", "fail", "; ".join(issues), issues=issues)
    return _check("total", "pass", f"lines, tax and total add up to {money(invoice.total)}")


def payment_status(paid: Decimal, total: Decimal, tol: Tolerances) -> PaymentStatus:
    if paid > total + tol.payment:
        return "overpaid"
    if paid >= total - tol.payment:
        return "paid"
    return "partial" if paid > 0 else "unpaid"


def check_payment(m: MatchInputs, tol: Tolerances) -> Check:
    invoice = m.invoice
    status = payment_status(m.paid, invoice.total, tol)
    outstanding = invoice.total - m.paid
    due = invoice.due_date
    overdue = (m.today - due).days if due is not None and due < m.today else 0
    details: dict[str, Any] = {
        "payment_status": status,
        "paid": money(m.paid),
        "outstanding": money(outstanding),
        "payments": m.payments,
        "days_overdue": overdue,
    }
    currency = invoice.currency
    if status == "paid":
        return _check("payment", "pass", f"paid in full ({money(m.paid)} {currency})", **details)
    if status == "overpaid":
        return _check("payment", "fail", f"overpaid by {money(-outstanding)} {currency}", **details)
    owed = f"{money(outstanding)} {currency} outstanding"
    state = "unpaid" if status == "unpaid" else f"partly paid ({money(m.paid)} {currency})"
    if overdue:
        return _check(
            "payment", "fail", f"{state}, {owed}, {overdue} day(s) past due {due}", **details
        )
    when = f"due {due}" if due is not None else "no due date"
    return _check(
        "payment", "pass" if status == "unpaid" else "warn", f"{state}, {owed}, {when}", **details
    )


def check_terms(m: MatchInputs, tol: Tolerances) -> Check:
    contract = m.contract
    agreed = contract.payment_terms_days if contract is not None else None
    if contract is None or agreed is None:
        return _check("terms", "skipped", "no contract payment terms to compare with")
    invoice = m.invoice
    number = contract.contract_number or "the contract"
    if invoice.due_date is None:
        return _check("terms", "warn", f"no due date; {number} says Net {agreed}")
    actual = (invoice.due_date - invoice.invoice_date).days
    details = {"agreed_days": agreed, "invoice_days": actual}
    if abs(actual - agreed) <= tol.terms_days:
        return _check("terms", "pass", f"Net {agreed} as agreed in {number}", **details)
    return _check(
        "terms",
        "fail",
        f"due {actual} day(s) after the invoice date, but {number} says Net {agreed}",
        **details,
    )


def run_checks(m: MatchInputs, tol: Tolerances | None = None) -> list[Check]:
    """Every check, in table order."""
    tol = tol or Tolerances.from_config()
    return [
        check_po_exists(m),
        check_vendor_match(m),
        check_line_qty(m, tol),
        check_unit_price(m, tol),
        check_total(m, tol),
        check_payment(m, tol),
        check_terms(m, tol),
    ]


# --------------------------------------------------------------------------- the tool


class MatchReport(BaseModel):
    """The 3-way match of one invoice: every check, and the mismatches among them."""

    invoice_id: str
    invoice_number: str
    citation: str | None
    vendor_id: str | None
    vendor_name: str
    po_number: str | None
    contract_number: str | None
    currency: str
    total: str
    as_of: date
    status: Literal["match", "mismatch"]
    """``mismatch`` if any check failed."""
    mismatches: list[CheckName] = Field(default_factory=list)
    warnings: list[CheckName] = Field(default_factory=list)
    payment_status: PaymentStatus
    checks: list[Check] = Field(default_factory=list)
    citations: dict[str, str] = Field(default_factory=dict)
    """Citation tag -> document id for the invoice, PO and contract documents."""


def load_inputs(
    session: Session, services: Services, invoice: str, vendor: str | None, today: date
) -> tuple[MatchInputs, dict[str, str]]:
    """The invoice and what it is matched against, from Postgres; and the citations of
    their documents."""
    from sqlalchemy import func, or_, select
    from sqlalchemy.orm import selectinload

    from ledgerlens.mcp_server.tools.records import citation, find_invoice
    from ledgerlens.storage.postgres.models import ContractRow, InvoiceRow, PaymentAllocationRow
    from ledgerlens.storage.postgres.repositories import (
        InvoiceRepository,
        PurchaseOrderRepository,
        to_contract,
        to_invoice,
        to_purchase_order,
    )

    row = find_invoice(session, services, invoice, vendor)
    record = to_invoice(row)
    cites: dict[str, str] = {}

    def cite(number: str | None, document_id: str | None) -> None:
        tag = citation(number, document_id)
        if tag and document_id:
            cites[tag] = document_id

    cite(row.invoice_number, row.document_id)
    po_row = None
    if record.po_number:
        repo = PurchaseOrderRepository(session)
        # the vendor's PO first; any vendor's next, so a PO issued to someone else shows up
        rows = repo.find_by_number(record.po_number, vendor_id=row.vendor_id)
        rows = rows or repo.find_by_number(record.po_number)
        po_row = rows[0] if rows else None
    contract_row = po_row.contract if po_row is not None else None
    if contract_row is None and row.vendor_id is not None:
        contract_row = session.scalars(
            select(ContractRow)
            .where(
                ContractRow.vendor_id == row.vendor_id,
                ContractRow.start_date <= record.invoice_date,
                or_(ContractRow.end_date.is_(None), ContractRow.end_date >= record.invoice_date),
            )
            .order_by(ContractRow.start_date.desc(), ContractRow.id)
            .limit(1)
        ).first()
    others: list[Invoice] = []
    if po_row is not None:
        cite(po_row.po_number, po_row.document_id)
        others = [
            to_invoice(other)
            for other in session.scalars(
                select(InvoiceRow)
                .where(
                    InvoiceRow.po_number == po_row.po_number,
                    InvoiceRow.id != row.id,
                    InvoiceRow.status != "void",
                )
                .options(selectinload(InvoiceRow.line_items))
            )
        ]
    if contract_row is not None:
        cite(contract_row.contract_number, contract_row.document_id)
    paid = InvoiceRepository(session).paid_amounts([row.id])[row.id]
    payments = session.scalar(select(func.count()).where(PaymentAllocationRow.invoice_id == row.id))
    inputs = MatchInputs(
        invoice=record,
        invoice_id=row.id,
        vendor_id=row.vendor_id,
        po=to_purchase_order(po_row) if po_row is not None else None,
        po_vendor_id=po_row.vendor_id if po_row is not None else None,
        contract=to_contract(contract_row) if contract_row is not None else None,
        other_invoices_on_po=others,
        paid=paid,
        payments=int(payments or 0),
        today=today,
    )
    return inputs, cites


def build_report(
    inputs: MatchInputs, checks: list[Check], citations: dict[str, str]
) -> MatchReport:
    invoice = inputs.invoice
    payment = next(c for c in checks if c.check == "payment")
    return MatchReport(
        invoice_id=inputs.invoice_id,
        invoice_number=invoice.invoice_number,
        citation=tag if (tag := f"doc:{invoice.invoice_number}") in citations else None,
        vendor_id=inputs.vendor_id,
        vendor_name=invoice.vendor_name,
        po_number=invoice.po_number,
        contract_number=inputs.contract.contract_number if inputs.contract else None,
        currency=invoice.currency,
        total=money(invoice.total),
        as_of=inputs.today,
        status="mismatch" if any(c.status == "fail" for c in checks) else "match",
        mismatches=[c.check for c in checks if c.status == "fail"],
        warnings=[c.check for c in checks if c.status == "warn"],
        payment_status=payment.details["payment_status"],
        checks=checks,
        citations=citations,
    )


def reconcile(
    services: Services, invoice: str, vendor: str | None = None, *, as_of: date | None = None
) -> MatchReport:
    """Match a stored invoice against its PO and contract (and its payments)."""
    today = as_of or services.today
    with services.sessions() as session:
        inputs, citations = load_inputs(session, services, invoice, vendor, today)
    return build_report(inputs, run_checks(inputs, Tolerances.from_config()), citations)


def register(server: FastMCP, services: Services) -> None:
    @server.tool(name="reconcile", annotations=READ_ONLY)
    async def reconcile_tool(
        invoice: Annotated[str, Field(description="Invoice id, or the number printed on it")],
        vendor: Annotated[
            str | None, Field(description="Narrow a number down to this vendor (id or name)")
        ] = None,
        as_of: Annotated[
            date | None, Field(description="Judge payment against this date (default today)")
        ] = None,
    ) -> MatchReport:
        """The 3-way match of an invoice with its purchase order and contract, as
        deterministic checks: po_exists, vendor_match, line_qty, unit_price, total,
        payment and terms. Each check has a status (pass / fail / warn / skipped) and a
        message; status is "mismatch" if any check failed. Explain the report; do not
        re-judge it."""
        return await in_thread(reconcile, services, invoice, vendor, as_of=as_of)
