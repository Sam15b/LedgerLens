"""Invoices, their line items, and receipts (SROIE-style till receipts).

Field names follow the ``invoices`` table and the extraction schema in
docs/layers/01-ingestion.md. Values are hard-validated; arithmetic and date order are
reported by :meth:`Invoice.consistency_issues` (see :mod:`ledgerlens.core.schemas.common`).
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import Field

from ledgerlens.core.schemas.common import (
    ROUNDING_TOLERANCE,
    CurrencyCode,
    ExtractedModel,
    Money,
    NonNegativeMoney,
    OptionalText,
    SchemaModel,
    Text,
    amounts_differ,
    sum_amounts,
)


class LineItem(SchemaModel):
    """One billed line. ``amount`` is the line total; it may be negative (discount/credit)."""

    description: Text
    quantity: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    unit_price: Money | None = None
    amount: Money

    def consistency_issues(self, *, tolerance: Decimal = ROUNDING_TOLERANCE) -> list[str]:
        if self.quantity is None or self.unit_price is None:
            return []
        expected = self.quantity * self.unit_price
        if amounts_differ(expected, self.amount, tolerance=tolerance):
            return [
                f"line {self.description!r}: {self.quantity} x {self.unit_price} = {expected}, "
                f"but amount is {self.amount}"
            ]
        return []


def totals_issues(
    lines: list[LineItem],
    subtotal: Decimal | None,
    tax: Decimal | None,
    total: Decimal,
    *,
    tolerance: Decimal = ROUNDING_TOLERANCE,
) -> list[str]:
    """Line, subtotal, tax and total arithmetic shared by invoices and purchase orders.

    Each line may round by ``tolerance``, so the line sum is allowed one ``tolerance`` per
    line.
    """
    issues = [issue for line in lines for issue in line.consistency_issues(tolerance=tolerance)]
    if lines:
        line_sum = sum_amounts(line.amount for line in lines)
        # Without a subtotal, the lines should add up to the total before tax.
        target, name = (
            (subtotal, "subtotal") if subtotal is not None else (total - (tax or 0), "total - tax")
        )
        if amounts_differ(line_sum, target, tolerance=tolerance * len(lines)):
            issues.append(f"line items add up to {line_sum}, but {name} is {target}")
    if subtotal is not None:
        expected_total = subtotal + (tax or Decimal(0))
        if amounts_differ(expected_total, total, tolerance=tolerance):
            issues.append(f"subtotal + tax = {expected_total}, but total is {total}")
    return issues


class Invoice(ExtractedModel):
    doc_type: Literal["invoice"] = "invoice"
    invoice_number: Text
    vendor_name: Text
    vendor_tax_id: OptionalText = None
    customer_name: OptionalText = None
    invoice_date: date
    due_date: date | None = None
    po_number: OptionalText = None
    """The buyer's purchase order this invoice bills against (key for the 3-way match)."""
    currency: CurrencyCode = "USD"
    subtotal: Money | None = None
    tax: Money | None = None
    total: Money
    line_items: list[LineItem] = Field(default_factory=list)

    def consistency_issues(self, *, tolerance: Decimal = ROUNDING_TOLERANCE) -> list[str]:
        """Arithmetic and date problems; an empty list means the invoice adds up."""
        issues = totals_issues(
            self.line_items, self.subtotal, self.tax, self.total, tolerance=tolerance
        )
        if self.due_date is not None and self.due_date < self.invoice_date:
            issues.append(f"due date {self.due_date} is before invoice date {self.invoice_date}")
        return issues


class Receipt(ExtractedModel):
    """A till/expense receipt (the SROIE dataset): who was paid, when, and how much."""

    doc_type: Literal["receipt"] = "receipt"
    vendor_name: Text
    vendor_address: OptionalText = None
    receipt_number: OptionalText = None
    receipt_date: date
    currency: CurrencyCode = "USD"
    tax: NonNegativeMoney | None = None
    total: NonNegativeMoney
    line_items: list[LineItem] = Field(default_factory=list)

    def consistency_issues(self, *, tolerance: Decimal = ROUNDING_TOLERANCE) -> list[str]:
        # Receipts usually print tax inside the total, so only lines vs total are checked.
        return totals_issues(self.line_items, None, None, self.total, tolerance=tolerance)
