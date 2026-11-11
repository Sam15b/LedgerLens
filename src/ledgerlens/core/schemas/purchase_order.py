"""Purchase orders: what the buyer agreed to buy, at what price (the "PO" in the 3-way match)."""

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
    OptionalText,
    Text,
)
from ledgerlens.core.schemas.invoice import LineItem, totals_issues


class PurchaseOrder(ExtractedModel):
    doc_type: Literal["purchase_order"] = "purchase_order"
    po_number: Text
    vendor_name: Text
    vendor_tax_id: OptionalText = None
    buyer_name: OptionalText = None
    order_date: date
    delivery_date: date | None = None
    contract_number: OptionalText = None
    """The framework contract whose rate card prices this order, if any."""
    currency: CurrencyCode = "USD"
    subtotal: Money | None = None
    tax: Money | None = None
    total: Money
    line_items: list[LineItem] = Field(default_factory=list)

    def consistency_issues(self, *, tolerance: Decimal = ROUNDING_TOLERANCE) -> list[str]:
        issues = totals_issues(
            self.line_items, self.subtotal, self.tax, self.total, tolerance=tolerance
        )
        if self.delivery_date is not None and self.delivery_date < self.order_date:
            issues.append(
                f"delivery date {self.delivery_date} is before order date {self.order_date}"
            )
        return issues
