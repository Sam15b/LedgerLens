"""Vendor contracts: the agreed rate card and payment terms invoices are checked against."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import Field

from ledgerlens.core.schemas.common import (
    ROUNDING_TOLERANCE,
    CurrencyCode,
    ExtractedModel,
    NonNegativeMoney,
    OptionalText,
    SchemaModel,
    Text,
)

MAX_PAYMENT_TERMS_DAYS = 365


class RateCardEntry(SchemaModel):
    """An agreed price: ``unit_price`` per ``unit`` for one item or service."""

    item: Text
    unit: OptionalText = None
    """e.g. ``hour``, ``each``, ``pallet``; ``None`` when the contract does not say."""
    unit_price: NonNegativeMoney


class Contract(ExtractedModel):
    doc_type: Literal["contract"] = "contract"
    contract_number: OptionalText = None
    title: OptionalText = None
    vendor_name: Text
    customer_name: OptionalText = None
    start_date: date
    end_date: date | None = None
    """``None`` for an open-ended contract."""
    payment_terms_days: int | None = Field(default=None, ge=0, le=MAX_PAYMENT_TERMS_DAYS)
    """Days after the invoice date that payment is due ("Net 30" -> 30)."""
    currency: CurrencyCode = "USD"
    rate_card: list[RateCardEntry] = Field(default_factory=list)

    def is_active(self, on: date) -> bool:
        return self.start_date <= on and (self.end_date is None or on <= self.end_date)

    def consistency_issues(self, *, tolerance: Decimal = ROUNDING_TOLERANCE) -> list[str]:
        issues: list[str] = []
        if self.end_date is not None and self.end_date < self.start_date:
            issues.append(f"end date {self.end_date} is before start date {self.start_date}")
        keys = [(e.item.casefold(), (e.unit or "").casefold()) for e in self.rate_card]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        issues += [
            f"rate card prices {item!r}" + (f" per {unit}" if unit else "") + " more than once"
            for item, unit in duplicates
        ]
        return issues
