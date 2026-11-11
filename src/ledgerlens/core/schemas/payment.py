"""Payments (payment / remittance advice): money sent to a vendor and which invoices it settles."""

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
    amounts_differ,
    sum_amounts,
)

PaymentMethod = Literal["bank_transfer", "card", "check", "direct_debit", "cash", "other"]


class PaymentAllocation(SchemaModel):
    """The part of a payment applied to one invoice."""

    invoice_number: Text
    amount: NonNegativeMoney


class Payment(ExtractedModel):
    doc_type: Literal["payment"] = "payment"
    payment_reference: Text
    """The payer's reference for this payment (remittance number, transaction id, ...)."""
    payer_name: OptionalText = None
    vendor_name: Text
    """Who was paid."""
    paid_at: date
    method: PaymentMethod | None = None
    currency: CurrencyCode = "USD"
    amount: NonNegativeMoney
    allocations: list[PaymentAllocation] = Field(default_factory=list)
    """Invoices this payment settles; empty when the advice does not say."""

    @property
    def invoice_numbers(self) -> list[str]:
        return [a.invoice_number for a in self.allocations]

    def consistency_issues(self, *, tolerance: Decimal = ROUNDING_TOLERANCE) -> list[str]:
        issues: list[str] = []
        if self.allocations:
            allocated = sum_amounts(a.amount for a in self.allocations)
            if amounts_differ(allocated, self.amount, tolerance=tolerance):
                issues.append(f"allocations add up to {allocated}, but amount is {self.amount}")
        seen: set[str] = set()
        for number in self.invoice_numbers:
            if number in seen:
                issues.append(f"invoice {number} is allocated more than once")
            seen.add(number)
        return issues
