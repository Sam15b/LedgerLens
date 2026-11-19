"""Deterministic anomaly rules: high-precision checks with a reason a reviewer can read.

Each rule looks at one incoming invoice (:class:`InvoiceRef`) against what is already
known (stored invoices, its PO, its vendor's contract, the vendor master) and returns
:class:`RuleHit` objects, each naming the rule, the invoice, the other records involved and
a one-line reason. See docs/layers/08-anomaly-detection.md, "Rules".

* **exact duplicate**: the same vendor and invoice number as a stored invoice, or the same
  file (SHA-256).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ledgerlens.core.schemas import Invoice
from ledgerlens.extraction.normalize import canonical_vendor

RuleName = Literal[
    "exact_duplicate", "near_duplicate", "over_po", "contract_rate_breach", "unknown_vendor"
]

_NOT_ALNUM = re.compile(r"[^0-9A-Z]")


class RuleHit(BaseModel):
    """One rule firing on one invoice."""

    model_config = ConfigDict(frozen=True)

    rule: RuleName
    invoice_id: str
    reason: str
    """One sentence for the reviewer, with the numbers that triggered the rule."""
    related_ids: list[str] = Field(default_factory=list)
    """Other records involved: the invoice it duplicates, the PO, the contract."""
    details: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class InvoiceRef:
    """An invoice as the rules see it: the record plus where it came from."""

    id: str
    invoice: Invoice
    vendor_id: str | None = None
    """The vendor master id, if the vendor was resolved to one."""
    sha256: str | None = None
    """The source file's hash, if it came from a file."""

    @property
    def vendor_key(self) -> str:
        """Same key, same vendor: the master id, or the canonical printed name."""
        return self.vendor_id or f"name:{canonical_vendor(self.invoice.vendor_name)}"

    @property
    def number_key(self) -> str:
        return number_key(self.invoice.invoice_number)


def number_key(number: str) -> str:
    """An invoice number for comparison: ``"inv-005275 "`` and ``"INV 005275"`` match."""
    return _NOT_ALNUM.sub("", number.upper())


def exact_duplicates(candidate: InvoiceRef, stored: Iterable[InvoiceRef]) -> list[RuleHit]:
    """``candidate`` repeats a stored invoice: same vendor and number, or the same file."""
    same_number, same_file = [], []
    for other in stored:
        if other.id == candidate.id:
            continue
        if other.vendor_key == candidate.vendor_key and other.number_key == candidate.number_key:
            same_number.append(other.id)
        elif candidate.sha256 and other.sha256 == candidate.sha256:
            same_file.append(other.id)
    if not same_number and not same_file:
        return []
    invoice = candidate.invoice
    if same_number:
        reason = (
            f"invoice {invoice.invoice_number} from {invoice.vendor_name} was already "
            f"received as {', '.join(same_number)}"
        )
    else:
        reason = f"the same file was already received as {', '.join(same_file)}"
    return [
        RuleHit(
            rule="exact_duplicate",
            invoice_id=candidate.id,
            reason=reason,
            related_ids=same_number + same_file,
            details={"same_number": same_number, "same_file": same_file},
        )
    ]
