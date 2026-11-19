"""Deterministic anomaly rules: high-precision checks with a reason a reviewer can read.

Each rule looks at one incoming invoice (:class:`InvoiceRef`) against what is already
known (stored invoices, its PO, its vendor's contract, the vendor master) and returns
:class:`RuleHit` objects, each naming the rule, the invoice, the other records involved and
a one-line reason. See docs/layers/08-anomaly-detection.md, "Rules".

* **exact duplicate**: the same vendor and invoice number as a stored invoice, or the same
  file (SHA-256).
* **near duplicate**: the same vendor, a total within 1 %, a date within 7 days and an
  invoice number at least 85 % similar (a resubmission with the number or date retyped).
  :func:`link_possible_duplicates` records each pair as a ``POSSIBLE_DUPLICATE`` edge in
  Neo4j, so the graph and the chat agent see it too.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from rapidfuzz import fuzz

from ledgerlens.core.schemas import Invoice
from ledgerlens.extraction.normalize import canonical_vendor

if TYPE_CHECKING:
    from neo4j import Driver

RuleName = Literal[
    "exact_duplicate", "near_duplicate", "over_po", "contract_rate_breach", "unknown_vendor"
]

_NOT_ALNUM = re.compile(r"[^0-9A-Z]")

NEAR_AMOUNT = Decimal("0.01")
"""Near duplicate: totals within 1 % of the larger one."""
NEAR_DAYS = 7
NEAR_NUMBER_SIMILARITY = 0.85


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


def number_similarity(a: str, b: str) -> float:
    """0-1 similarity of two invoice numbers (rapidfuzz ratio of their keys)."""
    return fuzz.ratio(number_key(a), number_key(b)) / 100


def amount_gap(a: Decimal, b: Decimal) -> Decimal:
    """``|a - b|`` as a share of the larger magnitude (0 when both are 0)."""
    scale = max(abs(a), abs(b))
    return abs(a - b) / scale if scale else Decimal(0)


def near_duplicates(candidate: InvoiceRef, stored: Iterable[InvoiceRef]) -> list[RuleHit]:
    """``candidate`` looks like a stored invoice re-sent with small edits: same vendor,
    total within 1 %, date within 7 days, invoice number > 85 % similar (but not equal:
    that is an exact duplicate)."""
    invoice = candidate.invoice
    hits = []
    for other in stored:
        if other.id == candidate.id or other.vendor_key != candidate.vendor_key:
            continue
        if other.number_key == candidate.number_key:
            continue
        gap = amount_gap(invoice.total, other.invoice.total)
        days = abs((invoice.invoice_date - other.invoice.invoice_date).days)
        if gap >= NEAR_AMOUNT or days > NEAR_DAYS:
            continue
        similarity = number_similarity(invoice.invoice_number, other.invoice.invoice_number)
        if similarity <= NEAR_NUMBER_SIMILARITY:
            continue
        reason = (
            f"looks like {other.invoice.invoice_number} ({other.id}) re-sent: total "
            f"{invoice.total} vs {other.invoice.total}, {days} day(s) apart, numbers "
            f"{similarity:.0%} alike"
        )
        hits.append(
            RuleHit(
                rule="near_duplicate",
                invoice_id=candidate.id,
                reason=reason,
                related_ids=[other.id],
                details={
                    "other_number": other.invoice.invoice_number,
                    "amount_gap": float(gap),
                    "days_apart": days,
                    "number_similarity": round(similarity, 3),
                },
            )
        )
    return hits


LINK_POSSIBLE_DUPLICATES = """
UNWIND $pairs AS pair
MATCH (a:Invoice {id: pair.invoice_id})
MATCH (b:Invoice {id: pair.other_id})
MERGE (a)-[r:POSSIBLE_DUPLICATE]->(b)
SET r.reason = pair.reason,
    r.amount_gap = pair.amount_gap,
    r.days_apart = pair.days_apart,
    r.number_similarity = pair.number_similarity
RETURN count(r) AS edges
"""


def link_possible_duplicates(
    driver: Driver, hits: Sequence[RuleHit], *, database: str | None = None
) -> int:
    """Write each near-duplicate hit as ``(invoice)-[:POSSIBLE_DUPLICATE]->(earlier)``
    (idempotent). Pairs whose invoices are not in the graph are skipped. Returns the
    number of edges written."""
    from ledgerlens.storage.neo4j.driver import run_write

    pairs = [
        {"invoice_id": hit.invoice_id, "other_id": other, "reason": hit.reason, **hit.details}
        for hit in hits
        if hit.rule == "near_duplicate"
        for other in hit.related_ids
    ]
    for pair in pairs:
        pair.pop("other_number", None)
    if not pairs:
        return 0
    rows = run_write(driver, LINK_POSSIBLE_DUPLICATES, {"pairs": pairs}, database=database)
    return int(rows[0]["edges"]) if rows else 0
