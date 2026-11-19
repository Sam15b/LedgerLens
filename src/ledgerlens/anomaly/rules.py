"""Deterministic anomaly rules: high-precision checks with a reason a reviewer can read.

Each rule looks at one incoming invoice (:class:`InvoiceRef`) against what is already
known (stored invoices, its PO, its vendor's contract, the vendor master) and returns
:class:`RuleHit` objects, each naming the rule, the invoice, the other records involved and
a one-line reason. See docs/layers/08-anomaly-detection.md, "Rules".

* **exact duplicate**: the same vendor and invoice number as a stored invoice, or the same
  file (SHA-256).
* **near duplicate**: the same vendor, a total within 1 %, a date within 7 days and an
  invoice number more than 85 % similar (:func:`number_similarity`: a resubmission under
  the next number, with a suffix, or with a typo).
  :func:`link_possible_duplicates` records each pair as a ``POSSIBLE_DUPLICATE`` edge in
  Neo4j, so the graph and the chat agent see it too.
* **over PO**: the total is more than what is left on the purchase order it cites (PO total
  minus what earlier invoices already billed against it), beyond 0.5 %.
* **contract-rate breach**: a line's unit price is more than 2 % over the rate for that item
  in the vendor's contract rate card.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from rapidfuzz import fuzz
from rapidfuzz.distance import OSA

from ledgerlens.core.schemas import Contract, Invoice, PurchaseOrder
from ledgerlens.core.schemas.common import ROUNDING_TOLERANCE
from ledgerlens.extraction.normalize import canonical_vendor

if TYPE_CHECKING:
    from neo4j import Driver

RuleName = Literal[
    "exact_duplicate", "near_duplicate", "over_po", "contract_rate_breach", "unknown_vendor"
]

_NOT_ALNUM = re.compile(r"[^0-9A-Z]")
_DIGIT_TAIL = re.compile(r"^(.*?)(\d+)$")

NEAR_AMOUNT = Decimal("0.01")
"""Near duplicate: totals within 1 % of the larger one."""
NEAR_DAYS = 7
NEAR_NUMBER_SIMILARITY = 0.85
OVER_PO_TOLERANCE = Decimal("0.005")
"""Over PO: allowed overrun as a share of the PO total (rounding, tax on tax)."""
RATE_TOLERANCE = Decimal("0.02")
"""Contract rate: allowed unit-price excess over the rate card."""
ITEM_MATCH = 90.0
"""rapidfuzz token-sort score (0-100) for a line description to match a rate-card item."""


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
    """0-1 similarity of two invoice numbers, as a re-sent invoice would differ.

    Numbers that differ only in their trailing digits (same prefix) are a vendor's
    numbering sequence, where most numbers share most digits: they count as alike only when
    adjacent (``...275`` / ``...276``, ``...359`` / ``...360``: 1.0) or the same digits
    transposed (a typo). Everything else (a suffix such as ``...275A``, a changed prefix)
    is the optimal-string-alignment edit similarity of the normalised keys, so one edit in a
    ten-character number scores 0.9.
    """
    key_a, key_b = number_key(a), number_key(b)
    tail_a, tail_b = _DIGIT_TAIL.match(key_a), _DIGIT_TAIL.match(key_b)
    if tail_a and tail_b and tail_a[1] == tail_b[1]:
        digits_a, digits_b = tail_a[2], tail_b[2]
        if abs(int(digits_a) - int(digits_b)) <= 1:
            return 1.0
        if len(digits_a) != len(digits_b) or sorted(digits_a) != sorted(digits_b):
            return 0.0  # other numbers in the same sequence
    return float(OSA.normalized_similarity(key_a, key_b))


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


def over_po(
    candidate: InvoiceRef,
    po: PurchaseOrder | None,
    *,
    already_billed: Decimal = Decimal(0),
    tolerance: Decimal = OVER_PO_TOLERANCE,
) -> list[RuleHit]:
    """The invoice bills more than is left on its PO: ``po.total - already_billed`` (what
    earlier invoices billed against it), with ``tolerance`` of the PO total as slack.
    Nothing to check without a PO, or if the PO is in another currency."""
    invoice = candidate.invoice
    if po is None or po.currency != invoice.currency:
        return []
    remaining = po.total - already_billed
    slack = max(ROUNDING_TOLERANCE, po.total * tolerance)
    if invoice.total <= remaining + slack:
        return []
    over = invoice.total - remaining
    reason = (
        f"total {invoice.total} {invoice.currency} is {over} over what is left on "
        f"{po.po_number} ({remaining} of {po.total}"
        + (f"; {already_billed} already billed" if already_billed else "")
        + ")"
    )
    return [
        RuleHit(
            rule="over_po",
            invoice_id=candidate.id,
            reason=reason,
            related_ids=[po.po_number],
            details={
                "po_total": str(po.total),
                "already_billed": str(already_billed),
                "remaining": str(remaining),
                "over_by": str(over),
            },
        )
    ]


def _item_key(text: str) -> str:
    return " ".join(re.findall(r"[0-9a-z]+", text.lower()))


def rate_for(description: str, contract: Contract) -> tuple[str, Decimal] | None:
    """The rate-card item ``description`` bills, and its unit price (best match above
    :data:`ITEM_MATCH`), or ``None``."""
    key = _item_key(description)
    best: tuple[float, str, Decimal] | None = None
    for entry in contract.rate_card:
        score = (
            100.0
            if _item_key(entry.item) == key
            else fuzz.token_sort_ratio(key, _item_key(entry.item))
        )
        if score >= ITEM_MATCH and (best is None or score > best[0]):
            best = (score, entry.item, entry.unit_price)
    return (best[1], best[2]) if best else None


def contract_rate_breaches(
    candidate: InvoiceRef, contract: Contract | None, *, tolerance: Decimal = RATE_TOLERANCE
) -> list[RuleHit]:
    """Lines priced over the contract rate for their item by more than ``tolerance``.
    Lines with no unit price or no matching item, and contracts in another currency, are
    not checked."""
    invoice = candidate.invoice
    if contract is None or contract.currency != invoice.currency:
        return []
    breaches: list[dict[str, Any]] = []
    for line in invoice.line_items:
        if line.unit_price is None:
            continue
        match = rate_for(line.description, contract)
        if match is None:
            continue
        item, rate = match
        if line.unit_price > rate * (1 + tolerance) + ROUNDING_TOLERANCE:
            excess = (line.unit_price / rate - 1) if rate else Decimal(1)
            breaches.append(
                {
                    "description": line.description,
                    "item": item,
                    "unit_price": str(line.unit_price),
                    "rate": str(rate),
                    "excess": round(float(excess), 4),
                }
            )
    if not breaches:
        return []
    worst = max(breaches, key=lambda b: b["excess"])
    reason = (
        f"{len(breaches)} line(s) over the {contract.contract_number or 'contract'} rate: "
        f"{worst['description']} at {worst['unit_price']} vs {worst['rate']} "
        f"(+{worst['excess']:.0%})"
    )
    return [
        RuleHit(
            rule="contract_rate_breach",
            invoice_id=candidate.id,
            reason=reason,
            related_ids=[contract.contract_number] if contract.contract_number else [],
            details={"lines": breaches},
        )
    ]
