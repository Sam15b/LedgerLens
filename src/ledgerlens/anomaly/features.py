"""Numeric features of an invoice for the anomaly autoencoder (docs/layers/08-anomaly-detection.md).

Each invoice becomes a fixed-length vector (:data:`FEATURE_NAMES`, in that order) from its
own fields and its vendor's history *before* it (:class:`VendorHistory`): what was known
when the invoice arrived, so a batch built for training sees the same values the scorer
sees at ingestion. Every feature is a finite float; a missing input becomes 0 with a flag
next to it where the gap itself is informative (no due date, first invoice from a vendor).

Groups:

* **amount**: log total, tax ratio, round-number flag, z-score of the log total against
  the vendor's earlier invoices;
* **timing**: days from invoice to due date, weekday, weekend flag, day of month, days
  since the vendor's previous invoice.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from ledgerlens.core.schemas import Invoice

AMOUNT_FEATURES = ("log_total", "tax_ratio", "round_total", "total_zscore")
TIMING_FEATURES = (
    "days_to_due",
    "has_due_date",
    "weekday",
    "is_weekend",
    "day_of_month",
    "log_days_since_last",
    "first_invoice",
)
FEATURE_NAMES: tuple[str, ...] = (*AMOUNT_FEATURES, *TIMING_FEATURES)

MIN_HISTORY = 2
"""Earlier invoices a vendor needs before its z-score means anything (else 0)."""
MIN_STD = 0.05
"""Floor on the vendor's log-total spread: a vendor that always bills the same amount
would otherwise turn a 1 % change into a huge z-score."""
MAX_ABS_Z = 10.0


def signed_log(value: Decimal | float) -> float:
    """``log(1 + |x|)`` with ``x``'s sign: amounts span cents to IDR millions."""
    x = float(value)
    return math.copysign(math.log1p(abs(x)), x)


@dataclass(frozen=True)
class VendorHistory:
    """A vendor's invoices before the one being scored."""

    count: int = 0
    mean_log_total: float = 0.0
    std_log_total: float = 0.0
    last_invoice_date: date | None = None

    @classmethod
    def from_invoices(cls, invoices: Iterable[tuple[date, Decimal]]) -> VendorHistory:
        """From ``(invoice_date, total)`` pairs (any order)."""
        rows = list(invoices)
        if not rows:
            return cls()
        logs = [signed_log(total) for _, total in rows]
        mean = sum(logs) / len(logs)
        std = math.sqrt(sum((x - mean) ** 2 for x in logs) / len(logs))
        return cls(len(rows), mean, std, max(day for day, _ in rows))

    def zscore(self, log_total: float) -> float:
        if self.count < MIN_HISTORY:
            return 0.0
        z = (log_total - self.mean_log_total) / max(self.std_log_total, MIN_STD)
        return max(-MAX_ABS_Z, min(MAX_ABS_Z, z))


def significant_digits(value: Decimal) -> int:
    """``1000`` -> 1, ``1500`` -> 2, ``285.90`` -> 4 (trailing zeros do not count)."""
    digits = format(abs(value).normalize(), "f").replace(".", "").strip("0")
    return len(digits)


def is_round(total: Decimal) -> bool:
    """A total with at most two significant digits (``1000``, ``2500``, ``15,000,000``):
    typed in, not added up from line items and tax."""
    return total != 0 and significant_digits(total) <= 2


def tax_ratio(invoice: Invoice) -> float:
    """Tax over the pre-tax amount (subtotal, else total - tax), clipped to [0, 1]."""
    if invoice.tax is None or invoice.tax == 0:
        return 0.0
    base = invoice.subtotal if invoice.subtotal else invoice.total - invoice.tax
    if base <= 0:
        return 0.0
    return min(1.0, max(0.0, float(invoice.tax / base)))


def amount_features(invoice: Invoice, history: VendorHistory) -> dict[str, float]:
    log_total = signed_log(invoice.total)
    return {
        "log_total": log_total,
        "tax_ratio": tax_ratio(invoice),
        "round_total": float(is_round(invoice.total)),
        "total_zscore": history.zscore(log_total),
    }


def timing_features(invoice: Invoice, history: VendorHistory) -> dict[str, float]:
    issued = invoice.invoice_date
    due = invoice.due_date
    last = history.last_invoice_date
    since = max(0, (issued - last).days) if last is not None else None
    return {
        "days_to_due": float((due - issued).days) if due is not None else 0.0,
        "has_due_date": float(due is not None),
        "weekday": float(issued.weekday()),
        "is_weekend": float(issued.weekday() >= 5),
        "day_of_month": float(issued.day),
        "log_days_since_last": math.log1p(since) if since is not None else 0.0,
        "first_invoice": float(since is None),
    }


def invoice_features(invoice: Invoice, history: VendorHistory | None = None) -> dict[str, float]:
    """Every feature of ``invoice``, by name (:data:`FEATURE_NAMES` order)."""
    history = history or VendorHistory()
    values = {**amount_features(invoice, history), **timing_features(invoice, history)}
    return {name: values[name] for name in FEATURE_NAMES}


def feature_vector(values: Mapping[str, float]) -> list[float]:
    """``values`` in :data:`FEATURE_NAMES` order."""
    return [float(values[name]) for name in FEATURE_NAMES]
