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
  since the vendor's previous invoice;
* **structure**: line count, mean and max unit price, whether it cites a PO and carries
  the vendor's tax id;
* **vendor history**: how many invoices the vendor sent before, and the mean and spread
  of their log totals. At ingestion they come from Postgres
  (:func:`load_vendor_history`); for a batch, :class:`HistoryTracker` keeps them as it
  walks the invoices in date order;
* **text** (optional): the invoice's header-chunk embedding reduced to 16 dimensions by
  :class:`EmbeddingPCA` (fit on training invoices). The vector is then
  :func:`feature_names` with ``embedding=True``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ledgerlens.core.schemas import Invoice, LineItem

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

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
STRUCTURE_FEATURES = (
    "line_count",
    "log_mean_unit_price",
    "log_max_unit_price",
    "has_po",
    "has_tax_id",
)
VENDOR_FEATURES = ("log_vendor_invoice_count", "vendor_mean_log_total", "vendor_std_log_total")
FEATURE_NAMES: tuple[str, ...] = (
    *AMOUNT_FEATURES,
    *TIMING_FEATURES,
    *STRUCTURE_FEATURES,
    *VENDOR_FEATURES,
)

EMBEDDING_DIM = 16
EMBEDDING_FEATURES = tuple(f"emb_pca_{i:02d}" for i in range(EMBEDDING_DIM))

Matrix = npt.NDArray[np.float64]

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


ROUND_MIN = Decimal(1000)
"""Below this, two significant digits is ordinary (``240.00``, ``15.00``)."""


def is_round(total: Decimal) -> bool:
    """A total of at least 1,000 with at most two significant digits (``1000``, ``2500``,
    ``15,000,000``): typed in, not added up from line items and tax."""
    return abs(total) >= ROUND_MIN and significant_digits(total) <= 2


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


def unit_price(line: LineItem) -> Decimal | None:
    """The line's unit price, or its amount over its quantity when only those are given."""
    if line.unit_price is not None:
        return line.unit_price
    if line.quantity:
        return line.amount / line.quantity
    return None


def structure_features(invoice: Invoice) -> dict[str, float]:
    prices = [signed_log(p) for p in map(unit_price, invoice.line_items) if p is not None]
    return {
        "line_count": float(len(invoice.line_items)),
        "log_mean_unit_price": sum(prices) / len(prices) if prices else 0.0,
        "log_max_unit_price": max(prices) if prices else 0.0,
        "has_po": float(bool(invoice.po_number)),
        "has_tax_id": float(bool(invoice.vendor_tax_id)),
    }


def vendor_features(history: VendorHistory) -> dict[str, float]:
    return {
        "log_vendor_invoice_count": math.log1p(history.count),
        "vendor_mean_log_total": history.mean_log_total,
        "vendor_std_log_total": history.std_log_total,
    }


def invoice_features(invoice: Invoice, history: VendorHistory | None = None) -> dict[str, float]:
    """Every feature of ``invoice``, by name (:data:`FEATURE_NAMES` order)."""
    history = history or VendorHistory()
    values = {
        **amount_features(invoice, history),
        **timing_features(invoice, history),
        **structure_features(invoice),
        **vendor_features(history),
    }
    return {name: values[name] for name in FEATURE_NAMES}


def feature_names(*, embedding: bool = False) -> tuple[str, ...]:
    """The vector's columns: :data:`FEATURE_NAMES`, then the PCA components if used."""
    return (*FEATURE_NAMES, *EMBEDDING_FEATURES) if embedding else FEATURE_NAMES


def feature_vector(
    values: Mapping[str, float], names: Iterable[str] = FEATURE_NAMES
) -> list[float]:
    """``values`` in ``names`` order."""
    return [float(values[name]) for name in names]


# --------------------------------------------------------------------------- vendor history


@dataclass
class HistoryTracker:
    """Vendor histories for a batch: call :meth:`history` for an invoice, then :meth:`add`
    it, going through the invoices in arrival (date) order. Invoices that arrived earlier
    on the same day count as before."""

    _seen: dict[str, list[tuple[date, Decimal]]] = field(default_factory=dict)

    def history(self, vendor: str) -> VendorHistory:
        return VendorHistory.from_invoices(self._seen.get(vendor, ()))

    def add(self, vendor: str, invoice_date: date, total: Decimal) -> None:
        self._seen.setdefault(vendor, []).append((invoice_date, total))


def load_vendor_history(
    session: Session, vendor_id: str, before: date, *, exclude_id: str | None = None
) -> VendorHistory:
    """The vendor's invoices in Postgres dated on or before ``before`` (but not
    ``exclude_id``, the invoice being scored if it is stored already)."""
    from sqlalchemy import select

    from ledgerlens.storage.postgres.models import InvoiceRow

    query = select(InvoiceRow.invoice_date, InvoiceRow.total).where(
        InvoiceRow.vendor_id == vendor_id, InvoiceRow.invoice_date <= before
    )
    if exclude_id is not None:
        query = query.where(InvoiceRow.id != exclude_id)
    return VendorHistory.from_invoices((day, total) for day, total in session.execute(query))


# --------------------------------------------------------------------------- text (optional)


@dataclass(frozen=True)
class EmbeddingPCA:
    """PCA of header-chunk embeddings (numpy SVD; nothing to install at serving time).

    Fit on the training invoices' embeddings only. The output always has
    :data:`EMBEDDING_DIM` columns: with fewer samples or dimensions than that, the extra
    components are zero. Signs are fixed (largest loading positive), so refitting on the
    same data gives the same features."""

    mean: Matrix
    components: Matrix
    """``(EMBEDDING_DIM, d)``: one unit vector per row."""
    explained_variance_ratio: Matrix

    @classmethod
    def fit(cls, embeddings: npt.ArrayLike, n_components: int = EMBEDDING_DIM) -> EmbeddingPCA:
        x = np.asarray(embeddings, dtype=np.float64)
        if x.ndim != 2 or len(x) < 2:
            raise ValueError("EmbeddingPCA.fit needs a 2-d array with at least 2 rows")
        mean = x.mean(axis=0)
        _, singular, vt = np.linalg.svd(x - mean, full_matrices=False)
        k = min(n_components, vt.shape[0])
        top = vt[:k]
        signs = np.sign(top[np.arange(k), np.abs(top).argmax(axis=1)])
        top = top * np.where(signs == 0, 1.0, signs)[:, None]
        components = np.zeros((n_components, x.shape[1]))
        components[:k] = top
        variance = singular**2
        ratio = np.zeros(n_components)
        if variance.sum() > 0:
            ratio[:k] = variance[:k] / variance.sum()
        return cls(mean, components, ratio)

    def transform(self, embeddings: npt.ArrayLike) -> Matrix:
        """``(n, d)`` embeddings (or one ``(d,)`` vector) -> ``(n, EMBEDDING_DIM)``."""
        x = np.atleast_2d(np.asarray(embeddings, dtype=np.float64))
        if x.shape[1] != self.mean.shape[0]:
            raise ValueError(f"expected {self.mean.shape[0]}-d embeddings, got {x.shape[1]}-d")
        result: Matrix = (x - self.mean) @ self.components.T
        return result

    def features(self, embedding: npt.ArrayLike) -> dict[str, float]:
        """One embedding's components, named like :data:`EMBEDDING_FEATURES`."""
        row = self.transform(embedding)[0]
        return dict(zip(EMBEDDING_FEATURES, map(float, row), strict=True))

    def to_json(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "components": self.components.tolist(),
            "explained_variance_ratio": self.explained_variance_ratio.tolist(),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> EmbeddingPCA:
        return cls(
            np.asarray(data["mean"], dtype=np.float64),
            np.asarray(data["components"], dtype=np.float64),
            np.asarray(data["explained_variance_ratio"], dtype=np.float64),
        )


# --------------------------------------------------------------------------- scaling


CONSTANT_TOL = 1e-9


@dataclass(frozen=True)
class Standardizer:
    """Per-column ``(x - mean) / std``, fit on training rows only and saved with the model.

    A column that is constant in training (std 0) is only centred, so a value the model
    never saw there shows up as a large reconstruction error, not as inf or NaN."""

    names: tuple[str, ...]
    mean: Matrix
    scale: Matrix

    @classmethod
    def fit(cls, x: npt.ArrayLike, names: Iterable[str]) -> Standardizer:
        data = np.asarray(x, dtype=np.float64)
        names = tuple(names)
        if data.ndim != 2 or data.shape[1] != len(names) or len(data) == 0:
            raise ValueError(f"expected a non-empty (n, {len(names)}) matrix, got {data.shape}")
        if not np.isfinite(data).all():
            raise ValueError("cannot fit a scaler on NaN or inf values")
        mean, std = data.mean(axis=0), data.std(axis=0)
        # float noise makes a constant column's std ~1e-16, not 0: treat that as constant
        constant = std <= CONSTANT_TOL * np.maximum(1.0, np.abs(mean))
        return cls(names, mean, np.where(constant, 1.0, std))

    def transform(self, x: npt.ArrayLike) -> Matrix:
        data = np.atleast_2d(np.asarray(x, dtype=np.float64))
        if data.shape[1] != len(self.names):
            raise ValueError(f"expected {len(self.names)} columns, got {data.shape[1]}")
        result: Matrix = (data - self.mean) / self.scale
        return result

    def to_json(self) -> dict[str, Any]:
        return {"names": list(self.names), "mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Standardizer:
        return cls(
            tuple(data["names"]),
            np.asarray(data["mean"], dtype=np.float64),
            np.asarray(data["scale"], dtype=np.float64),
        )
