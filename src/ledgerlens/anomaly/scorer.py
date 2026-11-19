"""One anomaly score per invoice from the rules and the autoencoder (layer 8, serving).

The two methods catch different things (``eval/reports/2026-11-20/anomaly.md``):

* the **rules** (:mod:`ledgerlens.anomaly.rules`) are deterministic and precise: a
  duplicate, a breached contract rate or an unknown vendor is a fact, not a guess. Any rule
  hit is a **hard hit** and flags the invoice whatever the autoencoder says;
* the **autoencoder** catches what no rule describes (round amounts, odd shapes for a
  vendor). It flags an invoice whose reconstruction error reaches its threshold.

:func:`combine` puts both on one 0..1 scale so a review list can be sorted by it. With
``a = error / (error + threshold)`` (0.5 exactly at the threshold):

* hard hit: ``score = 0.5 + a / 2``, in [0.5, 1);
* otherwise ``score = a / 2``, in [0, 0.5).

So every rule hit ranks above every invoice flagged by the autoencoder alone, and the
autoencoder orders invoices within each band. ``flagged`` is a hard hit or ``error >=
threshold`` (``score >= 0.25`` without a hit).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ledgerlens.anomaly.rules import RuleHit

Matrix = npt.NDArray[np.float64]
Flags = npt.NDArray[np.bool_]

RULE_BAND = 0.5
"""Hard hits score at least this; invoices without one score below it."""


def ae_score(error: float | None, threshold: float) -> float:
    """``error / (error + threshold)``: 0 for a perfect reconstruction, 0.5 at the
    threshold, towards 1 far above it. 0 without an error (no model)."""
    if error is None or error <= 0:
        return 0.0
    if threshold <= 0:
        return 1.0
    return float(error / (error + threshold))


@dataclass(frozen=True)
class AnomalyScore:
    """The combined verdict on one invoice."""

    score: float
    """0..1, for ranking: hard hits in [0.5, 1), the rest in [0, 0.5)."""
    flagged: bool
    ae_error: float | None
    """Reconstruction error (``None`` if the autoencoder was not run)."""
    ae_threshold: float
    rule_hits: tuple[RuleHit, ...] = ()

    @property
    def hard_hit(self) -> bool:
        return bool(self.rule_hits)


def combine(rule_hits: Sequence[RuleHit], error: float | None, threshold: float) -> AnomalyScore:
    """The rules' hits and the autoencoder's error on one invoice as one score."""
    a = ae_score(error, threshold)
    hit = bool(rule_hits)
    return AnomalyScore(
        score=RULE_BAND + a / 2 if hit else a / 2,
        flagged=hit or (error is not None and error >= threshold),
        ae_error=error,
        ae_threshold=threshold,
        rule_hits=tuple(rule_hits),
    )


def combine_arrays(hard_hits: Flags, errors: Matrix, threshold: float) -> tuple[Matrix, Flags]:
    """:func:`combine` over many invoices at once: ``(scores, flags)``."""
    hard_hits = np.asarray(hard_hits, dtype=bool)
    errors = np.asarray(errors, dtype=np.float64)
    a = np.where(errors > 0, errors / (errors + threshold), 0.0)
    scores: Matrix = np.where(hard_hits, RULE_BAND + a / 2, a / 2)
    flags: Flags = hard_hits | (errors >= threshold)
    return scores, flags
