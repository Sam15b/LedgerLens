"""Compare the anomaly detectors on the labelled synthetic ledger (layer 8 evaluation).

    python -m ml.anomaly.evaluate --features data/processed/anomaly \\
        --model models/anomaly/<version> [--split test] [--k 50] [--report anomaly.md]

Every method gives each invoice of a split a **score** (higher = more suspicious) and a
**flag** (its own yes/no cut). The labels are the anomaly types the synthetic ledger
injected (``normal`` otherwise). Per method:

* **PR-AUC** (average precision) and **ROC-AUC** of the scores. PR-AUC is the headline:
  anomalies are ~9 % of invoices, and ROC-AUC looks good on imbalanced data even when the
  top of the ranking is poor.
* **precision@k**: the share of the ``k`` highest-scored invoices that are anomalies (what a
  reviewer working down the list sees first).
* precision and recall of the **flags**, and recall **per anomaly type**.

The default split is ``test``: the latest 15 % of invoices, which no model or threshold saw
(the autoencoder picked its threshold on ``val``).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ledgerlens.anomaly.features import Matrix
from ml.anomaly.train_autoencoder import NORMAL, Split, load_model, load_split

Flags = npt.NDArray[np.bool_]
Strings = npt.NDArray[np.str_]

DEFAULT_K = 50


# --------------------------------------------------------------------------- metrics


def _ranked(scores: Matrix, positive: Flags) -> tuple[Matrix, Flags]:
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    return np.asarray(scores, dtype=np.float64)[order], np.asarray(positive, dtype=bool)[order]


def average_precision(scores: Matrix, positive: Flags) -> float:
    """PR-AUC as average precision: ``sum((R_n - R_{n-1}) * P_n)`` over the distinct score
    thresholds (equal scores are one threshold, as in scikit-learn). ``nan`` without
    positives."""
    ranked, hits = _ranked(scores, positive)
    if not hits.any():
        return float("nan")
    tp = np.cumsum(hits)
    last_of_tie = np.append(ranked[1:] != ranked[:-1], True)
    tp = tp[last_of_tie]
    seen = np.flatnonzero(last_of_tie) + 1
    precision = tp / seen
    recall = tp / tp[-1]
    return float(np.sum(np.diff(recall, prepend=0.0) * precision))


def _average_ranks(values: Matrix) -> Matrix:
    """1-based ranks, ties sharing their mean rank."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    for end in [*np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1, len(values)]:
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return ranks


def roc_auc(scores: Matrix, positive: Flags) -> float:
    """The chance that a random anomaly scores above a random normal invoice (ties count
    half). ``nan`` unless both classes are present."""
    scores = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return float("nan")
    ranks = _average_ranks(scores)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def precision_at_k(scores: Matrix, positive: Flags, k: int) -> float:
    """The share of anomalies among the ``k`` highest scores (ties broken by input order)."""
    if k <= 0:
        raise ValueError("k must be positive")
    _, hits = _ranked(scores, positive)
    top = hits[:k]
    return float(top.mean()) if len(top) else float("nan")


def recall_by_type(flags: Flags, labels: Strings) -> dict[str, tuple[int, int]]:
    """``{anomaly type: (flagged, total)}`` for every type in ``labels`` but ``normal``."""
    flags = np.asarray(flags, dtype=bool)
    labels = np.asarray(labels)
    return {
        label: (int(flags[labels == label].sum()), int((labels == label).sum()))
        for label in sorted(set(labels.tolist()) - {NORMAL})
    }


@dataclass(frozen=True)
class MethodResult:
    """One method's scores on one split, summarised."""

    method: str
    pr_auc: float
    roc_auc: float
    k: int
    precision_at_k: float
    flagged: int
    precision: float
    """Of the flagged invoices."""
    recall: float
    per_type: dict[str, tuple[int, int]] = field(default_factory=dict)
    note: str = ""


def evaluate(
    method: str,
    scores: Matrix,
    flags: Flags,
    labels: Strings,
    *,
    k: int = DEFAULT_K,
    note: str = "",
) -> MethodResult:
    labels = np.asarray(labels)
    positive = labels != NORMAL
    flags = np.asarray(flags, dtype=bool)
    flagged = int(flags.sum())
    caught = int((flags & positive).sum())
    return MethodResult(
        method=method,
        pr_auc=average_precision(scores, positive),
        roc_auc=roc_auc(scores, positive),
        k=k,
        precision_at_k=precision_at_k(scores, positive, k),
        flagged=flagged,
        precision=caught / flagged if flagged else 0.0,
        recall=caught / int(positive.sum()) if positive.any() else 0.0,
        per_type=recall_by_type(flags, labels),
        note=note,
    )


# --------------------------------------------------------------------------- methods


def autoencoder_scores(model_dir: Path, split: Split) -> tuple[Matrix, float]:
    """The saved autoencoder's reconstruction errors on ``split`` and its threshold."""
    saved = load_model(model_dir)
    return saved.errors(split.x_raw), saved.threshold


# --------------------------------------------------------------------------- report


def format_table(results: Sequence[MethodResult]) -> str:
    """A markdown table: one row per method, a recall column per anomaly type."""
    types = sorted({t for r in results for t in r.per_type})
    k = results[0].k if results else DEFAULT_K
    head = ["Method", "PR-AUC", "ROC-AUC", f"P@{k}", "Flagged", "Precision", "Recall", *types]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in results:
        cells = [
            r.method,
            f"{r.pr_auc:.3f}",
            f"{r.roc_auc:.3f}",
            f"{r.precision_at_k:.2f}",
            str(r.flagged),
            f"{r.precision:.2f}",
            f"{r.recall:.2f}",
            *(f"{r.per_type[t][0]}/{r.per_type[t][1]}" if t in r.per_type else "-" for t in types),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.anomaly.evaluate",
        description="Compare anomaly detectors on the labelled synthetic ledger.",
    )
    parser.add_argument("--features", type=Path, default=Path("data/processed/anomaly"))
    parser.add_argument("--model", type=Path, required=True, help="models/anomaly/<version>")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="for precision@k")
    return parser


def run(args: argparse.Namespace) -> list[MethodResult]:
    split = load_split(args.features, args.split)
    errors, threshold = autoencoder_scores(args.model, split)
    return [
        evaluate(
            "Autoencoder",
            errors,
            errors >= threshold,
            split.labels,
            k=args.k,
            note=f"reconstruction error >= {threshold:.3f}",
        )
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(format_table(run(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
