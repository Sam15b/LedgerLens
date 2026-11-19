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

Methods: the deterministic **rules** (run over the whole ledger in arrival order, so each
invoice is checked against the invoices, POs and contracts before it: :func:`ledger_rule_hits`),
an **Isolation Forest** baseline, the saved **autoencoder**, and **rules + AE**
(:func:`ledgerlens.anomaly.scorer.combine_arrays`). The combined method gets its own AE
threshold (:func:`select_combined_threshold`): once the rules have taken their hits, the
autoencoder can flag more at the same 0.8 target precision.

The default split is ``test``: the latest 15 % of invoices, which no model or threshold saw
(the autoencoder picked its threshold on ``val``).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ledgerlens.anomaly.features import Matrix
from ledgerlens.anomaly.rules import (
    InvoiceRef,
    RuleContext,
    RuleHit,
    VendorMaster,
    check_invoice,
)
from ledgerlens.anomaly.scorer import combine_arrays
from ledgerlens.core.schemas import Contract, Invoice, PurchaseOrder
from ml.anomaly.train_autoencoder import (
    NORMAL,
    TARGET_PRECISION,
    Split,
    Threshold,
    load_model,
    load_split,
    select_threshold,
)

Flags = npt.NDArray[np.bool_]
Strings = npt.NDArray[np.str_]

DEFAULT_K = 50
IF_TREES = 300
SEED = 13


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


def isolation_forest_scores(
    train: Split, splits: Sequence[Split], *, n_estimators: int = IF_TREES, seed: int = SEED
) -> list[Matrix]:
    """Isolation Forest baseline: fit on the training split's normal invoices (the rows
    the autoencoder learns from, scaled the same way), then score each of ``splits``.
    The score is ``-score_samples``: higher = isolated in fewer splits = more unusual."""
    from sklearn.ensemble import IsolationForest

    forest = IsolationForest(n_estimators=n_estimators, random_state=seed).fit(train.normal)
    return [-np.asarray(forest.score_samples(s.x), dtype=np.float64) for s in splits]


def ledger_rule_hits(records: Path) -> dict[str, list[RuleHit]]:
    """Every rule hit on every invoice of a synthetic ``records.jsonl``, keyed by invoice
    id. Invoices are checked in arrival order (date, then id), each against the invoices
    before it, its PO (less what earlier invoices billed against it), its contract and the
    vendor master, as ingestion would see them."""
    vendors: list[dict[str, object]] = []
    contracts: dict[str, Contract] = {}
    pos: dict[str, PurchaseOrder] = {}
    invoices: list[tuple[InvoiceRef, str | None]] = []
    with records.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            kind, data = record["record_type"], record["data"]
            if kind == "vendor":
                vendors.append(data)
            elif kind == "contract":
                contract = Contract.model_validate(data)
                contracts[contract.contract_number or record["id"]] = contract
            elif kind == "purchase_order":
                po = PurchaseOrder.model_validate(data)
                pos[po.po_number] = po
            elif kind == "invoice":
                ref = InvoiceRef(
                    record["id"], Invoice.model_validate(data), record.get("vendor_id")
                )
                invoices.append((ref, record.get("contract_number")))
    invoices.sort(key=lambda item: (item[0].invoice.invoice_date, item[0].id))
    master = VendorMaster.from_vendors(vendors)
    by_vendor: dict[str, list[InvoiceRef]] = defaultdict(list)
    billed: dict[str, Decimal] = defaultdict(Decimal)
    hits: dict[str, list[RuleHit]] = {}
    for ref, contract_number in invoices:
        po_number = ref.invoice.po_number
        context = RuleContext(
            stored=by_vendor[ref.vendor_key],
            purchase_order=pos.get(po_number) if po_number else None,
            already_billed=billed[po_number] if po_number else Decimal(0),
            contract=contracts.get(contract_number) if contract_number else None,
            master=master,
        )
        hits[ref.id] = check_invoice(ref, context)
        by_vendor[ref.vendor_key].append(ref)
        if po_number:
            billed[po_number] += ref.invoice.total
    return hits


def swapped_duplicates(records: Path) -> dict[str, str]:
    """``{copy id: original id}`` for the labelled duplicates that arrive *before* the
    invoice they copy (the ledger shifts a near duplicate's date by up to 7 days either
    way). Checked in arrival order, the rules then find the pair on the original instead:
    the copy is missed and the clean original is flagged (as a near duplicate, and over
    its PO, which the copy already billed)."""
    arrival: dict[str, tuple[str, str]] = {}
    copies: dict[str, str] = {}
    with records.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["record_type"] != "invoice":
                continue
            arrival[record["id"]] = (record["data"]["invoice_date"], record["id"])
            if record.get("duplicate_of"):
                copies[record["id"]] = record["duplicate_of"]
    return {
        copy: original
        for copy, original in copies.items()
        if original in arrival and arrival[copy] < arrival[original]
    }


def rule_scores(split: Split, hits: dict[str, list[RuleHit]]) -> Matrix:
    """How many different rules fired on each invoice of ``split``."""
    return np.array(
        [len({hit.rule for hit in hits.get(str(i), [])}) for i in split.ids], dtype=np.float64
    )


def select_combined_threshold(
    errors: Matrix,
    hard_hits: Flags,
    is_anomaly: Flags,
    *,
    target_precision: float = TARGET_PRECISION,
) -> Threshold:
    """The autoencoder threshold for rules + AE: the lowest error at which the combined
    flags (every hard hit, plus AE errors at or above it) keep ``target_precision``.
    Ranks hard hits first and reuses :func:`select_threshold` on that ranking."""
    ranked = np.where(hard_hits, np.inf, errors)
    threshold = select_threshold(ranked, is_anomaly, target_precision=target_precision)
    if np.isinf(threshold.value):
        # the rules alone are at the target: the AE flags nothing more
        return Threshold(
            float(np.max(errors)) + 1.0,
            threshold.method,
            target_precision,
            threshold.precision,
            threshold.recall,
            threshold.flagged,
        )
    return threshold


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
    parser.add_argument("--records", type=Path, default=Path("data/synthetic/records.jsonl"))
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="for precision@k")
    parser.add_argument(
        "--save-combined",
        action="store_true",
        help="write the rules + AE threshold into the model's metadata.json",
    )
    return parser


@dataclass
class Comparison:
    """Every method on one split, and the rules + AE threshold chosen on val."""

    split: str
    results: list[MethodResult]
    combined_threshold: Threshold
    swaps: str
    """How many rule misses and false alarms are swapped duplicate pairs."""


def compare(
    features: Path, model: Path, records: Path, *, split_name: str = "test", k: int = DEFAULT_K
) -> Comparison:
    train, val = load_split(features, "train"), load_split(features, "val")
    split = load_split(features, split_name)
    errors, threshold = autoencoder_scores(model, split)
    hits = ledger_rule_hits(records)
    rules = rule_scores(split, hits)
    # rules + AE: the AE threshold re-chosen on val with the hard hits already flagged
    val_errors, _ = autoencoder_scores(model, val)
    combined_cut = select_combined_threshold(val_errors, rule_scores(val, hits) > 0, val.is_anomaly)
    # the forest's cut is chosen like the autoencoder's: precision >= 0.8 on val, else p99
    if_val, if_split = isolation_forest_scores(train, [val, split])
    if_cut = select_threshold(if_val, val.is_anomaly)
    results = [
        evaluate("Rules", rules, rules > 0, split.labels, k=k, note="any rule hit"),
        evaluate(
            "Isolation Forest",
            if_split,
            if_split >= if_cut.value,
            split.labels,
            k=k,
            note=f"{IF_TREES} trees, train normals; cut {if_cut.value:.3f} ({if_cut.method})",
        ),
        evaluate(
            "Autoencoder",
            errors,
            errors >= threshold,
            split.labels,
            k=k,
            note=f"reconstruction error >= {threshold:.3f} (saved threshold)",
        ),
    ]
    for name, cut in (("Rules + AE", threshold), ("Rules + AE (combined cut)", combined_cut.value)):
        scores, flags = combine_arrays(rules > 0, errors, cut)
        results.append(
            evaluate(name, scores, flags, split.labels, k=k, note=f"any rule, or AE >= {cut:.3f}")
        )
    return Comparison(split_name, results, combined_cut, _swap_summary(split, hits, records))


def _swap_summary(split: Split, hits: dict[str, list[RuleHit]], records: Path) -> str:
    swapped = swapped_duplicates(records)
    originals = set(swapped.values())
    flagged = rule_scores(split, hits) > 0
    rows = list(zip([str(i) for i in split.ids], flagged, split.labels, strict=True))
    false_alarms = [i for i, f, label in rows if f and label == NORMAL]
    missed = [i for i, f, label in rows if not f and label in ("exact_duplicate", "near_duplicate")]
    return (
        f"{sum(i in originals for i in false_alarms)} of {len(false_alarms)} rule false alarms "
        "are originals that arrived after their own copy; "
        f"{sum(i in swapped for i in missed)} of {len(missed)} missed duplicates arrived "
        "before the invoice they copy"
    )


def save_combined_threshold(model: Path, threshold: Threshold) -> None:
    """Record the rules + AE threshold in the model's ``metadata.json`` (``combined``):
    the scorer uses it in place of the autoencoder's own threshold."""
    path = model / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["combined"] = {**asdict(threshold), "chosen_on": "val, rules hard hits flagged"}
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = compare(args.features, args.model, args.records, split_name=args.split, k=args.k)
    print(format_table(result.results))
    for r in result.results:
        print(f"- {r.method}: {r.note}")
    print(f"- {args.split}: {result.swaps}")
    if args.save_combined:
        save_combined_threshold(args.model, result.combined_threshold)
        print(f"saved combined threshold {result.combined_threshold.value:.4f} to {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
