"""Anomaly evaluation: ranking metrics, per-type recall, the ledger rule walk, the combined
threshold and the method table."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ml.anomaly.evaluate import (
    MethodResult,
    average_precision,
    evaluate,
    format_table,
    isolation_forest_scores,
    ledger_rule_hits,
    precision_at_k,
    recall_by_type,
    roc_auc,
    rule_scores,
    save_combined_threshold,
    select_combined_threshold,
    swapped_duplicates,
)
from ml.anomaly.train_autoencoder import Split, select_threshold

# --------------------------------------------------------------------------- metrics


def test_perfect_and_worst_rankings() -> None:
    positive = np.array([True, True, False, False])
    assert average_precision(np.array([4.0, 3, 2, 1]), positive) == 1.0
    assert roc_auc(np.array([4.0, 3, 2, 1]), positive) == 1.0
    assert roc_auc(np.array([1.0, 2, 3, 4]), positive) == 0.0
    # positives ranked last: precision 1/3 at the first hit, 2/4 at the second
    assert average_precision(np.array([1.0, 2, 3, 4]), positive) == pytest.approx(
        (1 / 3 + 2 / 4) / 2
    )


def test_tied_scores_are_one_threshold() -> None:
    # every score equal: one threshold, precision = base rate, AUC = 0.5
    positive = np.array([True, False, False, False])
    flat = np.ones(4)
    assert average_precision(flat, positive) == pytest.approx(0.25)
    assert roc_auc(flat, positive) == pytest.approx(0.5)


def test_metrics_without_both_classes_are_nan() -> None:
    assert np.isnan(average_precision(np.array([1.0, 2.0]), np.array([False, False])))
    assert np.isnan(roc_auc(np.array([1.0, 2.0]), np.array([True, True])))


def test_metrics_match_scikit_learn() -> None:
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    for _ in range(50):
        scores = rng.integers(0, 5, 40).astype(float)  # plenty of ties
        positive = rng.random(40) < 0.3
        if positive.all() or not positive.any():
            continue
        assert average_precision(scores, positive) == pytest.approx(
            metrics.average_precision_score(positive, scores)
        )
        assert roc_auc(scores, positive) == pytest.approx(metrics.roc_auc_score(positive, scores))


def test_precision_at_k() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    positive = np.array([True, False, True, False])
    assert precision_at_k(scores, positive, 1) == 1.0
    assert precision_at_k(scores, positive, 2) == 0.5
    assert precision_at_k(scores, positive, 10) == 0.5  # k past the end: all rows
    with pytest.raises(ValueError):
        precision_at_k(scores, positive, 0)


def test_recall_by_type_skips_normal() -> None:
    labels = np.array(["normal", "near_duplicate", "near_duplicate", "round_amount"])
    flags = np.array([True, True, False, False])
    assert recall_by_type(flags, labels) == {"near_duplicate": (1, 2), "round_amount": (0, 1)}


def test_evaluate_summarises_flags() -> None:
    labels = np.array(["normal", "normal", "exact_duplicate", "rush_terms"])
    result = evaluate(
        "m", np.array([0.1, 0.6, 0.9, 0.2]), np.array([False, True, True, False]), labels, k=2
    )
    assert (result.flagged, result.precision, result.recall) == (2, 0.5, 0.5)
    assert result.precision_at_k == 0.5
    assert result.per_type == {"exact_duplicate": (1, 1), "rush_terms": (0, 1)}


def test_format_table_has_a_column_per_type() -> None:
    row = MethodResult("Rules", 0.5, 0.75, 50, 0.4, 3, 0.67, 0.5, {"near_duplicate": (1, 2)})
    table = format_table([row]).splitlines()
    assert table[0] == (
        "| Method | PR-AUC | ROC-AUC | P@50 | Flagged | Precision | Recall | near_duplicate |"
    )
    assert table[2] == "| Rules | 0.500 | 0.750 | 0.40 | 3 | 0.67 | 0.50 | 1/2 |"


# --------------------------------------------------------------------------- thresholds


def test_combined_threshold_takes_hard_hits_first() -> None:
    errors = np.array([0.05, 0.9, 0.3, 0.25, 0.2, 0.1])
    hard = np.array([True, False, False, False, False, False])  # low AE error, rule hit
    anomaly = np.array([True, True, True, False, True, False])
    alone = select_threshold(errors, anomaly)
    combined = select_combined_threshold(errors, hard, anomaly)
    # with the rule hit already flagged, the AE can cut lower at the same precision
    assert combined.value <= alone.value
    assert combined.value == pytest.approx(0.2)
    assert combined.precision is not None and combined.precision >= 0.8


def test_combined_threshold_when_rules_are_enough() -> None:
    errors = np.array([0.1, 0.9, 0.5])
    hard = np.array([True, False, False])
    anomaly = np.array([True, False, False])  # the AE's top errors are all normal
    combined = select_combined_threshold(errors, hard, anomaly)
    assert combined.value > errors.max()  # the AE flags nothing more


def test_save_combined_threshold(tmp_path: Path) -> None:
    (tmp_path / "metadata.json").write_text(json.dumps({"version": "ae-x"}), encoding="utf-8")
    threshold = select_threshold(np.array([0.9, 0.1]), np.array([True, False]))
    save_combined_threshold(tmp_path, threshold)
    saved = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert saved["version"] == "ae-x"
    assert saved["combined"]["value"] == pytest.approx(0.9)


# --------------------------------------------------------------------------- ledger rules


def _invoice(record_id: str, number: str, day: str, total: str, **extra: Any) -> dict[str, Any]:
    vendor_id = extra.pop("vendor_id", "V1")
    data = {
        "doc_type": "invoice",
        "invoice_number": number,
        "vendor_name": extra.pop("vendor_name", "Acme Supplies Ltd"),
        "invoice_date": day,
        "po_number": extra.pop("po_number", None),
        "currency": "GBP",
        "total": total,
        "line_items": [{"description": "Pallet wrap 500m", "amount": total}],
    }
    return {
        "record_type": "invoice",
        "id": record_id,
        "vendor_id": vendor_id,
        "anomalies": extra.pop("anomalies", []),
        "duplicate_of": extra.pop("duplicate_of", None),
        "data": data,
    }


@pytest.fixture
def ledger(tmp_path: Path) -> Path:
    records = [
        {
            "record_type": "vendor",
            "id": "V1",
            "data": {"vendor_id": "V1", "name": "Acme Supplies Ltd", "tax_id": "GB1"},
        },
        {
            "record_type": "purchase_order",
            "id": "PO-1",
            "data": {
                "doc_type": "purchase_order",
                "po_number": "PO-1",
                "vendor_name": "Acme Supplies Ltd",
                "order_date": "2025-01-01",
                "currency": "GBP",
                "total": "100.00",
            },
        },
        _invoice("S1", "INV-100", "2025-01-10", "100.00", po_number="PO-1"),
        # exact copy, received later: duplicate, and over what is left on PO-1
        _invoice(
            "S2",
            "INV-100",
            "2025-01-12",
            "100.00",
            po_number="PO-1",
            anomalies=["exact_duplicate"],
            duplicate_of="S1",
        ),
        _invoice(
            "S3",
            "INV-900",
            "2025-01-11",
            "55.00",
            vendor_id=None,
            vendor_name="Nobody Trading Co",
            anomalies=["unknown_vendor"],
        ),
        # a near duplicate dated *before* the invoice it copies
        _invoice(
            "S4",
            "INV-201",
            "2025-02-01",
            "70.00",
            anomalies=["near_duplicate"],
            duplicate_of="S5",
        ),
        _invoice("S5", "INV-200", "2025-02-03", "70.00"),
    ]
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_ledger_rule_hits_in_arrival_order(ledger: Path) -> None:
    hits = ledger_rule_hits(ledger)
    assert hits["S1"] == []
    assert {h.rule for h in hits["S2"]} == {"exact_duplicate", "over_po"}
    assert [h.rule for h in hits["S3"]] == ["unknown_vendor"]
    # the copy arrives first, so the clean original is the one that looks re-sent
    assert hits["S4"] == []
    assert [(h.rule, h.related_ids) for h in hits["S5"]] == [("near_duplicate", ["S4"])]


def test_swapped_duplicates(ledger: Path) -> None:
    assert swapped_duplicates(ledger) == {"S4": "S5"}


def test_rule_scores_count_distinct_rules(ledger: Path) -> None:
    hits = ledger_rule_hits(ledger)
    split = Split(
        x=np.zeros((3, 1)),
        x_raw=np.zeros((3, 1)),
        ids=np.array(["S1", "S2", "S3"]),
        labels=np.array(["normal", "exact_duplicate", "unknown_vendor"]),
    )
    assert rule_scores(split, hits).tolist() == [0.0, 2.0, 1.0]


def test_isolation_forest_scores_outliers_higher() -> None:
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(1)
    normal = rng.normal(size=(200, 3))
    train = Split(normal, normal, np.arange(200).astype(str), np.full(200, "normal"))
    probe = np.vstack([np.zeros((1, 3)), np.full((1, 3), 8.0)])
    test = Split(probe, probe, np.array(["in", "out"]), np.array(["normal", "round_amount"]))
    (scores,) = isolation_forest_scores(train, [test], n_estimators=50)
    assert scores[1] > scores[0]
