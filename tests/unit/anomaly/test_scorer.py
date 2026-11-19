"""Combining rule hits with the autoencoder score."""

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from ledgerlens.anomaly.features import FEATURE_NAMES, Matrix, Standardizer
from ledgerlens.anomaly.rules import InvoiceRef, RuleContext, RuleHit
from ledgerlens.anomaly.scorer import (
    RULE_BAND,
    RULES_ONLY,
    AnomalyScorer,
    Autoencoder,
    ae_score,
    combine,
    combine_arrays,
    find_model,
    import_keras,
)
from ledgerlens.core.config import AnomalyConfig
from ledgerlens.core.schemas import Invoice

HIT = RuleHit(rule="exact_duplicate", invoice_id="S2", reason="already received as S1")


def test_ae_score_is_half_at_threshold() -> None:
    assert ae_score(0.2, 0.2) == 0.5
    assert ae_score(0.0, 0.2) == 0.0
    assert ae_score(None, 0.2) == 0.0
    assert 0.5 < ae_score(1.0, 0.2) < 1.0


def test_hard_hit_flags_and_outranks_the_autoencoder() -> None:
    ruled = combine([HIT], error=0.01, threshold=0.2)
    unusual = combine([], error=5.0, threshold=0.2)
    assert ruled.flagged and ruled.hard_hit
    assert unusual.flagged and not unusual.hard_hit
    assert ruled.score >= RULE_BAND > unusual.score


def test_without_hit_the_threshold_decides() -> None:
    assert not combine([], error=0.19, threshold=0.2).flagged
    assert combine([], error=0.2, threshold=0.2).flagged
    assert combine([], error=0.2, threshold=0.2).score == pytest.approx(0.25)
    assert not combine([], error=None, threshold=0.2).flagged


def test_combine_arrays_matches_combine() -> None:
    hard = np.array([True, False, False, True])
    errors = np.array([0.01, 0.5, 0.1, 0.0])
    scores, flags = combine_arrays(hard, errors, 0.2)
    for i in range(4):
        one = combine([HIT] if hard[i] else [], float(errors[i]), 0.2)
        assert scores[i] == pytest.approx(one.score)
        assert bool(flags[i]) == one.flagged


# --------------------------------------------------------------------------- serving


class StubModel:
    """Reconstructs every feature as the training mean (0 after scaling), so the error is
    the mean squared scaled value."""

    version = "ae-test"
    threshold = 1.0
    features = FEATURE_NAMES

    def __init__(self, scale: float = 1.0) -> None:
        self.scaler = Standardizer(
            FEATURE_NAMES, np.zeros(len(FEATURE_NAMES)), np.full(len(FEATURE_NAMES), scale)
        )

    def reconstruct(self, x_raw: Matrix) -> tuple[Matrix, Matrix]:
        x = self.scaler.transform(x_raw)
        return x, np.zeros_like(x)


def _ref(total: str = "120.00", number: str = "INV-1", id: str = "I1") -> InvoiceRef:
    invoice = Invoice(
        invoice_number=number,
        vendor_name="Acme Supplies Ltd",
        invoice_date=date(2026, 3, 2),
        total=Decimal(total),
    )
    return InvoiceRef(id=id, invoice=invoice, vendor_id="V001")


def test_rules_only_scorer() -> None:
    scorer = AnomalyScorer(None)
    assert scorer.model_version == RULES_ONLY
    clean = scorer.score(_ref(), RuleContext())
    assert clean.ae_error is None and not clean.flagged and clean.score == 0.0
    dup = scorer.score(_ref(id="I2"), RuleContext(stored=[_ref(id="I1")]))
    assert dup.flagged and [h.rule for h in dup.rule_hits] == ["exact_duplicate"]


def test_scorer_with_model_ranks_features_by_error() -> None:
    scorer = AnomalyScorer(StubModel())
    verdict = scorer.score(_ref(total="5000.00"), RuleContext())
    assert scorer.model_version == "ae-test"
    assert verdict.ae_error is not None and verdict.ae_threshold == 1.0
    errors = [f.error for f in verdict.feature_errors]
    assert errors == sorted(errors, reverse=True)
    assert verdict.ae_error == pytest.approx(sum(errors) / len(FEATURE_NAMES))
    top = verdict.feature_errors[0]
    assert top.name == "log_total" and top.above  # log(5001) is the largest raw value
    assert verdict.flagged and not verdict.hard_hit


def _model_dir(root: Path, name: str, *, combined: float | None = None) -> Path:
    path = root / name
    path.mkdir(parents=True)
    scaler = Standardizer(FEATURE_NAMES, np.zeros(len(FEATURE_NAMES)), np.ones(len(FEATURE_NAMES)))
    (path / "scaler.json").write_text(json.dumps(scaler.to_json()), encoding="utf-8")
    metadata = {"version": name, "features": list(FEATURE_NAMES), "threshold": {"value": 0.3}}
    if combined is not None:
        metadata["combined"] = {"value": combined}
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return path


def test_find_model(tmp_path: Path) -> None:
    assert find_model(tmp_path / "missing") is None
    _model_dir(tmp_path, "ae-20261101-000000")
    newest = _model_dir(tmp_path, "ae-20261119-160552")
    (tmp_path / "notes").mkdir()  # not a model: no metadata.json
    assert find_model(tmp_path) == newest
    assert find_model(tmp_path, "ae-20261101-000000") == tmp_path / "ae-20261101-000000"
    assert find_model(tmp_path, "ae-nope") is None


def test_from_config_without_model_scores_with_rules(tmp_path: Path) -> None:
    scorer = AnomalyScorer.from_config(AnomalyConfig(models_dir=str(tmp_path)))
    assert scorer.model is None and scorer.model_version == RULES_ONLY


def test_autoencoder_load_prefers_the_combined_threshold(tmp_path: Path) -> None:
    try:
        keras = import_keras()
    except ImportError:
        pytest.skip("keras is not installed (the `anomaly` extra)")
    path = _model_dir(tmp_path, "ae-x", combined=0.19)
    inputs = keras.Input(shape=(len(FEATURE_NAMES),))
    keras.Model(inputs, keras.layers.Dense(len(FEATURE_NAMES))(inputs)).save(path / "model.keras")
    assert Autoencoder.load(path).threshold == pytest.approx(0.19)
    own = Autoencoder.load(path, combined=False)
    assert own.threshold == pytest.approx(0.3) and own.version == "ae-x"
    x, reconstructed = own.reconstruct(np.ones((2, len(FEATURE_NAMES))))
    assert x.shape == reconstructed.shape == (2, len(FEATURE_NAMES))
    scorer = AnomalyScorer.from_config(AnomalyConfig(models_dir=str(tmp_path)))
    assert scorer.model_version == "ae-x"
