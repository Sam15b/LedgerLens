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

Serving: :class:`AnomalyScorer` holds the loaded model (:class:`Autoencoder`) and scores
one invoice at a time. :func:`get_scorer` loads it once per process (a worker scores many
invoices with one model), from ``configs/anomaly.yaml``. Without a saved model, or without
Keras installed, the scorer runs the rules alone and says so in its ``model_version``
(``rules-only``).
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from ledgerlens.anomaly.features import (
    FEATURE_NAMES,
    Standardizer,
    VendorHistory,
    feature_vector,
    invoice_features,
)
from ledgerlens.anomaly.rules import InvoiceRef, RuleContext, RuleHit, check_invoice
from ledgerlens.core.config import AnomalyConfig, get_config, repo_path

Matrix = npt.NDArray[np.float64]
Flags = npt.NDArray[np.bool_]

logger = logging.getLogger(__name__)

RULE_BAND = 0.5
"""Hard hits score at least this; invoices without one score below it."""
RULES_ONLY = "rules-only"
"""``model_version`` of scores made without an autoencoder."""


def ae_score(error: float | None, threshold: float) -> float:
    """``error / (error + threshold)``: 0 for a perfect reconstruction, 0.5 at the
    threshold, towards 1 far above it. 0 without an error (no model)."""
    if error is None or error <= 0:
        return 0.0
    if threshold <= 0:
        return 1.0
    return float(error / (error + threshold))


@dataclass(frozen=True)
class FeatureError:
    """How badly the autoencoder reconstructed one feature of one invoice."""

    name: str
    value: float
    """The invoice's raw (unscaled) value."""
    error: float
    """Squared error on the scaled feature: its part of the reconstruction error."""
    above: bool
    """The invoice's value is higher than the model expected."""


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
    feature_errors: tuple[FeatureError, ...] = ()
    """Per feature, largest error first (empty without the autoencoder)."""

    @property
    def hard_hit(self) -> bool:
        return bool(self.rule_hits)

    @property
    def ae_flagged(self) -> bool:
        return self.ae_error is not None and self.ae_error >= self.ae_threshold


def combine(
    rule_hits: Sequence[RuleHit],
    error: float | None,
    threshold: float,
    feature_errors: Sequence[FeatureError] = (),
) -> AnomalyScore:
    """The rules' hits and the autoencoder's error on one invoice as one score."""
    a = ae_score(error, threshold)
    hit = bool(rule_hits)
    return AnomalyScore(
        score=RULE_BAND + a / 2 if hit else a / 2,
        flagged=hit or (error is not None and error >= threshold),
        ae_error=error,
        ae_threshold=threshold,
        rule_hits=tuple(rule_hits),
        feature_errors=tuple(feature_errors),
    )


def combine_arrays(hard_hits: Flags, errors: Matrix, threshold: float) -> tuple[Matrix, Flags]:
    """:func:`combine` over many invoices at once: ``(scores, flags)``."""
    hard_hits = np.asarray(hard_hits, dtype=bool)
    errors = np.asarray(errors, dtype=np.float64)
    a = np.where(errors > 0, errors / (errors + threshold), 0.0)
    scores: Matrix = np.where(hard_hits, RULE_BAND + a / 2, a / 2)
    flags: Flags = hard_hits | (errors >= threshold)
    return scores, flags


# --------------------------------------------------------------------------- the model


def import_keras() -> Any:
    """``keras``, on TensorFlow if it is installed, else PyTorch (unless ``KERAS_BACKEND``
    says otherwise). Must run before anything else imports keras."""
    if (
        "KERAS_BACKEND" not in os.environ
        and importlib.util.find_spec("tensorflow") is None
        and importlib.util.find_spec("torch") is not None
    ):
        os.environ["KERAS_BACKEND"] = "torch"
    import keras

    return keras


class ReconstructionModel(Protocol):
    """What the scorer needs from a model: :class:`Autoencoder`, or a stand-in in tests."""

    @property
    def version(self) -> str: ...
    @property
    def threshold(self) -> float: ...
    @property
    def features(self) -> tuple[str, ...]: ...

    def reconstruct(self, x_raw: Matrix) -> tuple[Matrix, Matrix]:
        """``(scaled input, reconstruction)`` of unscaled feature rows."""
        ...


MODEL_FILE = "model.keras"


@dataclass(frozen=True)
class Autoencoder:
    """A saved ``models/anomaly/<version>/`` folder, loaded."""

    model: Any
    scaler: Standardizer
    version: str
    threshold: float
    """Flag at this reconstruction error: the rules + AE cut if one was saved (and wanted),
    else the autoencoder's own threshold."""
    features: tuple[str, ...]
    metadata: dict[str, Any]

    @classmethod
    def load(cls, path: Path, *, combined: bool = True) -> Autoencoder:
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        scaler = Standardizer.from_json(
            json.loads((path / "scaler.json").read_text(encoding="utf-8"))
        )
        features = tuple(metadata["features"])
        if scaler.names != features:
            raise ValueError(f"{path}: scaler.json and metadata.json list different features")
        if features != FEATURE_NAMES:
            # the PCA text features need the header chunk's embedding at scoring time
            raise ValueError(f"{path}: the scorer computes {len(FEATURE_NAMES)} features only")
        saved = metadata.get("combined") if combined else None
        threshold = float((saved or metadata["threshold"])["value"])
        model = import_keras().saving.load_model(path / MODEL_FILE)
        return cls(model, scaler, str(metadata["version"]), threshold, features, metadata)

    def reconstruct(self, x_raw: Matrix) -> tuple[Matrix, Matrix]:
        x = self.scaler.transform(x_raw)
        return x, np.asarray(self.model.predict(x, verbose=0), dtype=np.float64)


def find_model(models_dir: Path, version: str = "latest") -> Path | None:
    """``models_dir/<version>``, or for ``latest`` the newest folder with a
    ``metadata.json`` (names are ``ae-<UTC time>``, so the last one sorted). ``None`` if
    there is none."""
    if version != "latest":
        path = models_dir / version
        return path if (path / "metadata.json").exists() else None
    if not models_dir.is_dir():
        return None
    found = sorted(p for p in models_dir.iterdir() if (p / "metadata.json").exists())
    return found[-1] if found else None


# --------------------------------------------------------------------------- scorer


class AnomalyScorer:
    """Scores invoices with the rules and, if it has one, the autoencoder."""

    def __init__(self, model: ReconstructionModel | None = None) -> None:
        self.model = model

    @property
    def model_version(self) -> str:
        return self.model.version if self.model is not None else RULES_ONLY

    @classmethod
    def from_config(cls, config: AnomalyConfig | None = None) -> AnomalyScorer:
        """The configured model version; rules only if it is missing or will not load."""
        config = config or get_config().anomaly
        path = find_model(repo_path(config.models_dir), config.version)
        if path is None:
            logger.warning(
                "no anomaly model %r in %s: scoring with the rules only",
                config.version,
                config.models_dir,
            )
            return cls(None)
        try:
            model = Autoencoder.load(path, combined=config.use_combined_threshold)
        except ImportError as exc:
            logger.warning("cannot load %s (%s): scoring with the rules only", path, exc)
            return cls(None)
        logger.info("anomaly model %s (threshold %.4f)", model.version, model.threshold)
        return cls(model)

    def autoencoder(
        self, ref: InvoiceRef, history: VendorHistory
    ) -> tuple[float | None, float, tuple[FeatureError, ...]]:
        """``(error, threshold, per-feature errors)``; ``(None, 0.0, ())`` without a model."""
        if self.model is None:
            return None, 0.0, ()
        values = invoice_features(ref.invoice, history)
        x_raw = np.array([feature_vector(values, self.model.features)], dtype=np.float64)
        x, reconstructed = self.model.reconstruct(x_raw)
        diff = x[0] - reconstructed[0]
        squared = diff**2
        per_feature = sorted(
            (
                FeatureError(name, float(x_raw[0, i]), float(squared[i]), bool(diff[i] > 0))
                for i, name in enumerate(self.model.features)
            ),
            key=lambda f: -f.error,
        )
        return float(squared.mean()), self.model.threshold, tuple(per_feature)

    def score(
        self, ref: InvoiceRef, context: RuleContext, history: VendorHistory | None = None
    ) -> AnomalyScore:
        """One invoice: every rule against ``context``, the autoencoder on its features
        with its vendor's ``history`` (what was known before it), combined."""
        hits = check_invoice(ref, context)
        error, threshold, per_feature = self.autoencoder(ref, history or VendorHistory())
        return combine(hits, error, threshold, per_feature)


@lru_cache(maxsize=1)
def get_scorer() -> AnomalyScorer:
    """The process's scorer: the model is loaded on first use, then kept."""
    return AnomalyScorer.from_config()
