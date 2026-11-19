"""Train the Keras autoencoder that scores how unusual an invoice is (layer 8, anomalies).

    python -m ml.anomaly.train_autoencoder --features data/processed/anomaly

The model is a small dense autoencoder over the scaled invoice features from
``build_features.py``: 64 -> 32 -> 8 (bottleneck) -> 32 -> 64 -> n_features, ReLU, 10 %
dropout after the first layer, mean squared error, Adam at 1e-3. Invoices it reconstructs
badly are unlike the normal invoices it learned from.

It trains on **normal invoices only**: the ``normal`` rows of the training split. Validation
loss is computed on the validation split's normal rows. Training stops when that loss has
not improved for ``patience`` epochs, and the best weights are kept.

The flagging threshold on reconstruction error is chosen on the **labelled validation
split** (:func:`select_threshold`): the lowest error that still gives precision >= 0.8,
i.e. the most anomalies caught at that precision. If no threshold reaches it, or val has no
labels, the threshold is the 99th percentile of the normal validation invoices' errors.

Keras 3 runs on TensorFlow, JAX or PyTorch. TensorFlow has no wheel for every Python
version (none for 3.14 yet), so when ``KERAS_BACKEND`` is not set and TensorFlow is not
installed, PyTorch is used (:func:`import_keras`); the saved ``.keras`` file loads on any
backend.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from ledgerlens.anomaly.features import Matrix

Strings = npt.NDArray[np.str_]
Flags = npt.NDArray[np.bool_]

logger = logging.getLogger(__name__)

N_HIDDEN = (64, 32)
BOTTLENECK = 8
DROPOUT = 0.1
LEARNING_RATE = 1e-3
NORMAL = "normal"
TARGET_PRECISION = 0.8
FALLBACK_PERCENTILE = 99.0


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


def build_autoencoder(
    n_features: int, *, dropout: float = DROPOUT, learning_rate: float = LEARNING_RATE
) -> Any:
    """The compiled autoencoder: ``n_features`` -> 64 -> 32 -> 8 -> 32 -> 64 -> ``n_features``."""
    keras = import_keras()
    layers = keras.layers
    inputs = keras.Input(shape=(n_features,), name="features")
    x = layers.Dense(N_HIDDEN[0], activation="relu", name="encoder_1")(inputs)
    x = layers.Dropout(dropout, name="dropout")(x)
    x = layers.Dense(N_HIDDEN[1], activation="relu", name="encoder_2")(x)
    z = layers.Dense(BOTTLENECK, activation="relu", name="bottleneck")(x)
    x = layers.Dense(N_HIDDEN[1], activation="relu", name="decoder_1")(z)
    x = layers.Dense(N_HIDDEN[0], activation="relu", name="decoder_2")(x)
    outputs = layers.Dense(n_features, name="reconstruction")(x)
    model = keras.Model(inputs, outputs, name="invoice_autoencoder")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="mse")
    return model


# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class Split:
    """One ``<split>.npz`` from ``build_features.py``."""

    x: Matrix
    """Scaled features."""
    ids: Strings
    labels: Strings

    @property
    def normal(self) -> Matrix:
        rows: Matrix = self.x[self.labels == NORMAL]
        return rows

    @property
    def is_anomaly(self) -> Flags:
        flags: Flags = self.labels != NORMAL
        return flags


def load_split(features_dir: Path, split: str) -> Split:
    with np.load(features_dir / f"{split}.npz", allow_pickle=False) as data:
        return Split(np.asarray(data["x"], dtype=np.float64), data["ids"], data["labels"])


# --------------------------------------------------------------------------- training


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 300
    batch_size: int = 64
    patience: int = 15
    """Stop after this many epochs without a lower validation loss."""
    learning_rate: float = LEARNING_RATE
    dropout: float = DROPOUT
    seed: int = 13


@dataclass
class TrainResult:
    model: Any
    history: dict[str, list[float]]
    best_epoch: int
    """1-based epoch whose weights were kept."""
    stopped_early: bool


def train(x_normal: Matrix, x_val_normal: Matrix, config: TrainConfig | None = None) -> TrainResult:
    """Fit the autoencoder to reconstruct ``x_normal``, with early stopping on the
    reconstruction loss of ``x_val_normal`` (both: scaled features of normal invoices)."""
    config = config or TrainConfig()
    keras = import_keras()
    keras.utils.set_random_seed(config.seed)
    model = build_autoencoder(
        x_normal.shape[1], dropout=config.dropout, learning_rate=config.learning_rate
    )
    stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=config.patience, restore_best_weights=True
    )
    fit = model.fit(
        x_normal,
        x_normal,
        validation_data=(x_val_normal, x_val_normal),
        epochs=config.epochs,
        batch_size=config.batch_size,
        shuffle=True,
        callbacks=[stop],
        verbose=0,
    )
    history = {k: [float(v) for v in values] for k, values in fit.history.items()}
    best = int(np.argmin(history["val_loss"])) + 1
    stopped = len(history["val_loss"]) < config.epochs
    logger.info(
        "trained %d epochs (best %d, val loss %.4f)%s",
        len(history["val_loss"]),
        best,
        history["val_loss"][best - 1],
        ", stopped early" if stopped else "",
    )
    return TrainResult(model, history, best, stopped)


def reconstruction_error(model: Any, x: Matrix) -> Matrix:
    """Per-row mean squared reconstruction error."""
    reconstructed = np.asarray(model.predict(x, verbose=0), dtype=np.float64)
    errors: Matrix = np.mean((x - reconstructed) ** 2, axis=1)
    return errors


# --------------------------------------------------------------------------- threshold


@dataclass(frozen=True)
class Threshold:
    value: float
    """Flag an invoice whose reconstruction error is >= this."""
    method: Literal["target_precision", "percentile"]
    target_precision: float
    precision: float | None
    """On the validation split at ``value`` (``None`` without labelled anomalies)."""
    recall: float | None
    flagged: int


def precision_recall(
    errors: Matrix, is_anomaly: Flags, threshold: float
) -> tuple[float, float, int]:
    flagged = errors >= threshold
    n = int(flagged.sum())
    positives = int(is_anomaly.sum())
    caught = int((flagged & is_anomaly).sum())
    precision = caught / n if n else 0.0
    recall = caught / positives if positives else 0.0
    return precision, recall, n


def select_threshold(
    errors: Matrix,
    is_anomaly: Flags,
    *,
    target_precision: float = TARGET_PRECISION,
    percentile: float = FALLBACK_PERCENTILE,
) -> Threshold:
    """The lowest error threshold whose precision on ``errors`` is >= ``target_precision``
    (the most recall at that precision); else the ``percentile`` of the normal rows'
    errors."""
    errors = np.asarray(errors, dtype=np.float64)
    is_anomaly = np.asarray(is_anomaly, dtype=bool)
    if is_anomaly.any():
        order = np.argsort(-errors, kind="stable")
        ranked, hits = errors[order], is_anomaly[order]
        precision_at_k = np.cumsum(hits) / np.arange(1, len(hits) + 1)
        # only cut where the next error is lower: equal errors are flagged together
        cuttable = np.append(ranked[1:] < ranked[:-1], True)
        ok = np.flatnonzero(cuttable & (precision_at_k >= target_precision))
        if len(ok):
            value = float(ranked[ok[-1]])
            precision, recall, n = precision_recall(errors, is_anomaly, value)
            return Threshold(value, "target_precision", target_precision, precision, recall, n)
    normal = errors[~is_anomaly] if (~is_anomaly).any() else errors
    value = float(np.percentile(normal, percentile))
    if is_anomaly.any():
        precision, recall, n = precision_recall(errors, is_anomaly, value)
        return Threshold(value, "percentile", target_precision, precision, recall, n)
    return Threshold(
        value, "percentile", target_precision, None, None, int((errors >= value).sum())
    )


def percentile_threshold(
    errors: Matrix, split: Split, percentile: float = FALLBACK_PERCENTILE
) -> float:
    """The ``percentile`` of the split's normal invoices' errors."""
    return float(np.percentile(errors[~split.is_anomaly], percentile))


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.anomaly.train_autoencoder",
        description="Train the invoice anomaly autoencoder.",
    )
    parser.add_argument("--features", type=Path, default=Path("data/processed/anomaly"))
    defaults = TrainConfig()
    for name, value in asdict(defaults).items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = TrainConfig(**{k: getattr(args, k) for k in asdict(TrainConfig())})
    train_split, val_split = load_split(args.features, "train"), load_split(args.features, "val")
    result = train(train_split.normal, val_split.normal, config)
    best_loss = result.history["val_loss"][result.best_epoch - 1]
    print(f"best epoch {result.best_epoch}: val loss {best_loss:.4f}")
    errors = reconstruction_error(result.model, val_split.x)
    threshold = select_threshold(errors, val_split.is_anomaly)
    print(
        f"threshold {threshold.value:.4f} ({threshold.method}): val precision "
        f"{threshold.precision}, recall {threshold.recall}, {threshold.flagged} flagged"
    )
    for name, value in (
        ("chosen", threshold.value),
        ("p99", percentile_threshold(errors, val_split)),
    ):
        flagged = errors >= value
        per_type = ", ".join(
            f"{label} {int(flagged[rows].sum())}/{int(rows.sum())}"
            for label in sorted(set(val_split.labels.tolist()) - {NORMAL})
            if (rows := val_split.labels == label).any()
        )
        print(f"  {name} {value:.4f}: {per_type}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
