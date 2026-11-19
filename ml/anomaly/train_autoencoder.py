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

Everything needed to score is saved under ``models/anomaly/<version>/`` (:func:`save_model`):
``model.keras``, ``scaler.json`` (and ``pca.json`` if the features used embeddings) from
the feature build, and ``metadata.json`` with the version, feature names, threshold,
training history summary, data hashes and the Keras version and backend. Before it
finishes, the run loads the folder back (:func:`load_model`) and checks that the reloaded
model gives the same validation scores.

Keras 3 runs on TensorFlow, JAX or PyTorch. TensorFlow has no wheel for every Python
version (none for 3.14 yet), so when ``KERAS_BACKEND`` is not set and TensorFlow is not
installed, PyTorch is used (:func:`import_keras`); the saved ``.keras`` file loads on any
backend.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from ledgerlens.anomaly.features import Matrix, Standardizer
from ledgerlens.anomaly.scorer import import_keras as import_keras  # same backend choice

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
    x_raw: Matrix
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
        return Split(
            np.asarray(data["x"], dtype=np.float64),
            np.asarray(data["x_raw"], dtype=np.float64),
            data["ids"],
            data["labels"],
        )


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


# --------------------------------------------------------------------------- save / load

MODEL_FILE = "model.keras"
REPRODUCE_ATOL = 1e-9


def make_version(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).strftime("ae-%Y%m%d-%H%M%S")


def save_model(
    out: Path,
    *,
    result: TrainResult,
    threshold: Threshold,
    features_dir: Path,
    config: TrainConfig,
) -> Path:
    """Write ``model.keras``, the feature build's scaler (and PCA) and ``metadata.json``
    to ``out`` (a new ``models/anomaly/<version>`` folder). Returns ``out``."""
    keras = import_keras()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty: pick a new --version")
    out.mkdir(parents=True, exist_ok=True)
    result.model.save(out / MODEL_FILE)
    for name in ("scaler.json", "pca.json"):
        if (features_dir / name).exists():
            shutil.copy(features_dir / name, out / name)
    manifest = json.loads((features_dir / "manifest.json").read_text(encoding="utf-8"))
    losses = result.history
    metadata = {
        "version": out.name,
        "created_at": datetime.now(UTC).isoformat(),
        "features": manifest["features"],
        "threshold": asdict(threshold),
        "training": {
            "config": asdict(config),
            "epochs_run": len(losses["loss"]),
            "best_epoch": result.best_epoch,
            "stopped_early": result.stopped_early,
            "best_val_loss": losses["val_loss"][result.best_epoch - 1],
            "train_loss_at_best": losses["loss"][result.best_epoch - 1],
        },
        "data": {
            "features_dir": str(features_dir),
            "source_sha256": manifest.get("source_sha256"),
            "splits": manifest.get("splits"),
        },
        "keras": {"version": keras.__version__, "backend": keras.backend.backend()},
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return out


@dataclass
class SavedModel:
    """A trained autoencoder as saved: scores raw feature vectors."""

    model: Any
    scaler: Standardizer
    threshold: float
    metadata: dict[str, Any]

    def errors(self, x_raw: Matrix) -> Matrix:
        """Reconstruction error of unscaled feature rows."""
        return reconstruction_error(self.model, self.scaler.transform(x_raw))

    def flags(self, x_raw: Matrix) -> Flags:
        flagged: Flags = self.errors(x_raw) >= self.threshold
        return flagged


def load_model(path: Path) -> SavedModel:
    keras = import_keras()
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    scaler = Standardizer.from_json(json.loads((path / "scaler.json").read_text(encoding="utf-8")))
    if list(scaler.names) != metadata["features"]:
        raise ValueError(f"{path}: scaler.json and metadata.json list different features")
    model = keras.saving.load_model(path / MODEL_FILE)
    return SavedModel(model, scaler, float(metadata["threshold"]["value"]), metadata)


def check_reproduces(path: Path, x_raw: Matrix, expected: Matrix) -> float:
    """Load ``path`` and score ``x_raw``; raise unless the errors match ``expected``.
    Returns the largest difference."""
    diff = float(np.max(np.abs(load_model(path).errors(x_raw) - expected), initial=0.0))
    if diff > REPRODUCE_ATOL:
        raise AssertionError(f"reloaded model scores differ by up to {diff:g}")
    return diff


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.anomaly.train_autoencoder",
        description="Train the invoice anomaly autoencoder.",
    )
    parser.add_argument("--features", type=Path, default=Path("data/processed/anomaly"))
    parser.add_argument("--models-dir", type=Path, default=Path("models/anomaly"))
    parser.add_argument("--version", default=None, help="folder name (default: ae-<UTC time>)")
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
    out = save_model(
        args.models_dir / (args.version or make_version()),
        result=result,
        threshold=threshold,
        features_dir=args.features,
        config=config,
    )
    diff = check_reproduces(out, val_split.x_raw, errors)
    print(f"saved {out}; reloaded scores match (max diff {diff:.1e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
