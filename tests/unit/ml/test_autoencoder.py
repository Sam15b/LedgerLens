"""Anomaly autoencoder: model shape, threshold choice, and a train -> save -> load smoke run."""

import json
from pathlib import Path

import numpy as np
import pytest

from ledgerlens.anomaly.features import Standardizer
from ml.anomaly.train_autoencoder import (
    TrainConfig,
    build_autoencoder,
    check_reproduces,
    import_keras,
    load_model,
    load_split,
    main,
    make_version,
    reconstruction_error,
    save_model,
    select_threshold,
    train,
)

try:
    import_keras()  # picks the backend first: a bare `import keras` would want TensorFlow
except ImportError:
    pytest.skip("keras is not installed (the `anomaly` extra)", allow_module_level=True)

N_FEATURES = 6


def test_model_shape_and_bottleneck() -> None:
    model = build_autoencoder(N_FEATURES)
    widths = [layer.units for layer in model.layers if hasattr(layer, "units")]
    assert widths == [64, 32, 8, 32, 64, N_FEATURES]
    assert model.get_layer("bottleneck").units == 8
    assert model.loss == "mse"


# --------------------------------------------------------------------------- threshold


def test_threshold_reaches_target_precision_with_most_recall() -> None:
    errors = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    anomaly = np.array([1, 1, 0, 1, 1, 0, 0, 1, 0, 0], dtype=bool)

    t = select_threshold(errors, anomaly, target_precision=0.8)

    # top 5: 4 anomalies -> precision 0.8; top 8 would only reach 5/8
    assert (t.value, t.method, t.flagged) == (0.5, "target_precision", 5)
    assert t.precision == pytest.approx(0.8) and t.recall == pytest.approx(0.8)


def test_threshold_does_not_split_tied_errors() -> None:
    errors = np.array([0.9, 0.5, 0.5, 0.1])
    anomaly = np.array([1, 1, 0, 0], dtype=bool)
    # cutting between the two 0.5s would claim precision 1.0 at 2 flags
    t = select_threshold(errors, anomaly, target_precision=0.9)
    assert (t.value, t.flagged) == (0.9, 1)


def test_threshold_falls_back_to_the_normal_percentile() -> None:
    rng = np.random.default_rng(0)
    errors = rng.random(200)
    unreachable = select_threshold(errors, np.zeros(200, dtype=bool) | (np.arange(200) == 0))
    assert unreachable.method in {"target_precision", "percentile"}

    no_labels = select_threshold(errors, np.zeros(200, dtype=bool), percentile=95)
    assert no_labels.method == "percentile" and no_labels.precision is None
    assert no_labels.value == pytest.approx(np.percentile(errors, 95))

    noisy = select_threshold(np.array([0.9, 0.8, 0.1]), np.array([0, 0, 1], dtype=bool))
    assert noisy.method == "percentile" and noisy.recall == 0.0


# --------------------------------------------------------------------------- smoke run


def write_features(out: Path, seed: int = 0) -> None:
    """A tiny build_features.py output: correlated normal rows, a few far-off anomalies."""
    rng = np.random.default_rng(seed)
    names = [f"f{i}" for i in range(N_FEATURES)]

    def rows(n: int, anomalies: int) -> tuple[np.ndarray, np.ndarray]:
        base = rng.normal(size=(n, 2))
        x = np.hstack([base, base * 2, base.sum(axis=1, keepdims=True), -base[:, :1]])
        x[:anomalies] += rng.choice([-6.0, 6.0], size=(anomalies, N_FEATURES))
        labels = np.array(["round_amount"] * anomalies + ["normal"] * (n - anomalies))
        return x, labels

    splits = {"train": rows(160, 8), "val": rows(60, 6), "test": rows(60, 6)}
    train_x, train_labels = splits["train"]
    scaler = Standardizer.fit(train_x[train_labels == "normal"], names)
    out.mkdir(parents=True, exist_ok=True)
    for split, (x, labels) in splits.items():
        ids = np.array([f"{split}-{i}" for i in range(len(x))])
        np.savez(out / f"{split}.npz", x=scaler.transform(x), x_raw=x, ids=ids, labels=labels)
    (out / "scaler.json").write_text(json.dumps(scaler.to_json()), encoding="utf-8")
    manifest = {"features": names, "source_sha256": "test", "splits": {}}
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


QUICK = TrainConfig(epochs=40, batch_size=16, patience=5, seed=3)


def test_training_saves_and_reloads_the_same_scores(tmp_path: Path) -> None:
    features = tmp_path / "features"
    write_features(features)
    train_split, val_split = load_split(features, "train"), load_split(features, "val")

    result = train(train_split.normal, val_split.normal, QUICK)

    assert 1 <= result.best_epoch <= len(result.history["val_loss"]) <= QUICK.epochs
    assert result.history["val_loss"][-1] < result.history["val_loss"][0] * 2
    errors = reconstruction_error(result.model, val_split.x)
    # the far-off rows reconstruct worst
    assert set(np.argsort(-errors)[:6]) == set(np.flatnonzero(val_split.is_anomaly))
    threshold = select_threshold(errors, val_split.is_anomaly)
    # the most recall at precision >= 0.8: every anomaly, maybe a normal row or two
    assert threshold.recall == 1.0 and threshold.precision is not None
    assert threshold.precision >= 0.8

    out = save_model(
        tmp_path / "models" / "v1",
        result=result,
        threshold=threshold,
        features_dir=features,
        config=QUICK,
    )
    assert sorted(p.name for p in out.iterdir()) == ["metadata.json", "model.keras", "scaler.json"]
    assert check_reproduces(out, val_split.x_raw, errors) <= 1e-9

    saved = load_model(out)
    assert saved.threshold == threshold.value
    assert saved.metadata["features"] == [f"f{i}" for i in range(N_FEATURES)]
    assert saved.metadata["training"]["best_epoch"] == result.best_epoch
    flags = saved.flags(val_split.x_raw)
    assert int(flags.sum()) == threshold.flagged and flags[val_split.is_anomaly].all()
    with pytest.raises(FileExistsError):
        save_model(out, result=result, threshold=threshold, features_dir=features, config=QUICK)


def test_same_seed_same_model(tmp_path: Path) -> None:
    write_features(tmp_path)
    split = load_split(tmp_path, "train")
    a = reconstruction_error(train(split.normal, split.normal, QUICK).model, split.x)
    b = reconstruction_error(train(split.normal, split.normal, QUICK).model, split.x)
    assert np.allclose(a, b)


def test_cli_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_features(tmp_path / "features")
    args = ["--features", str(tmp_path / "features"), "--models-dir", str(tmp_path / "models")]
    assert main([*args, "--version", "cli", "--epochs", "20", "--patience", "3"]) == 0

    printed = capsys.readouterr().out
    assert "reloaded scores match" in printed and "p99" in printed
    assert (tmp_path / "models" / "cli" / "model.keras").exists()
    assert make_version().startswith("ae-")
