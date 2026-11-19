"""Train the Keras autoencoder that scores how unusual an invoice is (layer 8, anomalies).

    python -m ml.anomaly.train_autoencoder --features data/processed/anomaly

The model is a small dense autoencoder over the scaled invoice features from
``build_features.py``: 64 -> 32 -> 8 (bottleneck) -> 32 -> 64 -> n_features, ReLU, 10 %
dropout after the first layer, mean squared error, Adam at 1e-3. Invoices it reconstructs
badly are unlike the normal invoices it learned from.

Keras 3 runs on TensorFlow, JAX or PyTorch. TensorFlow has no wheel for every Python
version (none for 3.14 yet), so when ``KERAS_BACKEND`` is not set and TensorFlow is not
installed, PyTorch is used (:func:`import_keras`); the saved ``.keras`` file loads on any
backend.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any

N_HIDDEN = (64, 32)
BOTTLENECK = 8
DROPOUT = 0.1
LEARNING_RATE = 1e-3


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
