import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from ml.synthetic.gan.dataset import (
    ColumnSpec,
    NotFittedError,
    SeedDataset,
    TabularTransformer,
    make_loader,
)
from ml.synthetic.seed import FIELDS

CATEGORIES = ["food_beverage", "packaging", "it_services", "logistics"]
CURRENCIES = ["IDR", "GBP", "USD"]


def rows(n: int = 200, seed: int = 7) -> list[dict[str, Any]]:
    """Seed-like rows spanning the real ranges (cents to millions, 1-25 lines)."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        subtotal = round(10 ** rng.uniform(0.5, 6.5), 2)
        tax_rate = rng.choice([0.0, 0.06, 0.1, 0.2])
        out.append(
            {
                "line_count": rng.randint(1, 25),
                "qty_mean": round(rng.uniform(0.5, 40), 3),
                "unit_price_mean": round(10 ** rng.uniform(0, 5), 2),
                "subtotal": subtotal,
                "tax_rate": tax_rate,
                "total": round(subtotal * (1 + tax_rate), 2),
                "days_to_due": rng.choice([0, 7, 14, 30, 60]),
                "day_of_month": rng.randint(1, 31),
                "vendor_category": rng.choice(CATEGORIES),
                "currency": rng.choice(CURRENCIES),
            }
        )
    return out


def assert_rows_equal(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> None:
    assert len(expected) == len(actual)
    for want, got in zip(expected, actual, strict=True):
        assert set(got) == set(want)
        for key, value in want.items():
            if isinstance(value, str) or isinstance(value, int) and not isinstance(value, bool):
                assert got[key] == value, key
            else:
                assert math.isclose(got[key], value, rel_tol=1e-5, abs_tol=1e-6), key


def test_transform_then_inverse_gives_back_the_rows() -> None:
    data = rows()
    transformer = TabularTransformer().fit(data)

    restored = transformer.inverse_transform(transformer.transform(data))

    assert_rows_equal(data, restored)
    assert all(
        isinstance(r["line_count"], int) and isinstance(r["currency"], str) for r in restored
    )


def test_layout_of_the_transformed_vector() -> None:
    data = rows()
    transformer = TabularTransformer().fit(data)

    matrix = transformer.transform(data)

    assert matrix.dtype == np.float32
    assert transformer.output_dim == 8 + len(CATEGORIES) + len(CURRENCIES) == matrix.shape[1]
    assert [c.name for c in transformer.columns] == list(FIELDS)
    assert transformer.categorical_spans == [(8, len(CATEGORIES)), (12, len(CURRENCIES))]
    numeric = matrix[:, :8]
    assert float(np.min(numeric)) == pytest.approx(-1)
    assert float(np.max(numeric)) == pytest.approx(1)
    one_hot = matrix[:, 8:]
    assert set(np.unique(one_hot)) == {0.0, 1.0} and (one_hot.sum(axis=1) == 2).all()


def test_amounts_are_log_scaled() -> None:
    data = [{"total": 1.0}, {"total": 10_000.0}, {"total": 99.0}]
    transformer = TabularTransformer([ColumnSpec("total", "amount")]).fit(data)

    scaled = transformer.transform(data)[:, 0]

    # log1p(99) is halfway between log1p(1) and log1p(10_000) in log space, not near -1.
    assert scaled[2] == pytest.approx(
        2 * (math.log(100) - math.log(2)) / (math.log(10_001) - math.log(2)) - 1, rel=1e-5
    )
    assert scaled[2] > -0.6


def test_generator_overshoot_is_clipped_and_soft_categories_decode() -> None:
    data = rows(50)
    transformer = TabularTransformer().fit(data)
    generated = np.zeros((1, transformer.output_dim), dtype=np.float32)
    generated[0, :8] = 1.7  # tanh would never do this, but a raw output might
    currency = transformer.column("currency")
    generated[0, currency.start : currency.stop] = [0.1, 0.7, 0.2]  # Gumbel-softmax sample

    (row,) = transformer.inverse_transform(generated)

    assert row["total"] == pytest.approx(max(r["total"] for r in data), rel=1e-6)
    assert row["day_of_month"] == max(r["day_of_month"] for r in data)
    assert row["currency"] == currency.categories[1]


def test_save_and_load(tmp_path: Path) -> None:
    data = rows()
    transformer = TabularTransformer().fit(data)

    loaded = TabularTransformer.load(transformer.save(tmp_path / "gan" / "transformer.json"))

    np.testing.assert_array_equal(loaded.transform(data), transformer.transform(data))
    assert_rows_equal(data, loaded.inverse_transform(transformer.transform(data)))


def test_constant_column_maps_to_zero_and_back() -> None:
    data = [{"days_to_due": 30}, {"days_to_due": 30}]
    transformer = TabularTransformer([ColumnSpec("days_to_due", "integer")]).fit(data)

    matrix = transformer.transform(data)

    assert (matrix == 0).all()
    assert transformer.inverse_transform(matrix) == data


def test_errors() -> None:
    transformer = TabularTransformer().fit(rows(20))

    with pytest.raises(ValueError, match="unseen category 'XYZ'"):
        transformer.transform([{**rows(1)[0], "currency": "XYZ"}])
    with pytest.raises(ValueError, match="expected shape"):
        transformer.inverse_transform(np.zeros((1, 3)))
    with pytest.raises(NotFittedError):
        TabularTransformer().transform(rows(1))
    with pytest.raises(ValueError, match="amounts must be >= 0"):
        TabularTransformer([ColumnSpec("total", "amount")]).fit([{"total": -5.0}])


# --------------------------------------------------------------------------- torch


def test_dataset_yields_features_and_condition() -> None:
    dataset = SeedDataset.from_rows(rows(), condition="vendor_category")

    features, condition = dataset[0]

    assert len(dataset) == 200
    assert features.shape == (dataset.feature_dim,) and features.dtype == torch.float32
    assert condition.shape == (len(CATEGORIES),) and condition.sum() == 1
    category = dataset.transformer.column("vendor_category")
    assert torch.equal(condition, features[category.start : category.stop])


def test_unconditional_dataset_has_an_empty_condition() -> None:
    dataset = SeedDataset.from_rows(rows(10))

    assert dataset.condition_dim == 0 and dataset[3][1].shape == (0,)
    with pytest.raises(ValueError, match="categorical"):
        SeedDataset.from_rows(rows(10), condition="total")


def test_loader_batches_are_full_and_reproducible() -> None:
    dataset = SeedDataset.from_rows(rows(), condition="currency")

    batches = list(make_loader(dataset, batch_size=64, seed=3))
    again = list(make_loader(dataset, batch_size=64, seed=3))

    assert len(batches) == 3  # 200 // 64; the short last batch is dropped
    assert all(x.shape == (64, dataset.feature_dim) and c.shape == (64, 3) for x, c in batches)
    assert all(torch.equal(a[0], b[0]) for a, b in zip(batches, again, strict=True))
