"""Seed rows <-> the numeric matrix the GAN trains on, and back.

:class:`TabularTransformer` turns each seed row into one float vector:

* **amounts** (``subtotal``, ``total``, ``unit_price_mean``) are ``log1p``-transformed
  first -- invoice amounts span five orders of magnitude (a 3,100 IDR coffee to a
  50,000 GBP consulting bill) and a GAN cannot learn that on a linear scale;
* every numeric column is then min-max scaled to ``[-1, 1]`` (the generator's ``tanh``
  range); integers are rounded back on the way out;
* categoricals (``vendor_category``, ``currency``) are one-hot.

The fitted transformer saves to JSON, so generated vectors can be turned back into rows
long after training; :attr:`TabularTransformer.columns` tells the generator where each
categorical block sits, for its Gumbel-softmax heads.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

Kind = Literal["amount", "continuous", "integer", "categorical"]
Matrix = npt.NDArray[np.float32]


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    kind: Kind


SEED_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("line_count", "integer"),
    ColumnSpec("qty_mean", "continuous"),
    ColumnSpec("unit_price_mean", "amount"),
    ColumnSpec("subtotal", "amount"),
    ColumnSpec("tax_rate", "continuous"),
    ColumnSpec("total", "amount"),
    ColumnSpec("days_to_due", "integer"),
    ColumnSpec("day_of_month", "integer"),
    ColumnSpec("vendor_category", "categorical"),
    ColumnSpec("currency", "categorical"),
)


@dataclass
class ColumnInfo:
    """Where a column lives in the transformed vector, and how to undo its scaling."""

    name: str
    kind: Kind
    start: int
    width: int
    low: float = 0.0  # numeric: min after log (amounts) -- maps to -1
    high: float = 0.0  # numeric: max after log -- maps to +1
    categories: list[str] = field(default_factory=list)

    @property
    def stop(self) -> int:
        return self.start + self.width


class NotFittedError(RuntimeError):
    pass


class TabularTransformer:
    """Reversible preprocessing for the seed table (see the module docstring)."""

    def __init__(self, specs: Sequence[ColumnSpec] = SEED_COLUMNS) -> None:
        self.specs = list(specs)
        self.columns: list[ColumnInfo] = []

    # ------------------------------------------------------------------ fit

    def fit(self, rows: Sequence[Mapping[str, Any]]) -> TabularTransformer:
        if not rows:
            raise ValueError("cannot fit on no rows")
        columns, start = [], 0
        for spec in self.specs:
            values = [row[spec.name] for row in rows]
            if spec.kind == "categorical":
                categories = sorted({str(v) for v in values})
                columns.append(
                    ColumnInfo(spec.name, spec.kind, start, len(categories), categories=categories)
                )
                start += len(categories)
            else:
                raw = np.array(
                    [self._forward(spec.kind, float(v)) for v in values], dtype=np.float64
                )
                columns.append(
                    ColumnInfo(
                        spec.name, spec.kind, start, 1, float(np.min(raw)), float(np.max(raw))
                    )
                )
                start += 1
        self.columns = columns
        return self

    @property
    def output_dim(self) -> int:
        self._require_fit()
        return self.columns[-1].stop

    def column(self, name: str) -> ColumnInfo:
        for info in self.columns:
            if info.name == name:
                return info
        raise KeyError(name)

    @property
    def categorical_spans(self) -> list[tuple[int, int]]:
        """``(start, width)`` of every one-hot block, in order."""
        return [(c.start, c.width) for c in self.columns if c.kind == "categorical"]

    # ------------------------------------------------------------------ transform

    def transform(self, rows: Sequence[Mapping[str, Any]]) -> Matrix:
        self._require_fit()
        out = np.zeros((len(rows), self.output_dim), dtype=np.float64)
        for info in self.columns:
            values = [row[info.name] for row in rows]
            if info.kind == "categorical":
                index = {c: i for i, c in enumerate(info.categories)}
                for r, value in enumerate(values):
                    try:
                        out[r, info.start + index[str(value)]] = 1.0
                    except KeyError:
                        raise ValueError(f"{info.name}: unseen category {value!r}") from None
            else:
                raw = np.array([self._forward(info.kind, float(v)) for v in values])
                out[:, info.start] = self._scale(raw, info)
        return out.astype(np.float32)

    def inverse_transform(self, matrix: npt.NDArray[np.floating[Any]]) -> list[dict[str, Any]]:
        """Rows back from vectors. Numeric values outside ``[-1, 1]`` (a generator can
        overshoot) are clipped to the range seen in training; a categorical block decodes
        to its largest entry, so soft (Gumbel-softmax) outputs work too."""
        self._require_fit()
        data = np.asarray(matrix, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != self.output_dim:
            raise ValueError(f"expected shape (n, {self.output_dim}), got {data.shape}")
        rows: list[dict[str, Any]] = [{} for _ in range(data.shape[0])]
        for info in self.columns:
            block = data[:, info.start : info.stop]
            if info.kind == "categorical":
                for row, i in zip(rows, block.argmax(axis=1), strict=True):
                    row[info.name] = info.categories[int(i)]
                continue
            raw = self._unscale(np.clip(block[:, 0], -1.0, 1.0), info)
            for row, value in zip(rows, raw, strict=True):
                row[info.name] = self._backward(info.kind, float(value))
        return rows

    # ------------------------------------------------------------------ persistence

    def to_dict(self) -> dict[str, Any]:
        self._require_fit()
        return {
            "version": 1,
            "specs": [asdict(s) for s in self.specs],
            "columns": [asdict(c) for c in self.columns],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TabularTransformer:
        transformer = cls([ColumnSpec(**s) for s in data["specs"]])
        transformer.columns = [ColumnInfo(**c) for c in data["columns"]]
        return transformer

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> TabularTransformer:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _forward(kind: Kind, value: float) -> float:
        if kind == "amount":
            if value < 0:
                raise ValueError(f"amounts must be >= 0 for log scaling, got {value}")
            return math.log1p(value)
        return value

    @staticmethod
    def _backward(kind: Kind, value: float) -> float | int:
        if kind == "amount":
            return math.expm1(value)
        if kind == "integer":
            return int(round(value))
        return value

    @staticmethod
    def _scale(raw: npt.NDArray[np.float64], info: ColumnInfo) -> npt.NDArray[np.float64]:
        span = info.high - info.low
        if span == 0:
            return np.zeros_like(raw)
        return 2.0 * (raw - info.low) / span - 1.0

    @staticmethod
    def _unscale(scaled: npt.NDArray[np.float64], info: ColumnInfo) -> npt.NDArray[np.float64]:
        return (scaled + 1.0) / 2.0 * (info.high - info.low) + info.low

    def _require_fit(self) -> None:
        if not self.columns:
            raise NotFittedError("TabularTransformer is not fitted: call fit() or load()")
