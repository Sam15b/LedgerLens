"""How close are generated rows to the real seed rows?

    python -m ml.synthetic.gan.evaluate ml/synthetic/gan/runs/<run>/final.pt -n 5000 \\
        --report eval/reports/2026-11-13/gan.md

Metrics, per column and for the table as a whole:

* **KS statistic** (numeric columns): the largest gap between the real and synthetic
  empirical CDFs, from 0 (same distribution) to 1 (no overlap). It only depends on
  order, so it reads the same on raw or log-scaled amounts.
* **Wasserstein-1** (numeric columns), measured in the GAN's scaled space -- amounts
  ``log1p``-ed, every column min-max scaled to ``[-1, 1]`` -- so the numbers are comparable
  across columns (a raw-scale distance would be dominated by IDR amounts).
* **Total variation distance** (categorical columns): half the L1 distance between the
  category frequencies, from 0 to 1 -- per column, and **jointly** over all categorical
  columns, which catches pairs a real document never has (a marketing agency billing in
  IDR when no IDR vendor sells marketing).
* **Correlation difference**: the Pearson correlation matrices of the scaled numeric
  columns, real vs synthetic; the mean and largest absolute difference off the diagonal.
* **Rule violations**: the share of synthetic rows breaking the business rules
  (:func:`ml.synthetic.gan.sample.rule_violations`) -- should be 0.

A distance only means something next to a baseline, so the report also scores **two
random halves of the real rows** against each other: that is the noise floor from sample
size alone. The KS and Wasserstein functions match ``scipy.stats`` and need only NumPy.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # run as a script: make the repo root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import numpy.typing as npt

from ml.synthetic.gan.dataset import TabularTransformer

Array = npt.NDArray[np.float64]


def ks_statistic(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Two-sample Kolmogorov-Smirnov statistic ``sup |F_a(x) - F_b(x)|``."""
    x, y = np.sort(np.asarray(a, dtype=np.float64)), np.sort(np.asarray(b, dtype=np.float64))
    if not len(x) or not len(y):
        raise ValueError("KS needs two non-empty samples")
    grid = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, grid, side="right") / len(x)
    cdf_y = np.searchsorted(y, grid, side="right") / len(y)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def wasserstein(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """1-D Wasserstein-1 (earth mover's) distance: the area between the two CDFs."""
    x, y = np.sort(np.asarray(a, dtype=np.float64)), np.sort(np.asarray(b, dtype=np.float64))
    if not len(x) or not len(y):
        raise ValueError("Wasserstein needs two non-empty samples")
    grid = np.sort(np.concatenate([x, y]))
    widths = np.diff(grid)
    cdf_x = np.searchsorted(x, grid[:-1], side="right") / len(x)
    cdf_y = np.searchsorted(y, grid[:-1], side="right") / len(y)
    return float(np.sum(np.abs(cdf_x - cdf_y) * widths))


def total_variation(a: Sequence[str], b: Sequence[str]) -> float:
    """Half the L1 distance between two category frequency tables."""
    if not a or not b:
        raise ValueError("total variation needs two non-empty samples")
    pa, pb = Counter(a), Counter(b)
    return 0.5 * sum(abs(pa[c] / len(a) - pb[c] / len(b)) for c in pa.keys() | pb.keys())


@dataclass(frozen=True)
class ColumnMetric:
    column: str
    kind: str
    ks: float | None = None  # numeric only
    wasserstein: float | None = None  # numeric only, scaled space
    tvd: float | None = None  # categorical only


@dataclass(frozen=True)
class CorrelationDiff:
    mean_abs: float
    max_abs: float
    worst_pair: tuple[str, str]


@dataclass
class QualityReport:
    n_real: int
    n_synthetic: int
    columns: list[ColumnMetric]
    correlation: CorrelationDiff
    joint_tvd: float | None = None  # all categorical columns together
    violation_rate: float = 0.0
    violation_examples: list[str] = field(default_factory=list)

    def metric(self, column: str) -> ColumnMetric:
        return next(c for c in self.columns if c.column == column)

    @property
    def mean_ks(self) -> float:
        return _mean([c.ks for c in self.columns if c.ks is not None])

    @property
    def mean_wasserstein(self) -> float:
        return _mean([c.wasserstein for c in self.columns if c.wasserstein is not None])

    @property
    def mean_tvd(self) -> float:
        return _mean([c.tvd for c in self.columns if c.tvd is not None])


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _scaled(
    transformer: TabularTransformer, rows: Sequence[Mapping[str, Any]]
) -> tuple[Array, list[str]]:
    """Numeric columns of ``rows`` in the GAN's scaled space, and their names."""
    matrix = transformer.transform(rows).astype(np.float64)
    numeric = [c for c in transformer.columns if c.kind != "categorical"]
    return matrix[:, [c.start for c in numeric]], [c.name for c in numeric]


def correlation_difference(
    transformer: TabularTransformer,
    real: Sequence[Mapping[str, Any]],
    synthetic: Sequence[Mapping[str, Any]],
) -> CorrelationDiff:
    """Off-diagonal difference of the Pearson correlation matrices (scaled numeric columns).

    A column that is constant in one table has no correlation; it is counted as 0.
    """
    a, names = _scaled(transformer, real)
    b, _ = _scaled(transformer, synthetic)
    diff = np.abs(_corr(a) - _corr(b))
    upper = np.triu_indices(len(names), k=1)
    values = diff[upper]
    worst = int(np.argmax(values))
    return CorrelationDiff(
        mean_abs=float(values.mean()),
        max_abs=float(values[worst]),
        worst_pair=(names[upper[0][worst]], names[upper[1][worst]]),
    )


def _corr(matrix: Array) -> Array:
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(matrix, rowvar=False)
    return np.nan_to_num(np.asarray(corr, dtype=np.float64))


def compare(
    transformer: TabularTransformer,
    real: Sequence[Mapping[str, Any]],
    synthetic: Sequence[Mapping[str, Any]],
    *,
    check_rules: bool = True,
) -> QualityReport:
    """Score ``synthetic`` rows against ``real`` ones (see the module docstring)."""
    if not real or not synthetic:
        raise ValueError("need real and synthetic rows")
    a, names = _scaled(transformer, real)
    b, _ = _scaled(transformer, synthetic)
    columns: list[ColumnMetric] = []
    numeric = iter(range(len(names)))
    for info in transformer.columns:
        if info.kind == "categorical":
            tvd = total_variation(
                [str(r[info.name]) for r in real], [str(r[info.name]) for r in synthetic]
            )
            columns.append(ColumnMetric(info.name, info.kind, tvd=tvd))
            continue
        i = next(numeric)
        columns.append(
            ColumnMetric(
                info.name,
                info.kind,
                ks=ks_statistic(a[:, i], b[:, i]),
                wasserstein=wasserstein(a[:, i], b[:, i]),
            )
        )
    categorical = [c.name for c in transformer.columns if c.kind == "categorical"]
    joint = None
    if len(categorical) > 1:
        joint = total_variation(
            ["|".join(str(r[c]) for c in categorical) for r in real],
            ["|".join(str(r[c]) for c in categorical) for r in synthetic],
        )
    report = QualityReport(
        n_real=len(real),
        n_synthetic=len(synthetic),
        columns=columns,
        correlation=correlation_difference(transformer, real, synthetic),
        joint_tvd=joint,
    )
    if check_rules:
        from ml.synthetic.gan.sample import rule_violations

        broken = [(row, rule_violations(row)) for row in synthetic]
        broken = [(row, issues) for row, issues in broken if issues]
        report.violation_rate = len(broken) / len(synthetic)
        report.violation_examples = [issues[0] for _, issues in broken[:5]]
    return report


def split_halves(
    rows: Sequence[Mapping[str, Any]], *, seed: int = 0
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Two random halves of ``rows`` (the real-vs-real baseline)."""
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    middle = len(shuffled) // 2
    return shuffled[:middle], shuffled[middle:]


def baseline(
    transformer: TabularTransformer, real: Sequence[Mapping[str, Any]], *, seed: int = 0
) -> QualityReport:
    """Real half vs real half: the distances sample size alone produces."""
    first, second = split_halves(real, seed=seed)
    return compare(transformer, first, second, check_rules=False)


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_markdown(
    synthetic: QualityReport,
    reference: QualityReport,
    *,
    title: str,
    run_date: date,
    notes: Sequence[str] = (),
) -> str:
    lines = [
        f"# {title} — {run_date.isoformat()}",
        "",
        *notes,
        "",
        f"Synthetic: {synthetic.n_synthetic} sampled rows vs {synthetic.n_real} real seed rows. "
        f"Baseline: two random halves of the real rows ({reference.n_real} vs "
        f"{reference.n_synthetic}), i.e. the distance sample size alone gives. KS and "
        "total variation run 0-1; Wasserstein is on the GAN's scaled space "
        "(log amounts, min-max to [-1, 1]).",
        "",
        "## Per column",
        "",
        "| Column | Kind | KS (synthetic) | KS (baseline) | W1 (synthetic) | W1 (baseline) "
        "| TVD (synthetic) | TVD (baseline) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for col in synthetic.columns:
        ref = reference.metric(col.column)
        lines.append(
            f"| {col.column} | {col.kind} | {_fmt(col.ks)} | {_fmt(ref.ks)} "
            f"| {_fmt(col.wasserstein)} | {_fmt(ref.wasserstein)} "
            f"| {_fmt(col.tvd)} | {_fmt(ref.tvd)} |"
        )
    corr, ref_corr = synthetic.correlation, reference.correlation
    lines += [
        "",
        "## Summary",
        "",
        "| Metric | Synthetic | Baseline |",
        "|---|---:|---:|",
        f"| Mean KS (numeric) | {synthetic.mean_ks:.3f} | {reference.mean_ks:.3f} |",
        f"| Mean W1 (numeric, scaled) | {synthetic.mean_wasserstein:.3f} "
        f"| {reference.mean_wasserstein:.3f} |",
        f"| Mean TVD (categorical) | {synthetic.mean_tvd:.3f} | {reference.mean_tvd:.3f} |",
        f"| Joint TVD (all categoricals) | {_fmt(synthetic.joint_tvd)} "
        f"| {_fmt(reference.joint_tvd)} |",
        f"| Correlation diff, mean abs | {corr.mean_abs:.3f} | {ref_corr.mean_abs:.3f} |",
        f"| Correlation diff, max abs | {corr.max_abs:.3f} ({' × '.join(corr.worst_pair)}) "
        f"| {ref_corr.max_abs:.3f} |",
        f"| Rows breaking business rules | {synthetic.violation_rate:.1%} | — |",
    ]
    if synthetic.violation_examples:
        lines += ["", "Examples:", "", *(f"- {e}" for e in synthetic.violation_examples)]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    from ml.synthetic.gan.sample import category_mix, sample_rows
    from ml.synthetic.gan.train import SEED_CSV, load_generator, load_seed_rows

    parser = argparse.ArgumentParser(description="Score GAN samples against the seed rows")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("-n", type=int, default=5000, help="rows to sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-csv", type=Path, default=SEED_CSV)
    parser.add_argument("--report", type=Path, help="write the markdown report here")
    parser.add_argument("--json", type=Path, help="also write the metrics as JSON")
    args = parser.parse_args(argv)

    generator, transformer = load_generator(args.checkpoint)
    real, source = load_seed_rows(args.seed_csv)
    categories = category_mix(real, args.n, seed=args.seed) if generator.cond_dim else None
    synthetic = sample_rows(generator, transformer, args.n, categories=categories, seed=args.seed)
    result = compare(transformer, real, synthetic)
    reference = baseline(transformer, real, seed=args.seed)
    kind = "conditional on vendor_category" if generator.cond_dim else "unconditional"
    report = render_markdown(
        result,
        reference,
        title="GAN quality",
        run_date=date.today(),
        notes=[
            f"Checkpoint `{args.checkpoint.name}` ({kind}); seed rows by source: "
            f"{source['by_source']}."
        ],
    )
    print(report)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
    if args.json:
        payload = {"synthetic": _as_dict(result), "baseline": _as_dict(reference)}
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return 0


def _as_dict(report: QualityReport) -> dict[str, Any]:
    return {
        "n_real": report.n_real,
        "n_synthetic": report.n_synthetic,
        "columns": [vars(c) for c in report.columns],
        "correlation": vars(report.correlation),
        "joint_tvd": report.joint_tvd,
        "violation_rate": report.violation_rate,
    }


if __name__ == "__main__":
    raise SystemExit(main())
