"""Latency and cost as reproducible numbers.

Latency: :func:`latency_summary` gives count, mean, p50, p95 and max of a list of
timings, and :func:`stage_latencies` does it per retrieval stage (``dense``, ``rerank``,
``graph``, ...) from :attr:`RetrievalResult.timings_ms`. Percentiles interpolate
linearly between the two nearest values, so p95 of 40 timings is not just the 38th.

    python -m eval.latency_cost --results eval/reports/<date>/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def percentile(values: Iterable[float], pct: float) -> float:
    """The ``pct`` percentile (0-100) by linear interpolation; 0.0 for no values."""
    if not 0 <= pct <= 100:
        raise ValueError(f"pct must be between 0 and 100, not {pct}")
    items = sorted(values)
    if not items:
        return 0.0
    position = (len(items) - 1) * pct / 100
    low = math.floor(position)
    high = min(low + 1, len(items) - 1)
    return items[low] + (items[high] - items[low]) * (position - low)


@dataclass(frozen=True)
class LatencySummary:
    """Timings in milliseconds."""

    n: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 1) if isinstance(v, float) else v for k, v in asdict(self).items()}


def latency_summary(values: Iterable[float | None]) -> LatencySummary:
    """Count, mean, p50, p95 and max of ``values`` (``None`` entries are skipped)."""
    items = [float(v) for v in values if v is not None]
    if not items:
        return LatencySummary(0, 0.0, 0.0, 0.0, 0.0)
    return LatencySummary(
        n=len(items),
        mean_ms=sum(items) / len(items),
        p50_ms=percentile(items, 50),
        p95_ms=percentile(items, 95),
        max_ms=max(items),
    )


def stage_latencies(timings: Iterable[Mapping[str, float]]) -> dict[str, LatencySummary]:
    """Per stage (``total`` last), the summary of that stage's timings across runs. Stages
    that recorded an error marker (``graph_error``) are not timings and are skipped."""
    by_stage: dict[str, list[float]] = {}
    for run in timings:
        for stage, value in run.items():
            if stage.endswith("_error") or not isinstance(value, int | float):
                continue
            by_stage.setdefault(stage, []).append(float(value))
    order = sorted(by_stage, key=lambda s: (s == "total", s))
    return {stage: latency_summary(by_stage[stage]) for stage in order}


def render_latency_table(rows: Mapping[str, LatencySummary]) -> list[str]:
    """Markdown rows: one per named summary."""
    lines = ["| | n | mean ms | p50 ms | p95 ms | max ms |", "|---|---:|---:|---:|---:|---:|"]
    for name, s in rows.items():
        lines.append(
            f"| {name} | {s.n} | {s.mean_ms:.0f} | {s.p50_ms:.0f} | {s.p95_ms:.0f} | "
            f"{s.max_ms:.0f} |"
        )
    return lines


# --------------------------------------------------------------------------- CLI


def read_results(path: Path) -> list[dict[str, Any]]:
    """Per-question results (``eval.benchmark_retrieval --out``, ``eval.runner``)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="p50 / p95 latency of an evaluation run.")
    parser.add_argument("--results", type=Path, required=True, help="per-question JSONL")
    args = parser.parse_args(argv)
    results = read_results(args.results)
    rows = {"retrieval": latency_summary(r.get("latency_ms") for r in results)}
    rows |= {
        f"stage {k}": v
        for k, v in stage_latencies(r.get("timings_ms") or {} for r in results).items()
    }
    print("\n".join(render_latency_table(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
