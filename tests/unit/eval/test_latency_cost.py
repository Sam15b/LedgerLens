"""Latency percentiles and cost per query / per document."""

import json
from pathlib import Path

import pytest

from eval.latency_cost import (
    latency_summary,
    main,
    percentile,
    render_latency_table,
    stage_latencies,
)

# --------------------------------------------------------------------------- latency


def test_percentile_interpolates_between_neighbours() -> None:
    values = [float(v) for v in range(1, 101)]  # 1..100

    assert percentile(values, 50) == pytest.approx(50.5)
    assert percentile(values, 95) == pytest.approx(95.05)
    assert percentile(values, 0) == 1.0 and percentile(values, 100) == 100.0
    assert percentile([7.0], 95) == 7.0
    assert percentile([], 50) == 0.0


def test_percentile_rejects_a_bad_pct() -> None:
    with pytest.raises(ValueError, match="between 0 and 100"):
        percentile([1.0], 101)


def test_latency_summary_skips_missing_timings() -> None:
    summary = latency_summary([100.0, None, 300.0, 200.0])

    assert (summary.n, summary.mean_ms, summary.p50_ms, summary.max_ms) == (3, 200.0, 200.0, 300.0)
    assert summary.p95_ms == pytest.approx(290.0)
    assert latency_summary([]).n == 0


def test_stage_latencies_per_stage_total_last() -> None:
    runs = [
        {"dense": 10.0, "rerank": 100.0, "total": 120.0},
        {"dense": 30.0, "rerank": 300.0, "graph_error": 1.0, "total": 400.0},
    ]

    stages = stage_latencies(runs)

    assert list(stages) == ["dense", "rerank", "total"]
    assert stages["dense"].p50_ms == 20.0
    assert stages["total"].max_ms == 400.0


def test_latency_table_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    lines = render_latency_table({"retrieval": latency_summary([1.0, 3.0])})
    assert lines[2] == "| retrieval | 2 | 2 | 2 | 3 | 3 |"

    results = tmp_path / "results.jsonl"
    rows = [{"latency_ms": 100.0, "timings_ms": {"dense": 40.0, "total": 100.0}}] * 3
    results.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert main(["--results", str(results)]) == 0
    out = capsys.readouterr().out
    assert "| retrieval | 3 | 100 | 100 | 100 | 100 |" in out
    assert "| stage dense | 3 |" in out
