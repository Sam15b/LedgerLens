"""Latency percentiles and cost per query / per document."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval.latency_cost import (
    cost_report,
    latency_summary,
    load_llm_calls,
    main,
    percentile,
    render_cost_table,
    render_latency_table,
    request_kind,
    stage_latencies,
)
from ledgerlens.observability.cost import LLMCallRecord, document_request_id

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


# --------------------------------------------------------------------------- cost


def call(request_id: str | None, cost: float | None, tokens: int = 100) -> LLMCallRecord:
    return LLMCallRecord(
        request_id=request_id,
        alias="smart",
        provider="openai",
        model="gpt-4o",
        tokens_in=tokens,
        tokens_out=tokens // 10,
        cost_usd=cost,
        latency_ms=50.0,
    )


def test_request_kind_from_the_request_id() -> None:
    assert request_kind(document_request_id("d1")) == "document"
    assert request_kind("eval:run1:Q001") == "query"
    assert request_kind("0f3a9c") == "query"
    assert request_kind(None) is None and request_kind("") is None


def test_cost_per_query_and_per_document() -> None:
    records = [
        call("q1", 0.010),
        call("q1", 0.002),  # two calls, one query
        call("q2", 0.004),
        call(document_request_id("d1"), 0.001),
        call(document_request_id("d2"), None),  # unpriced (local model)
        call(None, 0.5),  # no request: in neither group
    ]

    report = cost_report(records)

    assert (report.query.n, report.query.calls) == (2, 3)
    assert report.query.total_usd == pytest.approx(0.016)
    assert report.query.mean_usd == pytest.approx(0.008)
    assert report.query.max_usd == pytest.approx(0.012)
    assert report.query.p50_usd == pytest.approx(0.008)
    assert (report.document.n, report.document.unpriced_calls) == (2, 1)
    assert report.document.mean_usd == pytest.approx(0.0005)
    assert (report.unattributed_calls, report.unattributed_usd) == (1, 0.5)
    assert report.by_model[0].calls == 6
    assert json.loads(json.dumps(report.as_dict()))["query"]["n"] == 2


def test_cost_report_of_nothing() -> None:
    report = cost_report([])
    assert report.query.n == 0 and report.query.mean_usd == 0.0 and report.by_model == []


def test_cost_table_and_cli_from_a_cost_log(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "llm_calls.jsonl"
    records = [call("q1", 0.01), call(document_request_id("d1"), None), call(None, 0.1)]
    log.write_text("".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")

    lines = render_cost_table(cost_report(load_llm_calls(str(log))))
    assert lines[2].startswith("| per query | 1 | 1 | 100 / 10 | 0.0100 |")
    assert any("1 calls had no price" in line for line in lines)
    assert any("1 calls had no request id" in line for line in lines)

    assert main(["--llm-calls", str(log), "--since", "2000-01-01T00:00:00+00:00"]) == 0
    assert "| per document | 1 |" in capsys.readouterr().out
    later = datetime(2999, 1, 1, tzinfo=UTC)
    assert load_llm_calls(str(log), since=later) == []


def test_cli_needs_something_to_report() -> None:
    with pytest.raises(SystemExit):
        main([])
