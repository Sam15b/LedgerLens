import json
from decimal import Decimal
from pathlib import Path

import pytest

from ledgerlens.core.config import ModelsConfig, Settings, repo_path
from ledgerlens.core.logging import request_context
from ledgerlens.observability.cost import (
    CostLedger,
    JsonlCostSink,
    LLMCallRecord,
    PostgresCostSink,
    llm_call_values,
    read_jsonl,
    summarize,
)


def call(
    provider: str = "openai",
    model: str = "gpt-4o",
    *,
    cost_usd: float | None = 0.01,
    latency_ms: float = 100.0,
) -> LLMCallRecord:
    return LLMCallRecord(
        alias="smart",
        provider=provider,
        model=model,
        tokens_in=1_000,
        tokens_out=200,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
    )


class BrokenSink:
    def write(self, record: LLMCallRecord) -> None:
        raise OSError("disk full")


def test_record_picks_up_the_request_id_from_the_logging_context() -> None:
    with request_context("req-42"):
        record = call()

    assert record.request_id == "req-42"
    assert record.ts.tzinfo is not None


def test_jsonl_sink_appends_one_line_per_call(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "llm_calls.jsonl"
    ledger = CostLedger([JsonlCostSink(path)])

    ledger.record(call())
    ledger.record(call("gemini", "gemini-3.8-flash", cost_usd=None))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["provider"] for line in lines] == ["openai", "gemini"]
    assert read_jsonl(path) == ledger.records


def test_read_jsonl_of_a_missing_file_is_empty(tmp_path: Path) -> None:
    assert read_jsonl(tmp_path / "none.jsonl") == []


def test_a_failing_sink_does_not_break_the_call(caplog: pytest.LogCaptureFixture) -> None:
    ledger = CostLedger([BrokenSink()])

    ledger.record(call())

    assert len(ledger.records) == 1
    assert "BrokenSink failed" in caplog.text


def test_memory_keeps_only_the_most_recent_records() -> None:
    ledger = CostLedger(max_records=2)
    for latency in (1.0, 2.0, 3.0):
        ledger.record(call(latency_ms=latency))

    assert [r.latency_ms for r in ledger.records] == [2.0, 3.0]


def test_totals_and_per_model_summary() -> None:
    ledger = CostLedger()
    ledger.record(call(cost_usd=0.01, latency_ms=100))
    ledger.record(call(cost_usd=0.03, latency_ms=300))
    ledger.record(call("ollama", "llama3:8b", cost_usd=0.0))
    ledger.record(call("mistral", "mistral-small-2603", cost_usd=None))

    assert ledger.total_cost_usd() == pytest.approx(0.04)
    by_model = {(s.provider, s.model): s for s in ledger.summary()}
    openai = by_model[("openai", "gpt-4o")]
    assert (openai.calls, openai.tokens_in, openai.tokens_out) == (2, 2_000, 400)
    assert openai.cost_usd == pytest.approx(0.04)
    assert openai.mean_latency_ms == 200
    assert by_model[("ollama", "llama3:8b")].cost_usd == 0.0
    mistral = by_model[("mistral", "mistral-small-2603")]
    assert (mistral.cost_usd, mistral.unpriced_calls) == (None, 1)


def test_summarize_keeps_first_appearance_order() -> None:
    records = [call("b", "m"), call("a", "m"), call("b", "m")]

    assert [s.provider for s in summarize(records)] == ["b", "a"]


NO_DB = Settings(llm_cost_db=False)


def test_from_config_writes_to_cost_log_relative_to_repo() -> None:
    ledger = CostLedger.from_config(ModelsConfig(cost_log="data/interim/x.jsonl"), NO_DB)

    sink = ledger.sinks[0]
    assert isinstance(sink, JsonlCostSink)
    assert sink.path == repo_path("data/interim/x.jsonl")
    assert sink.path.is_absolute()


def test_from_config_without_cost_log_keeps_memory_only() -> None:
    assert CostLedger.from_config(ModelsConfig(), NO_DB).sinks == []


def test_from_config_adds_the_llm_calls_sink_when_enabled() -> None:
    ledger = CostLedger.from_config(ModelsConfig(), Settings(llm_cost_db=True))

    assert [type(sink) for sink in ledger.sinks] == [PostgresCostSink]  # nothing connects yet


def test_llm_call_values_store_cost_as_the_shortest_decimal() -> None:
    values = llm_call_values(call(cost_usd=0.1))

    assert values["cost_usd"] == Decimal("0.1")
    assert llm_call_values(call(cost_usd=None))["cost_usd"] is None
    assert set(values) == set(LLMCallRecord.model_fields)
