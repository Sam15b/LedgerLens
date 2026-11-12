"""Tests for scripts/llm_smoke.py (loaded by path: scripts/ is not a package).

Only the parts that do not need live keys: model selection, scoring, the run loop against
a fake provider, and the report.
"""

import importlib.util
import sys
from collections.abc import Generator
from datetime import UTC, date, datetime
from pathlib import Path
from types import ModuleType

from ledgerlens.core.config import ModelsConfig
from ledgerlens.llm.router import LLMClient
from ledgerlens.llm.types import LLMProviderError, LLMRequest, LLMResponse, Usage
from ledgerlens.observability.cost import CostLedger

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "llm_smoke.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("llm_smoke", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(module)
    return module


smoke = _load()

CONFIG = ModelsConfig.model_validate(
    {
        "aliases": {"fast": ["ollama/a", "openai/b"], "smart": ["openai/b", "gemini/c"]},
        "pricing_usd_per_1m_tokens": {"ollama/a": {"in": 0, "out": 0}},
    }
)


class ScriptedProvider:
    def __init__(self, name: str, reply: str, *, fail: bool = False) -> None:
        self.name = name
        self.reply = reply
        self.fail = fail

    def complete(self, req: LLMRequest) -> LLMResponse:
        if self.fail:
            raise LLMProviderError("bad key", provider=self.name)
        return LLMResponse(
            text=self.reply,
            provider=self.name,
            model=req.model,
            usage=Usage(tokens_in=300, tokens_out=60),
            latency_ms=250.0,
        )

    def stream(self, req: LLMRequest) -> Generator[str, None, Usage]:
        yield "Payment is due "
        yield "30 days after the invoice date."
        return Usage(tokens_in=20, tokens_out=10)


def client_with(*providers: ScriptedProvider) -> LLMClient:
    return LLMClient(
        CONFIG, providers={p.name: p for p in providers}, factories={}, ledger=CostLedger()
    )


def test_models_in_aliases_lists_each_model_once_in_order() -> None:
    assert smoke.models_in_aliases(CONFIG) == ["ollama/a", "openai/b", "gemini/c"]


def test_score_counts_matching_fields() -> None:
    almost = smoke.EXPECTED.model_copy(
        update={"vendor_name": " acme supplies ltd ", "due_date": date(2026, 11, 2)}
    )

    assert smoke.score(smoke.EXPECTED) == len(smoke.SmokeInvoice.model_fields)
    assert smoke.score(almost) == len(smoke.SmokeInvoice.model_fields) - 1


def test_run_model_scores_extraction_and_times_the_stream() -> None:
    ollama = ScriptedProvider("ollama", smoke.EXPECTED.model_dump_json())

    result = smoke.run_model(client_with(ollama), "ollama/a")

    assert result.ok, result.error
    assert result.fields_correct == len(smoke.SmokeInvoice.model_fields)
    assert (result.tokens_in, result.tokens_out, result.latency_ms) == (300, 60, 250.0)
    assert result.cost_usd == 0.0
    assert result.first_chunk_ms is not None
    assert result.stream_text == "Payment is due 30 days after the invoice date."


def test_run_model_reports_failures_and_unpriced_models() -> None:
    failing = smoke.run_model(client_with(ScriptedProvider("openai", "", fail=True)), "openai/b")
    unpriced = smoke.run_model(
        client_with(ScriptedProvider("gemini", smoke.EXPECTED.model_dump_json())), "gemini/c"
    )

    assert not failing.ok
    assert failing.error is not None and "bad key" in failing.error
    assert unpriced.ok and unpriced.cost_usd is None


def test_render_markdown_has_a_row_per_model_and_lists_failures() -> None:
    good = smoke.SmokeResult(
        "ollama/a",
        fields_correct=9,
        latency_ms=250,
        tokens_in=300,
        tokens_out=60,
        cost_usd=0.0,
        first_chunk_ms=80,
        stream_text="Due in 30 days.",
    )
    bad = smoke.SmokeResult("openai/b", error="LLMProviderError: bad key")

    report = smoke.render_markdown([good, bad], when=datetime(2026, 11, 11, 9, 30, tzinfo=UTC))

    assert report.startswith("# Provider comparison — 2026-11-11")
    assert "| `ollama/a` | 9/9 | 250 | 300 / 60 | 0.000000 | 80 |" in report
    assert "| `openai/b` | failed |" in report
    assert "- `openai/b`: LLMProviderError: bad key" in report
    assert "- `ollama/a`: Due in 30 days." in report
