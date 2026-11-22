"""Tracing: spans nest into one tree per request, carry cost / token / retriever
attributes, sample whole traces -- and a run behaves the same with tracing off or broken."""

import asyncio
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from ledgerlens.agents.graph import AgentNodes, build_graph
from ledgerlens.agents.nodes.router import Router
from ledgerlens.agents.state import LedgerState
from ledgerlens.agents.tools import ToolOutcome, tool_attributes
from ledgerlens.api.agent import AgentProvider, AgentRuntime
from ledgerlens.api.main import create_app
from ledgerlens.core.config import ModelsConfig, Settings
from ledgerlens.llm.router import LLMClient
from ledgerlens.llm.types import LLMRequest, LLMResponse, Message, Usage
from ledgerlens.observability import tracing
from ledgerlens.observability.tracing import (
    NOOP_SPAN,
    JsonlBackend,
    LangSmithBackend,
    MemoryBackend,
    SpanRecord,
    Tracer,
    build_tracer,
    compact,
    current_span,
    format_tree,
    langsmith_usage,
    last_trace,
    read_spans,
    set_tracer,
    span,
)

MODELS = ModelsConfig.model_validate(
    {
        "aliases": {"smart": ["openai/big"]},
        "pricing_usd_per_1m_tokens": {"openai/big": {"in": 2.0, "out": 10.0}},
    }
)


class FakeProvider:
    name = "openai"

    def complete(self, req: LLMRequest) -> LLMResponse:
        return LLMResponse(
            text="answer",
            provider="openai",
            model=req.model,
            usage=Usage(tokens_in=1_000, tokens_out=100),
            latency_ms=5.0,
        )

    def stream(self, req: LLMRequest) -> Any:
        yield "ans"
        yield "wer"
        return Usage(tokens_in=10, tokens_out=2)


def llm_client() -> LLMClient:
    return LLMClient(MODELS, providers={"openai": FakeProvider()}, factories={})


class BrokenBackend:
    """Every call fails, as a LangSmith outage would."""

    calls = 0

    def start(self, record: SpanRecord, parent: Any) -> Any:
        self.calls += 1
        raise ConnectionError("langsmith is down")

    def end(self, record: SpanRecord, handle: Any) -> None:
        self.calls += 1
        raise ConnectionError("langsmith is down")

    def flush(self) -> None:
        raise ConnectionError("langsmith is down")


@pytest.fixture
def memory() -> Iterator[MemoryBackend]:
    backend = MemoryBackend()
    set_tracer(Tracer(backend))
    yield backend
    set_tracer(None)


@pytest.fixture
def tracing_off() -> Iterator[None]:
    set_tracer(Tracer())
    yield
    set_tracer(None)


def tree(backend: MemoryBackend) -> dict[str, str | None]:
    """span name -> parent span name."""
    names = {r.span_id: r.name for r in backend.records}
    return {r.name: names.get(r.parent_id) if r.parent_id else None for r in backend.records}


# --------------------------------------------------------------------------- spans


def test_spans_nest_across_await_and_threads(memory: MemoryBackend) -> None:
    def llm_call() -> None:
        with span("llm", kind="llm"):
            pass

    async def tool() -> None:
        with span("tool", kind="tool"):
            await asyncio.to_thread(llm_call)

    async def run() -> None:
        with span("chat"), span("qa"):
            await tool()

    asyncio.run(run())

    assert tree(memory) == {"chat": None, "qa": "chat", "tool": "qa", "llm": "tool"}
    assert len({r.trace_id for r in memory.records}) == 1
    assert current_span() is None


def test_an_error_is_recorded_and_raised(memory: MemoryBackend) -> None:
    with pytest.raises(ValueError, match="boom"), span("step"):
        raise ValueError("boom")

    (record,) = memory.records
    assert record.error == "ValueError: boom"
    assert record.end_time is not None


def test_expected_exceptions_are_not_errors(memory: MemoryBackend) -> None:
    with pytest.raises(KeyError), span("step", expected=(KeyError,)):
        raise KeyError("control flow")

    assert memory.records[0].error is None


def test_inputs_and_outputs_are_bounded() -> None:
    value = compact({"text": "x" * 10_000, "items": list(range(500)), "model": Usage()})

    assert len(value["text"]) < 4_100 and value["text"].endswith("[truncated]")
    assert len(value["items"]) == tracing.MAX_ITEMS + 1
    assert value["model"] == {"tokens_in": 0, "tokens_out": 0}


# --------------------------------------------------------------------------- tracing off


def test_tracing_off_runs_the_code_and_records_nothing(tracing_off: None) -> None:
    with span("chat") as opened:
        opened.set("x", 1)
        opened.add_usage(tokens_in=1, tokens_out=1, cost_usd=0.1)
        inner = current_span()
    assert opened is NOOP_SPAN and inner is NOOP_SPAN
    with pytest.raises(RuntimeError), span("chat"):
        raise RuntimeError("still raised")


def test_a_broken_backend_never_breaks_a_run(caplog: pytest.LogCaptureFixture) -> None:
    backend = BrokenBackend()
    tracer = Tracer(backend)
    set_tracer(tracer)
    try:
        with span("chat") as opened:
            response = llm_client().complete([Message.user("hi")])
            opened.set("ok", True)
        tracer.flush()
    finally:
        set_tracer(None)

    assert response.text == "answer"
    assert backend.calls == 4  # two spans, each start and end attempted
    warnings = [r for r in caplog.records if "tracing backend" in r.getMessage()]
    assert len(warnings) == 1  # logged once, not once per span


def test_build_tracer_is_off_unless_enabled(tmp_path: Path) -> None:
    assert not build_tracer(Settings(tracing_enabled=False, tracing_backend="jsonl")).enabled
    jsonl = build_tracer(
        Settings(tracing_enabled=True, tracing_backend="jsonl", tracing_path=tmp_path / "t.jsonl")
    )
    assert isinstance(jsonl.backend, JsonlBackend)


def test_langsmith_without_a_key_leaves_tracing_off() -> None:
    settings = Settings(tracing_enabled=True, tracing_backend="langsmith", langsmith_api_key="")
    assert not build_tracer(settings).enabled


def test_a_backend_that_cannot_start_leaves_tracing_off(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(*args: Any, **kwargs: Any) -> None:
        raise ImportError("No module named 'langsmith'")

    monkeypatch.setattr(tracing, "LangSmithBackend", broken)
    settings = Settings(tracing_enabled=True, langsmith_api_key="lsv2-test")

    assert not build_tracer(settings).enabled
    assert "failed to start" in caplog.text


# --------------------------------------------------------------------------- sampling


def test_sampling_keeps_or_drops_whole_traces() -> None:
    draws = iter([0.9, 0.1])  # first trace dropped (0.9 >= 0.5), second kept
    backend = MemoryBackend()
    set_tracer(Tracer(backend, sample_rate=0.5, rng=lambda: next(draws)))
    try:
        for name in ("dropped", "kept"):
            with span(name), span(f"{name}.child"), span(f"{name}.grandchild"):
                pass
    finally:
        set_tracer(None)

    assert sorted(r.name for r in backend.records) == ["kept", "kept.child", "kept.grandchild"]


def test_sample_rate_zero_records_nothing() -> None:
    backend = MemoryBackend()
    set_tracer(Tracer(backend, sample_rate=0.0))
    try:
        with span("chat"), span("router"):
            pass
    finally:
        set_tracer(None)
    assert backend.records == []


def test_sample_rate_must_be_a_share() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        Tracer(MemoryBackend(), sample_rate=1.5)


# --------------------------------------------------------------------------- attributes


def test_llm_spans_carry_cost_and_tokens_and_add_them_up(memory: MemoryBackend) -> None:
    client = llm_client()
    with span("chat"):
        with span("qa"):
            client.complete([Message.user("one")])
        client.complete([Message.user("two")])

    llm = memory.by_name("llm smart")[0].attributes
    assert llm["llm.provider"] == "openai" and llm["llm.model"] == "big"
    assert (llm["llm.tokens_in"], llm["llm.tokens_out"]) == (1_000, 100)
    assert llm["llm.cost_usd"] == pytest.approx(0.003)
    qa = memory.by_name("qa")[0].attributes
    assert qa["usage.llm_calls"] == 1 and qa["usage.cost_usd"] == pytest.approx(0.003)
    chat = memory.by_name("chat")[0].attributes
    assert chat["usage.llm_calls"] == 2
    assert (chat["usage.tokens_in"], chat["usage.tokens_out"]) == (2_000, 200)
    assert chat["usage.cost_usd"] == pytest.approx(0.006)


def test_a_streamed_call_is_one_span(memory: MemoryBackend) -> None:
    with span("chat"):
        chunks = list(llm_client().stream([Message.user("hi")]))
        assert current_span() is not None and current_span().record.name == "chat"  # type: ignore[union-attr]

    assert chunks == ["ans", "wer"]
    (llm,) = memory.by_name("llm smart")
    assert llm.attributes["llm.streamed"] is True
    assert llm.attributes["llm.tokens_in"] == 10
    assert llm.outputs["text"] == "answer"
    assert tree(memory)["llm smart"] == "chat"


def test_search_tool_spans_get_retriever_variant_and_top_k_ids() -> None:
    outcome = ToolOutcome(
        tool="search_documents",
        args={"query": "unpaid", "k": 2},
        call_id="c1",
        ok=True,
        data={
            "retriever": "r4_graphrag",
            "hits": [{"chunk_id": "d1#c0", "document_id": "d1"}, {"document_id": "d2"}],
        },
    )

    assert tool_attributes(outcome) == {
        "tool.ok": True,
        "retriever.variant": "r4_graphrag",
        "retriever.k": 2,
        "retriever.top_k_ids": ["d1#c0", "d2"],
    }
    other = ToolOutcome(tool="run_analytics", call_id="c2", ok=False, text="bad")
    assert tool_attributes(other) == {"tool.ok": False}


def test_langsmith_runs_get_usage_and_model_metadata() -> None:
    posted: list[Any] = []

    class FakeClient:
        def flush(self) -> None:
            pass

    backend = LangSmithBackend("ledgerlens", client=FakeClient())
    set_tracer(Tracer(backend))
    try:
        from langsmith.run_trees import RunTree

        original = RunTree.post
        RunTree.post = lambda self, *a, **k: posted.append(self)  # type: ignore[method-assign]
        try:
            with span("chat"):
                llm_client().complete([Message.user("hi")])
        finally:
            RunTree.post = original  # type: ignore[method-assign]
    finally:
        set_tracer(None)

    llm, chat = posted
    assert llm.parent_run_id == chat.id and llm.trace_id == chat.id
    assert llm.run_type == "llm"
    assert llm.outputs["usage_metadata"] == {
        "input_tokens": 1_000,
        "output_tokens": 100,
        "total_tokens": 1_100,
        "total_cost": pytest.approx(0.003),
    }
    assert llm.extra["metadata"]["ls_model_name"] == "big"
    assert chat.extra["metadata"]["usage.llm_calls"] == 1


def test_langsmith_usage_without_a_price() -> None:
    usage = langsmith_usage({"llm.tokens_in": 3, "llm.tokens_out": 4, "llm.cost_usd": None})
    assert usage == {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}


# --------------------------------------------------------------------------- agent runs


def qa(state: LedgerState) -> dict[str, Any]:
    return {"messages": [AIMessage("INV-889 is unpaid.", name="qa")], "step_count": 1}


def failing_qa(state: LedgerState) -> dict[str, Any]:
    raise RuntimeError("index offline")


def run_graph(node: Any) -> dict[str, Any]:
    graph = build_graph(AgentNodes(router=Router(None), qa=node), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t1"}}
    with span("chat"):
        result: dict[str, Any] = graph.invoke(
            {"messages": [HumanMessage("Which Acme invoices are unpaid?")]}, config
        )
    return result


def test_agent_runs_are_the_same_with_tracing_on_or_off(memory: MemoryBackend) -> None:
    traced = run_graph(qa)
    set_tracer(Tracer())
    untraced = run_graph(qa)

    assert [m.text for m in traced["messages"]] == [m.text for m in untraced["messages"]]
    assert tree(memory) == {"chat": None, "router": "chat", "qa": "chat", "respond": "chat"}
    assert memory.by_name("router")[0].attributes["agent.intent"] == "question"
    assert memory.by_name("qa")[0].outputs == {"reply": "INV-889 is unpaid."}


def test_a_failing_node_is_an_error_span_and_the_run_goes_on(memory: MemoryBackend) -> None:
    result = run_graph(failing_qa)

    assert "index offline" in result["messages"][-1].text
    assert memory.by_name("qa")[0].error == "RuntimeError: index offline"
    assert memory.by_name("chat")[0].error is None


def chat_client() -> TestClient:
    graph = build_graph(AgentNodes(router=Router(None), qa=qa), checkpointer=InMemorySaver())

    async def factory(stack: Any) -> AgentRuntime:
        return AgentRuntime(graph)

    return TestClient(create_app(AgentProvider(factory)), raise_server_exceptions=False)


def post_chat(client: TestClient) -> str:
    with client.stream("POST", "/chat", json={"message": "Which invoices are unpaid?"}) as r:
        assert r.status_code == 200
        return r.read().decode()


def test_a_chat_request_is_one_trace(memory: MemoryBackend) -> None:
    with chat_client() as client:
        post_chat(client)

    assert tree(memory) == {"chat": None, "router": "chat", "qa": "chat", "respond": "chat"}
    (root,) = memory.by_name("chat")
    assert root.inputs["message"] == "Which invoices are unpaid?"
    assert root.attributes["chat.status"] == "completed"
    assert root.attributes["request_id"] and root.attributes["thread_id"].startswith("chat-")


def test_chat_streams_the_same_with_tracing_off_or_broken(tracing_off: None) -> None:
    with chat_client() as client:
        off = post_chat(client)
    set_tracer(Tracer(BrokenBackend()))
    with chat_client() as client:
        broken = post_chat(client)

    def without_ids(text: str) -> list[str]:
        return [line for line in text.splitlines() if "thread_id" not in line]

    assert '"text": "unpaid.", "node": "qa"' in off and '"status": "completed"' in off
    assert without_ids(off) == without_ids(broken)


# --------------------------------------------------------------------------- viewing


def test_jsonl_spans_print_as_a_tree(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "traces.jsonl"
    set_tracer(Tracer(JsonlBackend(path)))
    try:
        with span("old"):
            pass
        with span("chat"), span("qa") as qa_span:
            qa_span.set("agent.node", "qa")
            with span("llm smart", kind="llm"):
                pass
    finally:
        set_tracer(None)

    spans = last_trace(read_spans(path))
    assert [s.name for s in spans] == ["llm smart", "qa", "chat"]
    text = format_tree(spans)
    assert text.splitlines()[0].startswith("chat [chain]")
    assert text.splitlines()[1].startswith("  qa [chain]") and "agent.node=qa" in text
    assert text.splitlines()[2].startswith("    llm smart [llm]")

    assert tracing.main(["show", "--path", str(path)]) == 0
    assert "llm smart" in capsys.readouterr().out
    assert tracing.main(["show", "--path", str(tmp_path / "none.jsonl")]) == 1


def test_span_records_round_trip() -> None:
    record = SpanRecord(
        name="x",
        kind="tool",
        trace_id="t",
        span_id="s",
        parent_id=None,
        start_time=datetime(2026, 11, 22, tzinfo=UTC),
        end_time=datetime(2026, 11, 22, 0, 0, 1, tzinfo=UTC),
        attributes={"a": 1},
    )
    data = json.loads(json.dumps(record.to_dict()))
    assert data["duration_ms"] == 1000.0
    assert SpanRecord.from_dict(data) == record
