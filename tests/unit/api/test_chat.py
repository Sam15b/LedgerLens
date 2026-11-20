"""POST /chat: a well-formed SSE stream with the events in order; request ids and error
bodies. The graph runs over stand-in agents (no stores, no models)."""

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from ledgerlens.agents.graph import AgentNodes, build_graph
from ledgerlens.agents.nodes.router import Router
from ledgerlens.agents.state import LedgerState
from ledgerlens.api.agent import AgentProvider, AgentRuntime
from ledgerlens.api.main import create_app
from ledgerlens.api.routers.chat import pieces

EVENTS = {"token", "tool_call", "tool_result", "citation", "interrupt", "error", "done"}


def parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    """Every event as (type, data), checking the framing: an ``event:`` line, one JSON
    ``data:`` line, and a blank line after each."""
    assert text.endswith("\n\n")
    events = []
    for block in text.split("\n\n")[:-1]:
        lines = block.split("\n")
        assert len(lines) == 2, block
        assert lines[0].startswith("event: ") and lines[1].startswith("data: "), block
        name = lines[0].removeprefix("event: ")
        assert name in EVENTS, name
        events.append((name, json.loads(lines[1].removeprefix("data: "))))
    return events


def qa(state: LedgerState) -> dict[str, Any]:
    call = {
        "name": "search_documents",
        "args": {"query": "unpaid"},
        "id": "c1",
        "type": "tool_call",
    }
    return {
        "messages": [
            AIMessage("", tool_calls=[call], name="qa"),
            ToolMessage('{"hits": 1}', tool_call_id="c1", name="search_documents"),
            AIMessage("INV-889 is unpaid [doc:INV-889].", name="qa"),
        ],
        "citations": {"doc:INV-889": "d1"},
        "step_count": 2,
    }


def writer(state: LedgerState) -> dict[str, Any]:
    draft = {"reminder_id": "r1", "invoice_number": "INV-889", "tone": "firm", "body": "..."}
    return {
        "messages": [AIMessage("I drafted 1 reminder.", name="reminder_writer")],
        "reminder_drafts": [draft],
        "step_count": 1,
    }


def app_with(**nodes: Any) -> TestClient:
    graph = build_graph(AgentNodes(router=Router(None), **nodes), checkpointer=InMemorySaver())

    async def factory(stack: Any) -> AgentRuntime:
        return AgentRuntime(graph)

    return TestClient(create_app(AgentProvider(factory)), raise_server_exceptions=False)


def chat(
    client: TestClient, message: str, **body: Any
) -> tuple[Any, list[tuple[str, dict[str, Any]]]]:
    with client.stream("POST", "/chat", json={"message": message, **body}) as response:
        text = response.read().decode()
    return response, parse_sse(text) if response.status_code == 200 else []


def test_question_streams_tool_events_tokens_citations_then_done() -> None:
    response, events = chat(app_with(qa=qa), "Which invoices are unpaid?")
    assert response.headers["content-type"].startswith("text/event-stream")
    assert [name for name, _ in events] == [
        "tool_call",
        "tool_result",
        *["token"] * len(pieces("INV-889 is unpaid [doc:INV-889].")),
        "citation",
        "done",
    ]
    data = dict(events)
    assert data["tool_call"] == {
        "id": "c1",
        "name": "search_documents",
        "args": {"query": "unpaid"},
        "node": "qa",
    }
    assert (
        data["tool_result"]["status"] == "success"
        and data["tool_result"]["name"] == "search_documents"
    )
    text = "".join(d["text"] for name, d in events if name == "token")
    assert text == "INV-889 is unpaid [doc:INV-889]."
    assert data["citation"] == {"tag": "doc:INV-889", "document_id": "d1"}
    done = events[-1][1]
    assert done["status"] == "completed" and done["intent"] == "question" and done["errors"] == []
    assert done["thread_id"] == response.headers["x-thread-id"]
    assert response.headers["x-request-id"]


def test_thread_continues_and_steps_reset_each_turn() -> None:
    client = app_with(qa=qa)
    first, _ = chat(client, "Which invoices are unpaid?")
    thread = first.headers["x-thread-id"]
    for _ in range(4):  # 4 turns x 4 steps would pass the limit of 8 if steps carried over
        _, events = chat(client, "And now?", thread_id=thread)
        assert events[-1][1]["status"] == "completed"
        assert not any(name == "error" for name, _ in events)


def test_reminder_drafts_stream_an_interrupt_and_hold_the_thread() -> None:
    client = app_with(reminder_writer=writer)
    response, events = chat(client, "Chase INV-889")
    names = [name for name, _ in events]
    assert names[-2:] == ["interrupt", "done"]
    pause = dict(events)["interrupt"]
    assert pause["type"] == "reminder_approval" and pause["reminders"][0]["reminder_id"] == "r1"
    assert events[-1][1]["status"] == "interrupted"
    thread = response.headers["x-thread-id"]
    again = client.post("/chat", json={"message": "hello", "thread_id": thread})
    assert again.status_code == 409
    assert again.json()["error"]["code"] == "conflict"


def test_a_failing_graph_streams_an_error_then_done() -> None:
    class Exploding:
        async def astream(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("checkpointer down")
            yield  # pragma: no cover

        async def aget_state(self, config: Any) -> Any:
            class State:
                next: tuple[str, ...] = ()
                values: dict[str, Any] = {}

            return State()

    async def factory(stack: Any) -> AgentRuntime:
        return AgentRuntime(Exploding())

    client = TestClient(create_app(AgentProvider(factory)))
    _, events = chat(client, "hi")
    assert [name for name, _ in events] == ["error", "done"]
    assert events[0][1] == {"message": "RuntimeError: checkpointer down"}


def test_errors_share_one_shape_with_the_request_id() -> None:
    async def down(stack: Any) -> AgentRuntime:
        raise ConnectionError("MCP server not running")

    client = TestClient(create_app(AgentProvider(down)), raise_server_exceptions=False)
    unavailable = client.post("/chat", json={"message": "hi"}, headers={"X-Request-ID": "req-42"})
    assert unavailable.status_code == 503
    assert unavailable.headers["x-request-id"] == "req-42"
    assert unavailable.json() == {
        "error": {
            "code": "unavailable",
            "message": "agents unavailable: ConnectionError: MCP server not running",
            "request_id": "req-42",
        }
    }
    invalid = client.post("/chat", json={"message": ""})
    assert invalid.status_code == 422
    body = invalid.json()["error"]
    assert body["code"] == "invalid_request" and body["message"].startswith("body.message")
    assert body["request_id"] == invalid.headers["x-request-id"]
    missing = client.get("/reminders/not-an-id")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "not_found"


def test_bad_request_ids_are_replaced() -> None:
    client = app_with(qa=qa)
    response = client.get("/health", headers={"X-Request-ID": "has spaces and; semicolons"})
    assert len(response.headers["x-request-id"]) == 32


def test_unexpected_errors_are_500_without_internals() -> None:
    app = create_app(AgentProvider())

    @app.get("/boom")
    def boom() -> None:
        raise ValueError("secret table name")

    response = TestClient(app, raise_server_exceptions=False).get("/boom")
    assert response.status_code == 500
    error = response.json()["error"]
    assert error["code"] == "internal_error" and "secret" not in error["message"]


@pytest.mark.parametrize(
    ("text", "expected"), [("a b", ["a ", "b"]), ("", []), ("x\n\ny", ["x\n\n", "y"])]
)
def test_pieces_keep_every_character(text: str, expected: list[str]) -> None:
    assert pieces(text) == expected
