"""``POST /chat``: run the agent graph on a message and stream what happens as SSE.

    curl -N -X POST localhost:8000/chat -H 'Content-Type: application/json' \\
         -d '{"message": "Show unpaid invoices from Acme over $5k"}'

The response is ``text/event-stream``; each event is ``event: <type>`` and one JSON
``data:`` line:

* ``tool_call`` -- ``{"id", "name", "args", "node"}``: an agent called an MCP tool;
* ``tool_result`` -- ``{"id", "name", "status", "content", "truncated", "node"}``: what it
  returned (``status`` ``error`` for a failed call; the agent carries on);
* ``token`` -- ``{"text", "node"}``: the answer text, in word-sized pieces, as each agent
  finishes;
* ``citation`` -- ``{"tag", "document_id"}``: a document the answer can cite, once each;
* ``interrupt`` -- ``{"thread_id", "type": "reminder_approval", "reminders": [...]}``: the
  run waits for a person to approve or reject reminder drafts
  (``POST /reminders/{id}/approve|reject``);
* ``error`` -- ``{"message"}``: the run itself failed (agents' own errors are replies);
* ``done`` -- ``{"thread_id", "status", "intent", "errors"}``: last event of every
  stream; ``status`` is ``completed``, or ``interrupted`` when the run waits for a person.

A conversation is a ``thread_id`` (one is made if the request has none; send it back to
continue). The graph's checkpointer keeps the thread, so each turn starts from the last.

Each request is one trace (``observability/tracing.py``): a ``chat`` span with the
router, the agents, their tool calls and LLM calls nested under it.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Overwrite
from pydantic import BaseModel, Field

from ledgerlens.api.agent import AgentRuntime, AgentUnavailableError
from ledgerlens.core.logging import get_request_id
from ledgerlens.observability.tracing import span

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

_PIECES = re.compile(r"\S+\s*|\s+")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4_000)
    thread_id: str | None = Field(default=None, max_length=100)
    document_ids: list[str] = Field(default_factory=list, max_length=10)
    """Documents uploaded with the message (already ingested)."""


async def get_agent(request: Request) -> AgentRuntime:
    try:
        runtime: AgentRuntime = await request.app.state.agents.get()
    except AgentUnavailableError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, f"agents unavailable: {exc}"
        ) from exc
    return runtime


def sse(event: str, data: dict[str, Any]) -> str:
    """One server-sent event."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def pieces(text: str) -> list[str]:
    """``text`` in word-sized pieces (whitespace kept), for ``token`` events."""
    return _PIECES.findall(text)


def turn_input(request: ChatRequest) -> dict[str, Any]:
    """A new turn: the message, and the per-turn state cleared (the step budget, errors and
    what the last turn found); the conversation itself carries on."""
    return {
        "messages": [HumanMessage(request.message)],
        "document_ids": request.document_ids,
        "intent": None,
        "entities": {},
        "extracted": None,
        "reconciliation": None,
        "step_count": Overwrite(0),
        "errors": Overwrite([]),
    }


RESULT_CHARS = 2_000
"""``tool_result`` events carry at most this much of the result text."""


def node_events(node: str, values: Any, cited: set[str]) -> list[str]:
    """The events for one node's state update, in order: its tool calls, their results,
    its answer text, and citations not sent before."""
    if not isinstance(values, dict):
        return []
    out: list[str] = []
    for message in values.get("messages", []):
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                out.append(
                    sse(
                        "tool_call",
                        {
                            "id": call["id"],
                            "name": call["name"],
                            "args": call["args"],
                            "node": node,
                        },
                    )
                )
            if message.text:
                out += [
                    sse("token", {"text": piece, "node": node}) for piece in pieces(message.text)
                ]
        elif isinstance(message, ToolMessage):
            text = message.text
            out.append(
                sse(
                    "tool_result",
                    {
                        "id": message.tool_call_id,
                        "name": message.name,
                        "status": message.status,
                        "content": text[:RESULT_CHARS],
                        "truncated": len(text) > RESULT_CHARS,
                        "node": node,
                    },
                )
            )
    for tag, document_id in (values.get("citations") or {}).items():
        if tag not in cited:
            cited.add(tag)
            out.append(sse("citation", {"tag": tag, "document_id": document_id}))
    return out


async def events(runtime: AgentRuntime, request: ChatRequest, thread_id: str) -> AsyncIterator[str]:
    attributes = {"thread_id": thread_id, "request_id": get_request_id()}
    inputs = {"message": request.message, "document_ids": request.document_ids}
    with span("chat", inputs=inputs, attributes=attributes) as traced:
        async for event in run_events(runtime, request, thread_id):
            yield event
        state = await runtime.graph.aget_state({"configurable": {"thread_id": thread_id}})
        values = state.values or {}
        done = {
            "thread_id": thread_id,
            "status": "interrupted" if state.next else "completed",
            "intent": values.get("intent"),
            "errors": values.get("errors", []),
        }
        traced.set_attributes({"agent.intent": done["intent"], "chat.status": done["status"]})
        traced.set_outputs(**done)
        yield sse("done", done)


async def run_events(
    runtime: AgentRuntime, request: ChatRequest, thread_id: str
) -> AsyncIterator[str]:
    """The run's events, up to (not including) ``done``."""
    config = {"configurable": {"thread_id": thread_id}}
    cited: set[str] = set()
    try:
        async for update in runtime.graph.astream(
            turn_input(request), config, stream_mode="updates"
        ):
            for node, values in update.items():
                if node == "__interrupt__":
                    for pause in values:
                        yield sse("interrupt", {"thread_id": thread_id, **pause.value})
                    continue
                for event in node_events(node, values, cited):
                    yield event
    except Exception as exc:  # the graph guards its nodes; this is the graph itself failing
        logger.exception("chat run failed")
        yield sse("error", {"message": f"{type(exc).__name__}: {exc}"})


@router.post("/chat", response_class=StreamingResponse)
async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
    """Run the agents on a message; stream ``tool_call`` / ``tool_result`` / ``token`` /
    ``citation`` / ``interrupt`` events and a final ``done`` (SSE)."""
    runtime = await get_agent(request)  # after the body is validated: a bad request is 422
    thread_id = body.thread_id or f"chat-{uuid.uuid4().hex[:16]}"
    current = await runtime.graph.aget_state({"configurable": {"thread_id": thread_id}})
    if current.next:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"thread {thread_id} is waiting for a decision on its reminder drafts: approve or "
            "reject them first (POST /reminders/{id}/approve|reject)",
        )
    return StreamingResponse(
        events(runtime, body, thread_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Thread-Id": thread_id},
    )
