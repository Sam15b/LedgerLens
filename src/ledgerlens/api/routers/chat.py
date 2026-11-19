"""``POST /chat``: run the agent graph on a message and stream what happens as SSE.

    curl -N -X POST localhost:8000/chat -H 'Content-Type: application/json' \\
         -d '{"message": "Show unpaid invoices from Acme over $5k"}'

The response is ``text/event-stream``; each event is ``event: <type>`` and one JSON
``data:`` line:

* ``token`` -- ``{"text", "node"}``: the answer text, in word-sized pieces, as each agent
  finishes;
* ``done`` -- ``{"thread_id", "status", "intent", "errors"}``: last event of every
  stream; ``status`` is ``completed``, or ``interrupted`` when the run waits for a person.

A conversation is a ``thread_id`` (one is made if the request has none; send it back to
continue). The graph's checkpointer keeps the thread, so each turn starts from the last.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Overwrite
from pydantic import BaseModel, Field

from ledgerlens.api.agent import AgentRuntime, AgentUnavailableError

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


async def events(runtime: AgentRuntime, request: ChatRequest, thread_id: str) -> AsyncIterator[str]:
    config = {"configurable": {"thread_id": thread_id}}
    try:
        async for update in runtime.graph.astream(
            turn_input(request), config, stream_mode="updates"
        ):
            for node, values in update.items():
                for message in (
                    (values or {}).get("messages", []) if isinstance(values, dict) else []
                ):
                    if isinstance(message, AIMessage) and message.text:
                        for piece in pieces(message.text):
                            yield sse("token", {"text": piece, "node": node})
    except Exception as exc:  # the graph guards its nodes; this is the graph itself failing
        logger.exception("chat run failed")
        yield sse("error", {"message": f"{type(exc).__name__}: {exc}"})
    state = await runtime.graph.aget_state(config)
    values = state.values or {}
    yield sse(
        "done",
        {
            "thread_id": thread_id,
            "status": "interrupted" if state.next else "completed",
            "intent": values.get("intent"),
            "errors": values.get("errors", []),
        },
    )


@router.post("/chat", response_class=StreamingResponse)
async def chat(
    body: ChatRequest, runtime: Annotated[AgentRuntime, Depends(get_agent)]
) -> StreamingResponse:
    """Run the agents on a message; stream ``token`` ... ``done`` events (SSE)."""
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
