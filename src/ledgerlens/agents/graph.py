"""The agent graph: a router hands each request to one specialist (LangGraph).

    graph = build_graph(AgentNodes(router=..., qa=...))
    result = graph.invoke({"messages": [HumanMessage("Which Acme invoices are unpaid?")]})

The router classifies the latest user message into an :data:`~ledgerlens.agents.state
.Intent`; :func:`route` sends ``question`` to QA, ``extract`` to the extractor,
``reconcile`` to the reconciler, ``remind`` to the reminder writer and ``other`` straight
to ``respond``. Every specialist ends in ``respond``, except that a newly extracted
invoice goes on to the reconciler first (docs/layers/04-agents-mcp.md).

Nodes are plain functions of the state, built with their dependencies (LLM, tools) and
passed in as :class:`AgentNodes`, so tests can run the graph with stand-ins. A node that
is not built yet answers that it is not available.

Guardrails:

* **step limit**: every node adds to ``step_count`` (a tool call is a step too); once it
  reaches ``max_steps`` (``configs/agents.yaml``, 8) the next hand-off goes to
  ``stopped``, which says the run was stopped, instead of another agent;
* **errors are messages**: :func:`guarded` wraps every node, so an exception inside one
  becomes an error reply and an ``errors`` entry instead of failing the request. Tool
  errors already come back as ``ToolMessage`` s (:class:`~ledgerlens.agents.tools.ToolBox`).

Human in the loop: after the reminder writer drafts reminders, :func:`human_approval`
pauses the run with LangGraph's ``interrupt()`` (the drafts are ``pending_approval``). A
person approves or rejects them -- the API records the decision and resumes the run with
``Command(resume={reminder_id: "approved" | "rejected"})`` -- and the run says what was
decided, waiting again while any draft is undecided. Nothing is sent either way. Pausing
needs a checkpointer (``build_graph(..., checkpointer=...)``): :func:`postgres_checkpointer`
keeps every run's state in Postgres, so a run waiting for approval survives a restart of
the API.
"""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ledgerlens.agents.state import Intent, LedgerState
from ledgerlens.core.config import get_config, get_settings
from ledgerlens.observability.tracing import Span, span

logger = logging.getLogger(__name__)


class Node(Protocol):
    """A graph node: the state in, the keys it changes out."""

    def __call__(self, state: LedgerState) -> dict[str, Any]: ...


class AsyncNode(Protocol):
    """A node that awaits (tools, models): run with ``graph.ainvoke`` / ``astream``."""

    async def __call__(self, state: LedgerState) -> dict[str, Any]: ...


AnyNode = Node | AsyncNode


SPECIALISTS: dict[Intent, str] = {
    "question": "qa",
    "extract": "extractor",
    "reconcile": "reconciler",
    "remind": "reminder_writer",
}
OUT_OF_SCOPE = (
    "I can help with the accounts-payable ledger: questions about invoices, purchase "
    "orders, contracts and payments; reading a new document; matching an invoice to its PO "
    "and contract; and drafting payment reminders for approval. What would you like to do?"
)


def unavailable(name: str) -> Node:
    """A placeholder for a specialist that is not built yet."""

    def node(state: LedgerState) -> dict[str, Any]:
        text = f"The {name.replace('_', ' ')} agent is not available yet."
        return {"messages": [AIMessage(text, name=name)], "step_count": 1}

    return node


def respond(state: LedgerState) -> dict[str, Any]:
    """The last step: answer out-of-scope requests; anything else is already answered."""
    if state.get("intent") in (None, "other"):
        return {"messages": [AIMessage(OUT_OF_SCOPE, name="respond")], "step_count": 1}
    return {"step_count": 1}


def make_stopped(max_steps: int) -> Node:
    def stopped(state: LedgerState) -> dict[str, Any]:
        """Where a run goes when the step limit stops it before the next node."""
        steps = state.get("step_count", 0)
        text = (
            f"I stopped after {steps} steps (the limit is {max_steps}) before finishing. "
            "Try a narrower request."
        )
        return {
            "messages": [AIMessage(text, name="stopped")],
            "errors": [f"step limit: {steps} >= {max_steps}"],
        }

    return stopped


def make_route(max_steps: int) -> Any:
    def route(state: LedgerState) -> str:
        """The node for the router's intent (``respond`` for ``other``; ``stopped`` once
        the step limit is reached)."""
        if state.get("step_count", 0) >= max_steps:
            return "stopped"
        intent = state.get("intent")
        return SPECIALISTS.get(intent, "respond") if intent else "respond"

    return route


Decision = str
DECISIONS = ("approved", "rejected")


def pending_drafts(state: LedgerState) -> list[dict[str, Any]]:
    """Drafts nobody has decided on yet."""
    decided = state.get("reminder_decisions") or {}
    return [d for d in state.get("reminder_drafts") or [] if d["reminder_id"] not in decided]


def read_decisions(answer: Any, pending: list[dict[str, Any]]) -> dict[str, Decision]:
    """The resume value as ``{reminder_id: decision}``: a mapping of ids to decisions, or
    one decision (a string, or ``{"decision": ...}``) for every pending draft. Drafts left
    out stay pending; anything that is not a decision is ignored."""
    ids = [d["reminder_id"] for d in pending]
    if isinstance(answer, str):
        answer = {"decision": answer}
    if not isinstance(answer, dict):
        return {}
    if "decision" in answer and answer["decision"] in DECISIONS:
        return {reminder_id: str(answer["decision"]) for reminder_id in ids}
    return {k: v for k, v in answer.items() if k in ids and v in DECISIONS}


def human_approval(state: LedgerState) -> dict[str, Any]:
    """Pause until a person approves or rejects the pending drafts, then report it."""
    pending = pending_drafts(state)
    if not pending:
        return {"step_count": 1}
    answer = interrupt(
        {
            "type": "reminder_approval",
            "reminders": [
                {
                    "reminder_id": d["reminder_id"],
                    "invoice_number": d["invoice_number"],
                    "tone": d["tone"],
                    "vendor_email": d.get("vendor_email"),
                    "body": d["body"],
                }
                for d in pending
            ],
        }
    )
    decisions = read_decisions(answer, pending)
    numbers = {d["reminder_id"]: d["invoice_number"] for d in pending}
    approved = [numbers[i] for i, v in decisions.items() if v == "approved"]
    rejected = [numbers[i] for i, v in decisions.items() if v == "rejected"]
    waiting = [numbers[i] for i in numbers if i not in decisions]
    lines = []
    if approved:
        lines.append(
            f"Approved: {', '.join(approved)}. Nothing has been sent: LedgerLens drafts "
            "reminders but does not send them, so send the approved text yourself."
        )
    if rejected:
        lines.append(f"Rejected: {', '.join(rejected)}. Those drafts will not be used.")
    if waiting:
        lines.append(f"Still waiting for a decision: {', '.join(waiting)}.")
    return {
        "reminder_decisions": decisions,
        "messages": [AIMessage(" ".join(lines), name="human_approval")],
        "step_count": 1,
    }


def make_after_reminders(max_steps: int) -> Any:
    def after_reminders(state: LedgerState) -> str:
        """Drafts wait for a person; with none, the run ends."""
        return "human_approval" if pending_drafts(state) else "respond"

    return after_reminders


def make_after_extract(max_steps: int) -> Any:
    def after_extract(state: LedgerState) -> str:
        """A newly saved invoice is reconciled straight away (if the step budget allows:
        the reconciler needs two steps and respond one)."""
        extracted = state.get("extracted") or {}
        if extracted.get("doc_type") == "invoice" and state.get("step_count", 0) + 3 <= max_steps:
            return "reconciler"
        return "respond"

    return after_extract


def _failure(name: str, exc: Exception) -> dict[str, Any]:
    logger.exception("the %s node failed", name)
    text = f"Sorry, the {name.replace('_', ' ')} step failed ({type(exc).__name__}: {exc})."
    return {
        "messages": [AIMessage(text, name=name)],
        "errors": [f"{name}: {type(exc).__name__}: {exc}"],
        "step_count": 1,
    }


def _takes_config(node: Any) -> bool:
    try:
        return "config" in inspect.signature(node).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return False


def _node_span(name: str) -> Any:
    """The span of one node run (a graph interrupt is not an error)."""
    return span(name, attributes={"agent.node": name}, expected=(GraphBubbleUp,))


def _annotate(opened: Span, result: dict[str, Any]) -> None:
    """What the node decided, on its span."""
    if result.get("intent"):
        opened.set("agent.intent", result["intent"])
    for message in result.get("messages") or []:
        if isinstance(message, AIMessage) and message.text:
            opened.set_outputs(reply=message.text)


def guarded(name: str, node: Any) -> Any:
    """``node`` (sync or async) with any exception turned into an error reply, run in a
    span of its own. A node with a ``config`` parameter gets the run's config (its
    ``thread_id``)."""
    with_config = _takes_config(node)
    if inspect.iscoroutinefunction(node) or inspect.iscoroutinefunction(
        getattr(node, "__call__", None)  # noqa: B004 - a callable object's async __call__
    ):

        async def run_async(state: LedgerState, config: RunnableConfig) -> dict[str, Any]:
            with _node_span(name) as opened:
                try:
                    result: dict[str, Any] = await (
                        node(state, config) if with_config else node(state)
                    )
                except GraphBubbleUp:  # interrupt() and other control flow: not an error
                    raise
                except Exception as exc:
                    opened.record_error(exc)
                    return _failure(name, exc)
                _annotate(opened, result)
                return result

        return run_async

    def run(state: LedgerState, config: RunnableConfig) -> dict[str, Any]:
        with _node_span(name) as opened:
            try:
                result: dict[str, Any] = node(state, config) if with_config else node(state)
            except GraphBubbleUp:
                raise
            except Exception as exc:
                opened.record_error(exc)
                return _failure(name, exc)
            _annotate(opened, result)
            return result

    return run


def traced(name: str, node: Node) -> Node:
    """A built-in node (respond, stopped, human approval) in a span of its own."""

    def run(state: LedgerState) -> dict[str, Any]:
        with _node_span(name) as opened:
            try:
                result = node(state)
            except GraphBubbleUp:  # waiting for a person
                opened.set("agent.interrupted", True)
                raise
            _annotate(opened, result)
            return result

    return run


@dataclass
class AgentNodes:
    """The graph's nodes; any left out is a placeholder."""

    router: AnyNode
    qa: AnyNode = field(default_factory=lambda: unavailable("qa"))
    extractor: AnyNode = field(default_factory=lambda: unavailable("extractor"))
    reconciler: AnyNode = field(default_factory=lambda: unavailable("reconciler"))
    reminder_writer: AnyNode = field(default_factory=lambda: unavailable("reminder_writer"))


def build_graph(
    nodes: AgentNodes, *, checkpointer: Any = None, max_steps: int | None = None
) -> Any:
    """The compiled graph: START -> router -> one specialist -> respond -> END."""
    max_steps = max_steps or get_config().agents.max_steps
    graph = StateGraph(LedgerState)
    graph.add_node("router", guarded("router", nodes.router))
    for name in SPECIALISTS.values():
        graph.add_node(name, guarded(name, getattr(nodes, name)))
    graph.add_node("respond", traced("respond", respond))
    graph.add_node("stopped", traced("stopped", make_stopped(max_steps)))
    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        make_route(max_steps),
        {name: name for name in [*SPECIALISTS.values(), "respond", "stopped"]},
    )
    graph.add_node("human_approval", traced("human_approval", human_approval))
    for name in SPECIALISTS.values():
        if name not in ("extractor", "reminder_writer"):
            graph.add_edge(name, "respond")
    graph.add_conditional_edges(
        "reminder_writer",
        make_after_reminders(max_steps),
        {"human_approval": "human_approval", "respond": "respond"},
    )
    graph.add_conditional_edges(
        "human_approval",
        make_after_reminders(max_steps),  # undecided drafts: wait again
        {"human_approval": "human_approval", "respond": "respond"},
    )
    graph.add_conditional_edges(
        "extractor",
        make_after_extract(max_steps),
        {"reconciler": "reconciler", "respond": "respond"},
    )
    graph.add_edge("respond", END)
    graph.add_edge("stopped", END)
    return graph.compile(checkpointer=checkpointer)


# --------------------------------------------------------------------------- checkpointer


def checkpointer_url(database_url: str | None = None) -> str:
    """``DATABASE_URL`` as psycopg wants it (no SQLAlchemy driver suffix)."""
    url = database_url or get_settings().database_url
    return re.sub(r"^postgresql\+psycopg2?://", "postgresql://", url)


@asynccontextmanager
async def postgres_checkpointer(database_url: str | None = None) -> AsyncIterator[Any]:
    """A LangGraph ``AsyncPostgresSaver`` on ``DATABASE_URL``, its tables created (setup
    is idempotent), for the life of the block::

        async with postgres_checkpointer() as saver:
            graph = build_graph(nodes, checkpointer=saver)

    Every step of every run (``thread_id``) is saved, so an interrupted run -- reminders
    waiting for approval -- resumes after a restart. psycopg's async mode needs a selector
    event loop: on Windows, run with ``asyncio.SelectorEventLoop`` (the default Proactor
    loop is not supported)."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async with AsyncPostgresSaver.from_conn_string(checkpointer_url(database_url)) as saver:
        await saver.setup()
        yield saver
