"""The agent graph: a router hands each request to one specialist (LangGraph).

    graph = build_graph(AgentNodes(router=..., qa=...))
    result = graph.invoke({"messages": [HumanMessage("Which Acme invoices are unpaid?")]})

The router classifies the latest user message into an :data:`~ledgerlens.agents.state
.Intent`; :func:`route` sends ``question`` to QA, ``extract`` to the extractor,
``reconcile`` to the reconciler, ``remind`` to the reminder writer and ``other`` straight
to ``respond``. Every specialist ends in ``respond`` (docs/layers/04-agents-mcp.md).

Nodes are plain functions of the state, built with their dependencies (LLM, tools) and
passed in as :class:`AgentNodes`, so tests can run the graph with stand-ins. A node that
is not built yet answers that it is not available.

Guardrails:

* **step limit**: every node adds to ``step_count`` (a tool call is a step too); once it
  reaches ``max_steps`` (``configs/agents.yaml``, 8) the graph goes straight to
  ``respond``, which says the run was stopped;
* **errors are messages**: :func:`guarded` wraps every node, so an exception inside one
  becomes an error reply and an ``errors`` entry instead of failing the request. Tool
  errors already come back as ``ToolMessage`` s (:class:`~ledgerlens.agents.tools.ToolBox`).
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from ledgerlens.agents.state import Intent, LedgerState
from ledgerlens.core.config import get_config

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


def make_respond(max_steps: int) -> Node:
    def respond(state: LedgerState) -> dict[str, Any]:
        """The last step: say so if the step limit stopped the run; answer out-of-scope
        requests; anything else is already answered."""
        steps = state.get("step_count", 0)
        if steps >= max_steps:
            text = (
                f"I stopped after {steps} steps (the limit is {max_steps}) before finishing. "
                "Try a narrower request."
            )
            return {
                "messages": [AIMessage(text, name="respond")],
                "errors": [f"step limit: {steps} >= {max_steps}"],
                "step_count": 1,
            }
        if state.get("intent") in (None, "other"):
            return {"messages": [AIMessage(OUT_OF_SCOPE, name="respond")], "step_count": 1}
        return {"step_count": 1}

    return respond


def make_route(max_steps: int) -> Any:
    def route(state: LedgerState) -> str:
        """The node for the router's intent (``respond`` for ``other``, or once the step
        limit is reached)."""
        if state.get("step_count", 0) >= max_steps:
            return "respond"
        intent = state.get("intent")
        return SPECIALISTS.get(intent, "respond") if intent else "respond"

    return route


def _failure(name: str, exc: Exception) -> dict[str, Any]:
    logger.exception("the %s node failed", name)
    text = f"Sorry, the {name.replace('_', ' ')} step failed ({type(exc).__name__}: {exc})."
    return {
        "messages": [AIMessage(text, name=name)],
        "errors": [f"{name}: {type(exc).__name__}: {exc}"],
        "step_count": 1,
    }


def guarded(name: str, node: Any) -> Any:
    """``node`` (sync or async) with any exception turned into an error reply."""
    if inspect.iscoroutinefunction(node) or inspect.iscoroutinefunction(
        getattr(node, "__call__", None)  # noqa: B004 - a callable object's async __call__
    ):

        async def run_async(state: LedgerState) -> dict[str, Any]:
            try:
                result: dict[str, Any] = await node(state)
            except Exception as exc:
                return _failure(name, exc)
            return result

        return run_async

    def run(state: LedgerState) -> dict[str, Any]:
        try:
            result: dict[str, Any] = node(state)
        except Exception as exc:
            return _failure(name, exc)
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
    graph.add_node("respond", make_respond(max_steps))
    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        make_route(max_steps),
        {name: name for name in [*SPECIALISTS.values(), "respond"]},
    )
    for name in SPECIALISTS.values():
        graph.add_edge(name, "respond")
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=checkpointer)
