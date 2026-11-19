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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from ledgerlens.agents.state import Intent, LedgerState


class Node(Protocol):
    """A graph node: the state in, the keys it changes out."""

    def __call__(self, state: LedgerState) -> dict[str, Any]: ...


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


def route(state: LedgerState) -> str:
    """The node for the router's intent (``respond`` for ``other``)."""
    intent = state.get("intent")
    return SPECIALISTS.get(intent, "respond") if intent else "respond"


@dataclass
class AgentNodes:
    """The graph's nodes; any left out is a placeholder."""

    router: Node
    qa: Node = field(default_factory=lambda: unavailable("qa"))
    extractor: Node = field(default_factory=lambda: unavailable("extractor"))
    reconciler: Node = field(default_factory=lambda: unavailable("reconciler"))
    reminder_writer: Node = field(default_factory=lambda: unavailable("reminder_writer"))


def build_graph(nodes: AgentNodes, *, checkpointer: Any = None) -> Any:
    """The compiled graph: START -> router -> one specialist -> respond -> END."""
    graph = StateGraph(LedgerState)
    graph.add_node("router", nodes.router)
    for name in SPECIALISTS.values():
        graph.add_node(name, getattr(nodes, name))
    graph.add_node("respond", respond)
    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router", route, {name: name for name in [*SPECIALISTS.values(), "respond"]}
    )
    for name in SPECIALISTS.values():
        graph.add_edge(name, "respond")
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=checkpointer)
