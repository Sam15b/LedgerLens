"""The state every agent node reads and updates (docs/layers/04-agents-mcp.md).

A node returns only the keys it changes; LangGraph merges them in with each key's
reducer: ``messages`` are appended (by id, :func:`add_messages`), ``errors`` and
``reminder_drafts`` are appended, ``citations`` are merged, and ``step_count`` adds up, so
each node returns ``{"step_count": 1}`` and the graph can stop a run that goes in circles.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Intent = Literal["question", "extract", "reconcile", "remind", "other"]
INTENTS: tuple[Intent, ...] = ("question", "extract", "reconcile", "remind", "other")


def merge_citations(left: dict[str, str] | None, right: dict[str, str] | None) -> dict[str, str]:
    """Citation tag -> document id, from every node that cited something."""
    return {**(left or {}), **(right or {})}


class LedgerState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    intent: Intent | None
    entities: dict[str, Any]
    """What the router found in the request: invoice / PO numbers, a vendor, ..."""
    document_ids: list[str]
    """Documents the request is about (an upload, or ones the user named)."""
    extracted: dict[str, Any] | None
    """The record the extractor read from a document."""
    reconciliation: dict[str, Any] | None
    """The reconcile tool's match report."""
    reminder_drafts: Annotated[list[dict[str, Any]], operator.add]
    reminder_decisions: Annotated[dict[str, str], merge_citations]
    """Reminder id -> ``approved`` / ``rejected``, as a person decided."""
    citations: Annotated[dict[str, str], merge_citations]
    step_count: Annotated[int, operator.add]
    errors: Annotated[list[str], operator.add]
