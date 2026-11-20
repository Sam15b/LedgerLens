"""The QA node: answer a question about the ledger with GraphRAG through MCP tools.

1. **Plan** up to three calls to its tools, and only these: ``search_documents``,
   ``graph_query`` and ``run_analytics``. The ``smart`` model plans with structured output
   (:class:`QAPlan`, prompt ``qa_plan.md``); without a model, :func:`heuristic_plan` picks
   them from the question's words and the router's entities.
2. **Call** them through the :class:`~ledgerlens.agents.tools.ToolBox`. A failed call
   comes back as an error message (and in ``errors``), never as an exception.
3. **Collect evidence** (:class:`Evidence`): one line per search hit, graph row or fact
   and report row, each tagged with what it can be cited as (``doc:INV-889``, ``graph``
   for records without a document, ``analytics:aging``).
4. **Answer** with the ``answer_grounded`` prompt over that evidence (``smart``); without
   a model, list the evidence itself.
5. **Check the citations** (:func:`enforce_citations`): every answer cites evidence it
   was given, or says no evidence was found; an uncited answer is replaced by the cited
   evidence. The verdict is on the answer message (``citation_check``).

The node's messages show the trajectory: an ``AIMessage`` with the planned tool calls,
one ``ToolMessage`` per result, then the answer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel, Field

from ledgerlens.agents.nodes.router import latest_user_text
from ledgerlens.agents.state import LedgerState
from ledgerlens.agents.tools import ToolBox, ToolOutcome
from ledgerlens.core.prompts import load_prompt
from ledgerlens.llm.types import LLMError, Message

if TYPE_CHECKING:
    from ledgerlens.llm.router import LLMClient

logger = logging.getLogger(__name__)

QA_TOOLS = ("search_documents", "graph_query", "run_analytics")
QAToolName = Literal["search_documents", "graph_query", "run_analytics"]
MAX_CALLS = 3
MAX_LINES = 25
SNIPPET = 280


class PlannedCall(BaseModel):
    tool: QAToolName
    args: dict[str, Any] = Field(default_factory=dict)
    why: str = ""


class QAPlan(BaseModel):
    """The model's lookup plan (structured output)."""

    calls: list[PlannedCall] = Field(default_factory=list, max_length=MAX_CALLS)


# --------------------------------------------------------------------------- planning

_UNPAID = re.compile(r"\b(?:unpaid|open|outstanding|owe[sd]?|overdue|not (?:yet )?paid)\b", re.I)
_TOTALS = re.compile(r"\b(?:how much|total|balance|ledger|paid)\b", re.I)
_AGING = re.compile(r"\bage?ing\b", re.I)
_SPEND = re.compile(r"\b(?:spend|spent)\b", re.I)
_MIN_AMOUNT = re.compile(
    r"\b(?:over|above|more than|greater than|exceeding|at least)\s*[$£€]?\s*"
    r"(\d[\d,]*(?:\.\d+)?)\s*(k|m)?\b",
    re.I,
)


def min_amount(question: str) -> float | None:
    """``"over $5k"`` -> 5000.0 (``None`` if the question sets no lower bound)."""
    match = _MIN_AMOUNT.search(question)
    if match is None:
        return None
    value = float(match.group(1).replace(",", ""))
    return value * {"k": 1_000, "m": 1_000_000}.get((match.group(2) or "").lower(), 1)


def heuristic_plan(question: str, entities: Mapping[str, Any]) -> list[PlannedCall]:
    """Calls chosen by keywords, for when no model plans."""
    vendor_ids = list(entities.get("vendor_ids") or [])
    vendor = vendor_ids[0] if vendor_ids else entities.get("vendor")
    search_args: dict[str, Any] = {"query": question, "k": 5}
    if vendor:
        search_args["vendor"] = vendor
    calls = [PlannedCall(tool="search_documents", args=search_args, why="the documents")]
    seeds = [*entities.get("invoice_numbers", []), *entities.get("po_numbers", [])]
    vendor_param = {"vendor_ids": [vendor]} if vendor else {}
    if seeds:
        calls.append(
            PlannedCall(
                tool="graph_query",
                args={"template": "invoice_neighborhood", "params": {"seeds": seeds}},
                why="the records named",
            )
        )
    elif _UNPAID.search(question):
        calls.append(
            PlannedCall(
                tool="graph_query",
                args={"template": "unpaid_with_po", "params": {**vendor_param, "limit": 50}},
                why="unpaid invoices",
            )
        )
    elif _TOTALS.search(question) and vendor:
        calls.append(
            PlannedCall(
                tool="graph_query",
                args={"template": "vendor_ledger", "params": vendor_param},
                why="the vendor's ledger",
            )
        )
    if _AGING.search(question):
        calls.append(PlannedCall(tool="run_analytics", args={"report": "aging"}, why="aging"))
    elif _SPEND.search(question):
        params = {"vendor": vendor} if vendor else {}
        calls.append(
            PlannedCall(
                tool="run_analytics",
                args={"report": "spend_by_vendor", "params": params},
                why="spend",
            )
        )
    return calls[:MAX_CALLS]


# --------------------------------------------------------------------------- evidence


Kind = Literal["row", "fact", "report", "hit"]
KIND_ORDER: dict[str, int] = {"row": 0, "fact": 1, "report": 2, "hit": 3}
"""Graph rows answer a question directly; document text supports it."""
_CONTEXT_HEADER = re.compile(r"\A\[[^\]\n]*\]\s*")


@dataclass(frozen=True)
class Line:
    kind: Kind
    tags: tuple[str, ...]
    """Citation tags without brackets: ``doc:INV-889``, ``graph``, ``analytics:aging``."""
    text: str

    def render(self) -> str:
        return f"{' '.join(f'[{tag}]' for tag in self.tags)} {self.text}"


@dataclass
class Evidence:
    """What the tool calls found, as citable lines."""

    lines: list[Line] = field(default_factory=list)
    citations: dict[str, str] = field(default_factory=dict)
    """``doc:`` tag -> document id."""
    notes: list[str] = field(default_factory=list)
    """Things to say that are not evidence (a filter that left nothing)."""

    @property
    def tags(self) -> set[str]:
        return {tag for line in self.lines for tag in line.tags}

    def add(self, kind: Kind, tags: Sequence[str], text: str) -> None:
        line = Line(kind, tuple(tags) or ("graph",), " ".join(text.split()))
        if line not in self.lines:
            self.lines.append(line)

    def ranked(self) -> list[Line]:
        return sorted(self.lines, key=lambda line: KIND_ORDER[line.kind])

    def strongest(self) -> list[Line]:
        """The lines of the most direct kind found (graph rows before document text)."""
        ranked = self.ranked()
        return [line for line in ranked if line.kind == ranked[0].kind] if ranked else []

    def render(self, limit: int = MAX_LINES) -> str:
        return "\n".join(line.render() for line in self.ranked()[:limit])


def _tags_in(text: str, citations: Mapping[str, str]) -> list[str]:
    """The citation tags whose document number appears in ``text``."""
    return [tag for tag in citations if tag.startswith("doc:") and tag[4:] in text]


def _money(value: Any) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)


def _invoice_line(row: Mapping[str, Any]) -> str:
    po = row.get("po") or {}
    parts = [
        f"Invoice {row.get('invoice_number')} from {row.get('vendor')}",
        f"total {_money(row.get('total'))} {row.get('currency', '')}".strip(),
        f"outstanding {_money(row.get('outstanding'))}",
        f"due {row.get('due_date')}" if row.get("due_date") else "",
        f"PO {po.get('po_number')}" if isinstance(po, Mapping) and po.get("po_number") else "",
    ]
    return ", ".join(part for part in parts if part)


def _add_invoice_rows(
    evidence: Evidence, rows: Sequence[Mapping[str, Any]], minimum: float | None
) -> None:
    kept = [r for r in rows if minimum is None or float(r.get("outstanding") or 0) >= minimum]
    if minimum is not None and not kept:
        # nothing passes the filter: the rows found are the evidence for saying so
        evidence.notes.append(
            f"none of the {len(rows)} invoice(s) found has more than {_money(minimum)} outstanding"
        )
        kept = list(rows)
    for row in kept:
        document_id = row.get("document_id")
        tag = f"doc:{row['invoice_number']}" if document_id else "graph"
        if document_id:
            evidence.citations[tag] = str(document_id)
        evidence.add("row", [tag], _invoice_line(row))


def collect_evidence(outcomes: Sequence[ToolOutcome], *, minimum: float | None = None) -> Evidence:
    evidence = Evidence()
    for outcome in outcomes:
        if not outcome.ok:
            continue
        data = outcome.data or {}
        citations = {k: v for k, v in (data.get("citations") or {}).items() if k.startswith("doc:")}
        evidence.citations.update(citations)
        if outcome.tool == "search_documents":
            for hit in data.get("hits", []):
                text = _CONTEXT_HEADER.sub("", hit["text"])[:SNIPPET]
                evidence.add("hit", [hit["citation"]], text)
            for fact in data.get("facts", []):
                evidence.add("fact", _tags_in(fact, citations), fact)
        elif outcome.tool == "graph_query":
            rows = data.get("rows", [])
            if rows and all("invoice_number" in row and "outstanding" in row for row in rows):
                _add_invoice_rows(evidence, rows, minimum)
            else:
                for fact in data.get("facts", []):
                    evidence.add("fact", _tags_in(fact, citations), fact)
        elif outcome.tool == "run_analytics":
            report = data.get("report", "report")
            for row in data.get("rows", [])[:10]:
                text = ", ".join(f"{k} {v}" for k, v in row.items())
                evidence.add("report", [f"analytics:{report}"], text)
    return evidence


def evidence_answer(question: str, evidence: Evidence) -> str:
    """The answer when no model writes one: the most direct evidence, listed."""
    if not evidence.lines:
        note = f" ({'; '.join(evidence.notes)})" if evidence.notes else ""
        return f"I found no evidence in the ledger for that{note}."
    lines = evidence.strongest()
    head = "Here is what the ledger shows"
    if evidence.notes:
        head = f"{head}: {'; '.join(evidence.notes)}"
    bullets = [
        f"- {line.text} {' '.join(f'[{tag}]' for tag in line.tags)}" for line in lines[:MAX_LINES]
    ]
    if len(lines) > MAX_LINES:
        bullets.append(f"- ... and {len(lines) - MAX_LINES} more")
    return head + ":\n" + "\n".join(bullets)


# --------------------------------------------------------------------------- citations

TAG = re.compile(r"\[(doc:[^\]\s]+|graph|analytics:[a-z_]+)\]")
NO_EVIDENCE = re.compile(
    r"\b(?:no evidence|could ?n[o']t find|can ?n[o']t find|did ?n[o']t find|"
    r"(?:does|do) ?n[o']t (?:contain|show|include)|there (?:is|are) no (?:record|invoice|"
    r"payment|data|information)|not (?:in|found in) the (?:context|ledger))",
    re.IGNORECASE,
)
CitationCheck = Literal["cited", "no_evidence", "replaced"]


def enforce_citations(
    answer: str, evidence: Evidence, question: str = ""
) -> tuple[str, CitationCheck]:
    """Every answer cites the evidence it rests on, or says that none was found.

    Citations to anything the evidence does not hold are removed (a model can invent
    ``[doc:INV-999]``). An answer left with a real citation stands (``cited``); one that
    says nothing was found stands when that is true (``no_evidence``); anything else is
    replaced by the cited evidence itself (``replaced``)."""
    valid = evidence.tags
    cleaned = TAG.sub(lambda m: m.group(0) if m.group(1) in valid else "", answer)
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", re.sub(r"[ \t]{2,}", " ", cleaned)).strip()
    if TAG.search(cleaned):
        return cleaned, "cited"
    if not evidence.lines:
        return evidence_answer(question, evidence), "no_evidence"
    if NO_EVIDENCE.search(cleaned):
        return cleaned, "no_evidence"
    logger.warning("answer cites none of its evidence; answering with the evidence instead")
    return evidence_answer(question, evidence), "replaced"


# --------------------------------------------------------------------------- the node


class QANode:
    """Answers ledger questions through its three MCP tools."""

    def __init__(
        self,
        tools: ToolBox,
        llm: LLMClient | None = None,
        *,
        model: str = "smart",
        max_calls: int = MAX_CALLS,
        max_steps: int | None = None,
        today: date | None = None,
    ) -> None:
        self.tools = tools.only(*QA_TOOLS)
        self.llm = llm
        self.model = model
        self.max_calls = max_calls
        self.max_steps = max_steps
        """The graph's step limit: tool calls stop where they would cross it."""
        self._today = today
        self.plan_prompt = load_prompt("qa_plan")
        self.answer_prompt = load_prompt("answer_grounded")

    @property
    def today(self) -> date:
        return self._today or date.today()

    async def plan(self, question: str, entities: Mapping[str, Any]) -> list[PlannedCall]:
        if self.llm is not None:
            prompt = self.plan_prompt.render(
                today=self.today.isoformat(),
                question=question,
                entities=json.dumps(dict(entities), default=str),
                max_calls=self.max_calls,
            )
            try:
                plan = await asyncio.to_thread(
                    self.llm.structured, [Message.user(prompt)], QAPlan, model=self.model
                )
                if plan.calls:
                    return plan.calls[: self.max_calls]
            except LLMError as exc:
                logger.warning("QA planner unavailable (%s); planning by keywords", exc)
        return heuristic_plan(question, entities)[: self.max_calls]

    async def answer(self, question: str, evidence: Evidence) -> tuple[str, CitationCheck]:
        """The answer, checked by :func:`enforce_citations`."""
        if self.llm is None or not evidence.lines:
            text = evidence_answer(question, evidence)
            return text, "cited" if evidence.lines else "no_evidence"
        prompt = self.answer_prompt.render(
            today=self.today.isoformat(), question=question, context=evidence.render()
        )
        try:
            response = await asyncio.to_thread(
                self.llm.complete, [Message.user(prompt)], model=self.model
            )
        except LLMError as exc:
            logger.warning("QA answer model unavailable (%s); listing the evidence", exc)
            return evidence_answer(question, evidence), "cited"
        return enforce_citations(response.text.strip(), evidence, question)

    async def __call__(self, state: LedgerState) -> dict[str, Any]:
        question = latest_user_text(state.get("messages", []))
        entities = state.get("entities") or {}
        calls = await self.plan(question, entities)
        errors: list[str] = []
        if self.max_steps is not None:
            # this node and respond take a step each; every tool call takes one
            budget = max(0, self.max_steps - state.get("step_count", 0) - 2)
            if len(calls) > budget:
                errors.append(f"step limit: {len(calls) - budget} planned tool call(s) skipped")
                calls = calls[:budget]
        outcomes: list[ToolOutcome] = []
        tool_messages: list[BaseMessage] = []
        for call in calls:
            outcome, message = await self.tools.call(call.tool, call.args)
            outcomes.append(outcome)
            tool_messages.append(message)
        evidence = collect_evidence(outcomes, minimum=min_amount(question))
        text, check = await self.answer(question, evidence)
        planned = AIMessage(
            "",
            name="qa",
            tool_calls=[
                {"name": o.tool, "args": o.args, "id": o.call_id, "type": "tool_call"}
                for o in outcomes
            ],
        )
        return {
            "messages": [
                planned,
                *tool_messages,
                AIMessage(text, name="qa", additional_kwargs={"citation_check": check}),
            ],
            "citations": evidence.citations,
            "errors": [*errors, *(f"{o.tool}: {o.text}" for o in outcomes if not o.ok)],
            "step_count": 1 + len(outcomes),
        }
