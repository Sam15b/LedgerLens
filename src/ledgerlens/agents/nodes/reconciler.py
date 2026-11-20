"""The reconciler node: run the deterministic 3-way match and explain it.

The match itself is the MCP ``reconcile`` tool (code, not judgment: see
``mcp_server/tools/reconcile.py``). This node picks the invoice(s) -- the one the
extractor just saved, else the invoice numbers in the request -- calls ``reconcile`` for
each, and answers with:

* a short explanation by the ``smart`` model (prompt ``reconciler.md``), when one answers;
* then, always, the report itself (:func:`report_text`): the verdict, every failed or
  flagged check with its message, and the checks that passed. The model explains the
  report; it cannot change the mismatch list, because the list is printed from the
  report, not from the model.

Its tools: ``get_invoice``, ``get_po``, ``get_contract``, ``get_payments`` and
``reconcile``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage

from ledgerlens.agents.nodes.router import latest_user_text
from ledgerlens.agents.state import LedgerState
from ledgerlens.agents.tools import ToolBox, ToolOutcome
from ledgerlens.llm.types import LLMError, Message

if TYPE_CHECKING:
    from ledgerlens.llm.router import LLMClient

logger = logging.getLogger(__name__)

RECONCILER_TOOLS = ("get_invoice", "get_po", "get_contract", "get_payments", "reconcile")
MAX_INVOICES = 3
PROMPT = "reconciler"


def report_text(report: Mapping[str, Any]) -> str:
    """The match report in words: the verdict, the failed and flagged checks, the passes."""
    number = report["invoice_number"]
    cite = f" [{report['citation']}]" if report.get("citation") else ""
    against = " and ".join(
        part
        for part in (
            f"PO {report['po_number']}" if report.get("po_number") else "",
            f"contract {report['contract_number']}" if report.get("contract_number") else "",
        )
        if part
    )
    against = f" against {against}" if against else ""
    checks: Sequence[Mapping[str, Any]] = report.get("checks", [])
    if report["status"] == "match":
        head = f"{number}{cite} matches{against}: every check passed"
        if report.get("warnings"):
            head += f" (to look at: {', '.join(report['warnings'])})"
    else:
        head = (
            f"{number}{cite} does not match{against}. Mismatches: {', '.join(report['mismatches'])}"
        )
    lines = [head + "."]
    for check in checks:
        if check["status"] in ("fail", "warn"):
            label = "mismatch" if check["status"] == "fail" else "to look at"
            lines.append(f"- {check['check']} ({label}): {check['message']}")
    passed = [c["check"] for c in checks if c["status"] == "pass"]
    skipped = [c["check"] for c in checks if c["status"] == "skipped"]
    if passed and report["status"] != "match":
        lines.append(f"Passed: {', '.join(passed)}.")
    if skipped:
        lines.append(f"Not checked (nothing to compare): {', '.join(skipped)}.")
    return "\n".join(lines)


def _targets(state: LedgerState) -> tuple[list[str], str | None]:
    """The invoices to match, and a vendor to narrow numbers down with."""
    extracted = state.get("extracted") or {}
    if extracted.get("doc_type") == "invoice" and extracted.get("record_id"):
        return [str(extracted["record_id"])], None
    entities = state.get("entities") or {}
    vendor_ids = entities.get("vendor_ids") or []
    return list(entities.get("invoice_numbers") or []), (vendor_ids[0] if vendor_ids else None)


class ReconcilerNode:
    """Matches invoices with the ``reconcile`` tool and explains the reports."""

    def __init__(
        self, tools: ToolBox, llm: LLMClient | None = None, *, model: str = "smart"
    ) -> None:
        self.tools = tools.only(*RECONCILER_TOOLS)
        self.llm = llm
        self.model = model

    async def narrate(self, question: str, reports: Sequence[Mapping[str, Any]]) -> str | None:
        """The model's short explanation of the reports (``None`` without one)."""
        if self.llm is None or not reports:
            return None
        try:
            from ledgerlens.core.prompts import load_prompt

            prompt = load_prompt(PROMPT).render(
                question=question, reports=json.dumps(list(reports), indent=1, default=str)
            )
            response = await asyncio.to_thread(
                self.llm.complete, [Message.user(prompt)], model=self.model
            )
        except (LLMError, ValueError) as exc:  # no model, or no prompt
            logger.warning("reconciler narration unavailable (%s); reporting the checks", exc)
            return None
        return response.text.strip() or None

    async def __call__(self, state: LedgerState) -> dict[str, Any]:
        invoices, vendor = _targets(state)
        if not invoices:
            text = "Which invoice should I match? Give its number (for example INV-889)."
            return {"messages": [AIMessage(text, name="reconciler")], "step_count": 1}
        outcomes: list[ToolOutcome] = []
        trajectory: list[BaseMessage] = []
        for invoice in invoices[:MAX_INVOICES]:
            args: dict[str, Any] = {"invoice": invoice}
            if vendor:
                args["vendor"] = vendor
            outcome, message = await self.tools.call("reconcile", args)
            outcomes.append(outcome)
            trajectory.append(message)
        reports = [o.data for o in outcomes if o.ok and o.data]
        problems = [f"reconcile {o.args['invoice']}: {o.text}" for o in outcomes if not o.ok]
        question = latest_user_text(state.get("messages", []))
        narrative = await self.narrate(question, reports)
        parts = [narrative] if narrative else []
        parts += [report_text(report) for report in reports]
        parts += [f"Could not match: {problem}" for problem in problems]
        citations: dict[str, str] = {}
        for report in reports:
            citations.update(report.get("citations") or {})
        planned = AIMessage(
            "",
            name="reconciler",
            tool_calls=[
                {"name": o.tool, "args": o.args, "id": o.call_id, "type": "tool_call"}
                for o in outcomes
            ],
        )
        reconciliation: dict[str, Any] | None
        if len(reports) == 1:
            reconciliation = reports[0]
        else:
            reconciliation = {"reports": reports} if reports else None
        return {
            "messages": [planned, *trajectory, AIMessage("\n\n".join(parts), name="reconciler")],
            "reconciliation": reconciliation,
            "citations": citations,
            "errors": problems,
            "step_count": 1 + len(outcomes),
        }
