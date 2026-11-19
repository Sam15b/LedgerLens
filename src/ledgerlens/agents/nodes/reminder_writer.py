"""The reminder writer: draft payment reminders for overdue invoices, for a person to
approve. Nothing is ever sent (docs/layers/04-agents-mcp.md, "Reminder writer").

Which invoices: the ones the request names, else what ``list_overdue`` returns (for the
vendor named, if any), at most :data:`MAX_DRAFTS`. For each, through the MCP tools:

1. ``get_invoice`` -- amount, outstanding, original due date, PO number, status;
2. the tone from days overdue (:func:`~ledgerlens.mcp_server.tools.reminders.tone_for`):
   1-15 ``friendly``, 16-45 ``firm``, 46+ ``final_notice``; not overdue, paid or void:
   no draft;
3. ``get_vendor_contact`` -- who it goes to, and whether there is an email on file;
4. the draft (:func:`render_body`): the tone's template (``reminder_<tone>.md``) filled
   with the facts, so every draft states the invoice number, the amount, the original due
   date and the PO number;
5. ``draft_reminder`` -- stored as ``pending_approval``; it checks the facts again.

The drafts go to ``reminder_drafts``; the graph then waits for a person to approve or
reject them (:mod:`ledgerlens.agents.graph`).
"""

from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig

from ledgerlens.agents.state import LedgerState
from ledgerlens.agents.tools import ToolBox, ToolOutcome
from ledgerlens.core.prompts import PromptError, load_prompt
from ledgerlens.mcp_server.tools.reminders import Tone, tone_for

logger = logging.getLogger(__name__)

REMINDER_TOOLS = ("list_overdue", "get_invoice", "get_vendor_contact", "draft_reminder")
MAX_DRAFTS = 5
SENDER = "Accounts Payable"


def _amount(value: Any) -> str:
    return f"{Decimal(str(value)):,.2f}"


def _day(value: str) -> str:
    """``2026-04-13`` -> ``13 April 2026``."""
    day = date.fromisoformat(value)
    return f"{day.day} {day:%B %Y}"


def reminder_facts(
    invoice: dict[str, Any], contact: dict[str, Any] | None, days_overdue: int
) -> dict[str, str]:
    """What a reminder states, as template values."""
    return {
        "recipient": (contact or {}).get("name") or invoice["vendor_name"],
        "invoice_number": invoice["invoice_number"],
        "currency": invoice["currency"],
        "amount": _amount(invoice["outstanding"]),
        "total": _amount(invoice["total"]),
        "due_date": _day(invoice["due_date"]),
        "po_number": invoice.get("po_number") or "(no PO quoted)",
        "days_overdue": str(days_overdue),
        "sender": SENDER,
    }


def render_body(tone: Tone, facts: dict[str, str]) -> str:
    """The tone's template filled with ``facts``; a plain statement of the facts if the
    template cannot be used."""
    try:
        return load_prompt(f"reminder_{tone}").render(**facts).strip()
    except PromptError as exc:
        logger.warning(
            "reminder template for %s unusable (%s); writing the facts plainly", tone, exc
        )
        return (
            f"Dear {facts['recipient']},\n\nInvoice {facts['invoice_number']} for "
            f"{facts['currency']} {facts['amount']} (PO {facts['po_number']}) was due on "
            f"{facts['due_date']} and is {facts['days_overdue']} days overdue.\n\n"
            f"{facts['sender']}"
        )


class ReminderWriterNode:
    """Drafts reminders through ``list_overdue``, ``get_invoice``, ``get_vendor_contact``
    and ``draft_reminder``."""

    def __init__(
        self, tools: ToolBox, *, today: date | None = None, max_drafts: int = MAX_DRAFTS
    ) -> None:
        self.tools = tools.only(*REMINDER_TOOLS)
        self._today = today
        self.max_drafts = max_drafts

    @property
    def today(self) -> date:
        return self._today or date.today()

    async def __call__(
        self, state: LedgerState, config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        thread_id = ((config or {}).get("configurable") or {}).get("thread_id")
        entities = state.get("entities") or {}
        vendor_ids = entities.get("vendor_ids") or []
        vendor = vendor_ids[0] if vendor_ids else None
        outcomes: list[ToolOutcome] = []
        trajectory: list[BaseMessage] = []

        async def call(name: str, args: dict[str, Any]) -> ToolOutcome:
            outcome, message = await self.tools.call(name, args)
            outcomes.append(outcome)
            trajectory.append(message)
            return outcome

        invoices = list(entities.get("invoice_numbers") or [])
        if not invoices:
            args: dict[str, Any] = {
                "min_days": 1,
                "limit": self.max_drafts,
                "as_of": self.today.isoformat(),
            }
            if vendor:
                args["vendor"] = vendor
            overdue = await call("list_overdue", args)
            invoices = [item["invoice_id"] for item in (overdue.data or {}).get("invoices", [])]
        drafts: list[dict[str, Any]] = []
        notes: list[str] = []
        for invoice in invoices[: self.max_drafts]:
            found = await call(
                "get_invoice", {"invoice": invoice, **({"vendor": vendor} if vendor else {})}
            )
            if not found.ok or found.data is None:
                notes.append(f"{invoice}: {found.text}")
                continue
            record = found.data
            if record["status"] in ("paid", "void"):
                notes.append(f"{record['invoice_number']} is {record['status']}: no reminder")
                continue
            due = date.fromisoformat(record["due_date"]) if record.get("due_date") else None
            days = (self.today - due).days if due is not None else 0
            tone = tone_for(days)
            if tone is None:
                when = f"due {due.isoformat()}" if due else "no due date"
                notes.append(f"{record['invoice_number']} is not overdue ({when}): no reminder")
                continue
            contact_out = await call(
                "get_vendor_contact", {"vendor": record.get("vendor_id") or record["vendor_name"]}
            )
            contact = contact_out.data if contact_out.ok else None
            body = render_body(tone, reminder_facts(record, contact, days))
            draft_args = {"invoice": record["id"], "body": body, "tone": tone}
            if thread_id:
                draft_args["thread_id"] = str(thread_id)  # approving it resumes this run
            drafted = await call("draft_reminder", draft_args)
            if not drafted.ok or drafted.data is None:
                notes.append(f"{record['invoice_number']}: {drafted.text}")
                continue
            drafts.append(
                {
                    **drafted.data,
                    "vendor_name": record["vendor_name"],
                    "citation": record.get("citation"),
                    "document_id": record.get("document_id"),
                }
            )
        lines = []
        if drafts:
            lines.append(
                f"I drafted {len(drafts)} reminder(s). They are waiting for your approval; "
                "nothing has been sent."
            )
        for draft in drafts:
            email = draft.get("vendor_email") or "no email on file"
            cite = f" [{draft['citation']}]" if draft.get("citation") else ""
            lines.append(
                f"- {draft['invoice_number']}{cite} to {draft['vendor_name']} ({email}): "
                f"{draft['tone'].replace('_', ' ')}, {draft['days_overdue']} days overdue; "
                f"reminder {draft['reminder_id']}"
            )
        lines += [f"- {note}" for note in notes]
        if not drafts and not notes:
            lines.append("Nothing is overdue" + (f" for {vendor}" if vendor else "") + ".")
        citations = {d["citation"]: d.get("document_id", "") for d in drafts if d.get("citation")}
        planned = AIMessage(
            "",
            name="reminder_writer",
            tool_calls=[
                {"name": o.tool, "args": o.args, "id": o.call_id, "type": "tool_call"}
                for o in outcomes
            ],
        )
        return {
            "messages": [planned, *trajectory, AIMessage("\n".join(lines), name="reminder_writer")],
            "reminder_drafts": drafts,
            "citations": {k: v for k, v in citations.items() if v},
            "errors": [n for n in notes if ": no reminder" not in n],
            "step_count": 1 + len(outcomes),
        }
