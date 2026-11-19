"""``draft_reminder``: store a payment reminder for a person to approve. It never sends.

A draft is saved in ``reminders`` with status ``pending_approval``; only a person moves it
on (approve or reject, Day 60's API), and there is no code path from here to any outbox.
The tone follows days overdue (docs/layers/04-agents-mcp.md): 1-15 ``friendly``, 16-45
``firm``, 46+ ``final_notice``. A reminder has to carry the facts the vendor needs to find
the invoice -- its number, the amount, the original due date and the PO number --
:func:`missing_facts` lists any the body leaves out, so the writer can fix the draft.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from ledgerlens.anomaly.rules import money
from ledgerlens.mcp_server.services import Services
from ledgerlens.mcp_server.tools import in_thread
from ledgerlens.mcp_server.tools.records import find_invoice

Tone = Literal["friendly", "firm", "final_notice"]
TONE_DAYS: tuple[tuple[int, Tone], ...] = ((46, "final_notice"), (16, "firm"), (1, "friendly"))
MAX_BODY = 5_000

DRAFT = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False
)
"""Adds a draft (for approval; nothing leaves the system)."""


def tone_for(days_overdue: int) -> Tone | None:
    """The tone for an invoice this many days past due (``None`` if not overdue)."""
    return next((tone for start, tone in TONE_DAYS if days_overdue >= start), None)


def _date_forms(day: date) -> tuple[str, ...]:
    """How a due date may be written in a reminder."""
    return (
        day.isoformat(),
        day.strftime("%d %b %Y"),
        day.strftime("%d %B %Y"),
        f"{day.day} {day:%B %Y}",
        f"{day.day} {day:%b %Y}",
        f"{day:%B} {day.day}, {day.year}",
        f"{day:%b} {day.day}, {day.year}",
        day.strftime("%d/%m/%Y"),
    )


def missing_facts(
    body: str,
    *,
    invoice_number: str,
    amounts: tuple[str, ...],
    due_date: date | None,
    po_number: str | None,
) -> list[str]:
    """The facts a reminder must state that ``body`` does not: ``invoice_number``,
    ``amount`` (the total or the outstanding amount, with or without thousands
    separators), ``due_date`` and ``po_number`` (when the invoice has them)."""
    text = body.casefold()
    plain = text.replace(",", "")
    missing = []
    if invoice_number.casefold() not in text:
        missing.append("invoice_number")
    if not any(a.casefold() in plain for a in amounts):
        missing.append("amount")
    if due_date is not None and not any(f.casefold() in text for f in _date_forms(due_date)):
        missing.append("due_date")
    if po_number and po_number.casefold() not in text:
        missing.append("po_number")
    return missing


class ReminderDraft(BaseModel):
    reminder_id: str
    invoice_id: str
    invoice_number: str
    tone: Tone
    suggested_tone: Tone | None
    """The tone for its days overdue (``None``: not overdue)."""
    days_overdue: int
    status: Literal["pending_approval"] = "pending_approval"
    sent: Literal[False] = False
    """Never: a person approves the draft; this system does not send."""
    vendor_email: str | None
    """``None``: no email on file for the vendor."""
    missing_facts: list[str] = Field(default_factory=list)
    """Facts the body should state but does not (invoice_number, amount, due_date,
    po_number): fix and draft again before asking for approval."""
    body: str
    created_at: datetime


def draft_reminder(
    services: Services, invoice: str, body: str, tone: Tone, vendor: str | None = None
) -> ReminderDraft:
    from ledgerlens.storage.postgres.db import session_scope
    from ledgerlens.storage.postgres.repositories import (
        InvoiceRepository,
        ReminderRepository,
        VendorRepository,
    )

    body = body.strip()
    if not body:
        raise ToolError("the reminder body is empty")
    if len(body) > MAX_BODY:
        raise ToolError(f"the reminder body is over {MAX_BODY} characters")
    if tone not in ("friendly", "firm", "final_notice"):
        raise ToolError("tone must be friendly, firm or final_notice")
    with session_scope(services.sessions) as session:
        row = find_invoice(session, services, invoice, vendor)
        if row.status in ("paid", "void"):
            raise ToolError(f"{row.invoice_number} is {row.status}: there is nothing to chase")
        paid = InvoiceRepository(session).paid_amounts([row.id])[row.id]
        overdue = (
            (services.today - row.due_date).days
            if row.due_date and row.due_date < services.today
            else 0
        )
        missing = missing_facts(
            body,
            invoice_number=row.invoice_number,
            amounts=(money(row.total), money(row.total - paid)),
            due_date=row.due_date,
            po_number=row.po_number,
        )
        vendor_row = VendorRepository(session).get(row.vendor_id) if row.vendor_id else None
        reminder = ReminderRepository(session).create(row.id, body, tone)
        return ReminderDraft(
            reminder_id=reminder.id,
            invoice_id=row.id,
            invoice_number=row.invoice_number,
            tone=tone,
            suggested_tone=tone_for(overdue),
            days_overdue=overdue,
            vendor_email=vendor_row.email if vendor_row else None,
            missing_facts=missing,
            body=body,
            created_at=reminder.created_at,
        )


def register(server: FastMCP, services: Services) -> None:
    @server.tool(name="draft_reminder", annotations=DRAFT)
    async def draft_reminder_tool(
        invoice: Annotated[str, Field(description="Invoice id, or the number printed on it")],
        body: Annotated[str, Field(description="The reminder text, as it would be sent")],
        tone: Annotated[
            Tone,
            Field(description="friendly (1-15 days overdue), firm (16-45), final_notice (46+)"),
        ],
        vendor: Annotated[
            str | None, Field(description="Narrow a number down to this vendor (id or name)")
        ] = None,
    ) -> ReminderDraft:
        """Save a payment reminder as a draft pending a person's approval. It is NEVER
        sent by this system. The body must state the invoice number, the amount, the
        original due date and the PO number; missing_facts lists any it leaves out."""
        return await in_thread(draft_reminder, services, invoice, body, tone, vendor)
