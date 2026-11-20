"""Reminder drafts and a person's decision on them. Nothing is ever sent.

* ``GET /reminders`` -- drafts, newest first (``?status=pending_approval`` by default;
  ``?status=all`` for every one), optionally for one invoice;
* ``GET /reminders/{id}`` -- one draft;
* ``POST /reminders/{id}/approve`` and ``.../reject`` -- record the decision (the draft
  must still be ``pending_approval``: 409 otherwise), then resume the agent run that
  drafted it (``thread_id``) if it is still waiting for that draft. The response says
  whether the run was resumed and what it answered.

Approving only marks a draft ``approved``. LedgerLens has no send path: the person sends
the approved text themselves.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from langchain_core.messages import AIMessage
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session, sessionmaker
from starlette.concurrency import run_in_threadpool

from ledgerlens.api.agent import AgentProvider, AgentUnavailableError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reminders", tags=["reminders"])

Decision = Literal["approved", "rejected"]
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class ReminderOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    invoice_id: str
    tone: str
    status: str
    draft: str
    thread_id: str | None = None
    created_at: datetime
    decided_at: datetime | None = None


class DecisionOut(BaseModel):
    reminder: ReminderOut
    resumed: bool
    """The agent run that drafted it was waiting, and has been resumed."""
    thread_status: Literal["completed", "interrupted"] | None = None
    """After resuming: ``interrupted`` while other drafts of the run are undecided."""
    reply: str | None = None
    """What the resumed run answered."""
    note: str | None = None
    """Why the run was not resumed."""


def get_sessions() -> sessionmaker[Session]:
    from ledgerlens.storage.postgres.db import get_session_factory

    return get_session_factory()


Sessions = Annotated[sessionmaker[Session], Depends(get_sessions)]


def _checked_id(reminder_id: str) -> str:
    if not _UUID.match(reminder_id.lower()):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no reminder {reminder_id!r}")
    return reminder_id.lower()


@router.get("", response_model=list[ReminderOut])
def list_reminders(
    sessions: Sessions,
    status_filter: Annotated[
        Literal["pending_approval", "approved", "rejected", "all"], Query(alias="status")
    ] = "pending_approval",
    invoice_id: str | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[ReminderOut]:
    from ledgerlens.storage.postgres.repositories import ReminderRepository

    with sessions() as session:
        rows = ReminderRepository(session).find(
            status=None if status_filter == "all" else status_filter,
            invoice_id=invoice_id,
            limit=limit,
        )
        return [ReminderOut.model_validate(row) for row in rows]


@router.get("/{reminder_id}", response_model=ReminderOut)
def get_reminder(reminder_id: str, sessions: Sessions) -> ReminderOut:
    from ledgerlens.storage.postgres.repositories import ReminderRepository

    with sessions() as session:
        row = ReminderRepository(session).get(_checked_id(reminder_id))
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"no reminder {reminder_id!r}")
        return ReminderOut.model_validate(row)


def record_decision(
    sessions: sessionmaker[Session], reminder_id: str, decision: Decision
) -> ReminderOut:
    from ledgerlens.storage.postgres.db import session_scope
    from ledgerlens.storage.postgres.repositories import ReminderRepository

    try:
        with session_scope(sessions) as session:
            return ReminderOut.model_validate(
                ReminderRepository(session).decide(reminder_id, decision)
            )
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no reminder {reminder_id!r}") from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


def _waiting_for(state: Any) -> set[str]:
    """The reminder ids a paused run is waiting on."""
    ids: set[str] = set()
    for pause in getattr(state, "interrupts", ()) or ():
        value = pause.value if isinstance(pause.value, dict) else {}
        ids.update(str(r.get("reminder_id")) for r in value.get("reminders", []))
    return ids


async def resume_run(
    agents: AgentProvider, reminder: ReminderOut, decision: Decision
) -> DecisionOut:
    if not reminder.thread_id:
        return DecisionOut(reminder=reminder, resumed=False, note="drafted outside an agent run")
    try:
        runtime = await agents.get()
    except AgentUnavailableError as exc:
        return DecisionOut(
            reminder=reminder, resumed=False, note=f"decision recorded; agents unavailable: {exc}"
        )
    config = {"configurable": {"thread_id": reminder.thread_id}}
    state = await runtime.graph.aget_state(config)
    if reminder.id not in _waiting_for(state):
        return DecisionOut(
            reminder=reminder, resumed=False, note="the run is no longer waiting for this draft"
        )
    result = await runtime.graph.ainvoke(Command(resume={reminder.id: decision}), config)
    after = await runtime.graph.aget_state(config)
    replies = [m for m in result.get("messages", []) if isinstance(m, AIMessage) and m.text]
    return DecisionOut(
        reminder=reminder,
        resumed=True,
        thread_status="interrupted" if after.next else "completed",
        reply=replies[-1].text if replies else None,
    )


async def decide(
    request: Request, reminder_id: str, decision: Decision, sessions: sessionmaker[Session]
) -> DecisionOut:
    reminder = await run_in_threadpool(
        record_decision, sessions, _checked_id(reminder_id), decision
    )
    logger.info("reminder %s %s", reminder.id, decision)
    return await resume_run(request.app.state.agents, reminder, decision)


@router.post("/{reminder_id}/approve", response_model=DecisionOut)
async def approve(request: Request, reminder_id: str, sessions: Sessions) -> DecisionOut:
    """Approve a pending draft (it is not sent: send the text yourself) and resume its run."""
    return await decide(request, reminder_id, "approved", sessions)


@router.post("/{reminder_id}/reject", response_model=DecisionOut)
async def reject(request: Request, reminder_id: str, sessions: Sessions) -> DecisionOut:
    """Reject a pending draft and resume its run."""
    return await decide(request, reminder_id, "rejected", sessions)
