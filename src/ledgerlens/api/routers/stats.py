"""``GET /stats/overview``: the dashboard's headline numbers, in one request.

* **documents** -- how many uploads are in each pipeline status, and in all;
* **unpaid** -- per currency, what is still owed and how much of it is past due, as of
  ``as_of`` (default today), from the ``unpaid_totals`` analytics query (payments
  allocated by then are subtracted; void invoices are not counted);
* **flagged invoices** -- invoices whose newest anomaly score is a flag (``score >=
  0.25``, :mod:`ledgerlens.api.routers.anomalies`) that nobody has reviewed yet;
* **pending reminders** -- drafts waiting for a person to approve or reject them;
* **LLM cost per day** -- the last ``days`` days (UTC) of ``llm_calls``, zero-filled. The
  table is written only when ``LLM_COST_DB=true``; ``llm_costs_recorded`` says whether any
  call is recorded at all, so an empty chart is not read as "free".

Amounts are summed per currency, never converted.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ledgerlens.api.routers.anomalies import FLAG_SCORE

if TYPE_CHECKING:
    from sqlalchemy.orm import Session, sessionmaker

router = APIRouter(prefix="/stats", tags=["stats"])

DOCUMENT_STATUSES = ("queued", "processing", "ready", "failed")


class DocumentCounts(BaseModel):
    total: int
    queued: int = 0
    processing: int = 0
    ready: int = 0
    failed: int = 0


class UnpaidByCurrency(BaseModel):
    currency: str
    open_invoices: int
    outstanding: Decimal
    overdue_invoices: int
    overdue: Decimal


class DailyCost(BaseModel):
    day: date
    calls: int
    cost_usd: float
    """Priced calls only; a model with no price counts as $0."""


class Overview(BaseModel):
    as_of: date
    documents: DocumentCounts
    unpaid: list[UnpaidByCurrency]
    """Most open invoices first (amounts in different currencies do not compare)."""
    open_invoices: int
    overdue_invoices: int
    flagged_invoices: int
    pending_reminders: int
    llm_cost_by_day: list[DailyCost]
    """Oldest day first; the last entry is ``as_of``."""
    llm_costs_recorded: bool


def sum_unpaid(rows: list[tuple[str, int, Decimal, int, Decimal]]) -> list[UnpaidByCurrency]:
    """Vendor rows ``(currency, open, outstanding, overdue_n, overdue)`` summed per currency."""
    totals: dict[str, UnpaidByCurrency] = {}
    for currency, open_n, outstanding, overdue_n, overdue in rows:
        current = totals.get(currency)
        if current is None:
            totals[currency] = UnpaidByCurrency(
                currency=currency,
                open_invoices=open_n,
                outstanding=outstanding,
                overdue_invoices=overdue_n,
                overdue=overdue,
            )
        else:
            totals[currency] = current.model_copy(
                update={
                    "open_invoices": current.open_invoices + open_n,
                    "outstanding": current.outstanding + outstanding,
                    "overdue_invoices": current.overdue_invoices + overdue_n,
                    "overdue": current.overdue + overdue,
                }
            )
    return sorted(totals.values(), key=lambda u: (-u.open_invoices, u.currency))


def fill_days(found: dict[date, tuple[int, float]], last: date, days: int) -> list[DailyCost]:
    """One entry per day from ``last - days + 1`` to ``last``, zero where nothing ran."""
    first = last - timedelta(days=days - 1)
    out = []
    for n in range(days):
        day = first + timedelta(days=n)
        calls, cost = found.get(day, (0, 0.0))
        out.append(DailyCost(day=day, calls=calls, cost_usd=round(cost, 6)))
    return out


class StatsStore:
    """The overview's queries, in one read-only session."""

    def __init__(self, sessions: sessionmaker[Session] | None = None) -> None:
        self._sessions = sessions

    @property
    def sessions(self) -> sessionmaker[Session]:
        from ledgerlens.storage.postgres.db import get_session_factory

        return self._sessions or get_session_factory()

    def overview(self, as_of: date, days: int) -> Overview:
        from sqlalchemy import func, select
        from sqlalchemy.dialects.postgresql import distinct_on

        from ledgerlens.storage.postgres.analytics import unpaid_totals
        from ledgerlens.storage.postgres.models import (
            AnomalyScoreRow,
            DocumentRow,
            LLMCallRow,
            ReminderRow,
        )

        with self.sessions() as session:
            by_status = dict(
                session.execute(
                    select(DocumentRow.status, func.count()).group_by(DocumentRow.status)
                ).all()
            )
            counts = {s: int(by_status.get(s, 0)) for s in DOCUMENT_STATUSES}
            documents = DocumentCounts(total=sum(int(n) for n in by_status.values()), **counts)

            unpaid = sum_unpaid(
                [
                    (u.currency, u.open_invoices, u.outstanding, u.overdue_invoices, u.overdue)
                    for u in unpaid_totals(session, as_of)
                ]
            )

            newest = (
                select(AnomalyScoreRow.invoice_id, AnomalyScoreRow.score, AnomalyScoreRow.review)
                .ext(distinct_on(AnomalyScoreRow.invoice_id))
                .order_by(
                    AnomalyScoreRow.invoice_id,
                    AnomalyScoreRow.created_at.desc(),
                    AnomalyScoreRow.id.desc(),
                )
                .subquery()
            )
            flagged = session.scalar(
                select(func.count())
                .select_from(newest)
                .where(newest.c.score >= FLAG_SCORE, newest.c.review.is_(None))
            )

            pending = session.scalar(
                select(func.count())
                .select_from(ReminderRow)
                .where(ReminderRow.status == "pending_approval")
            )

            start = datetime.combine(as_of - timedelta(days=days - 1), time.min, tzinfo=UTC)
            end = datetime.combine(as_of + timedelta(days=1), time.min, tzinfo=UTC)
            day = func.date(func.timezone("UTC", LLMCallRow.ts))
            cost_rows = session.execute(
                select(day, func.count(), func.coalesce(func.sum(LLMCallRow.cost_usd), 0))
                .where(LLMCallRow.ts >= start, LLMCallRow.ts < end)
                .group_by(day)
            ).all()
            recorded = session.scalar(select(func.count()).select_from(LLMCallRow)) or 0

        return Overview(
            as_of=as_of,
            documents=documents,
            unpaid=unpaid,
            open_invoices=sum(u.open_invoices for u in unpaid),
            overdue_invoices=sum(u.overdue_invoices for u in unpaid),
            flagged_invoices=int(flagged or 0),
            pending_reminders=int(pending or 0),
            llm_cost_by_day=fill_days(
                {d: (int(n), float(cost)) for d, n, cost in cost_rows}, as_of, days
            ),
            llm_costs_recorded=recorded > 0,
        )


def get_stats_store() -> StatsStore:
    return StatsStore()


Store = Annotated[StatsStore, Depends(get_stats_store)]


@router.get("/overview", response_model=Overview)
async def overview(
    store: Store,
    as_of: date | None = None,
    days: Annotated[int, Query(ge=1, le=90)] = 7,
) -> Overview:
    """Documents by status, unpaid and overdue totals per currency, unreviewed anomaly
    flags, reminders awaiting approval, and LLM cost per day."""
    return await run_in_threadpool(store.overview, as_of or date.today(), days)
