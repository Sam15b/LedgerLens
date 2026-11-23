"""``GET /eval/runs``: the recorded evaluation runs, for the dashboard's metrics and trend.

Each row of ``eval_runs`` is one run of the eval harness (``eval/runner.py``): the suite
it ran (``full``: the 200-question golden set; ``ci``: the 40-question subset the CI gate
runs), the git commit, the config, and the metrics it scored (``recall@5``, ``mrr``,
latency percentiles, cost, ...; the keys depend on what the run could measure).

Newest first; ``?suite=`` keeps one suite, ``limit`` caps the count. Runs of different
suites answer different question sets, so their numbers only compare within a suite.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

if TYPE_CHECKING:
    from sqlalchemy.orm import Session, sessionmaker

router = APIRouter(prefix="/eval", tags=["eval"])


class EvalRunOut(BaseModel):
    id: str
    suite: str
    git_sha: str | None = None
    config: dict[str, Any]
    metrics: dict[str, float | int | str | None]
    """Metric name -> value (``null`` when the run could not measure it)."""
    created_at: datetime


class EvalStore:
    """``eval_runs`` in Postgres."""

    def __init__(self, sessions: sessionmaker[Session] | None = None) -> None:
        self._sessions = sessions

    @property
    def sessions(self) -> sessionmaker[Session]:
        from ledgerlens.storage.postgres.db import get_session_factory

        return self._sessions or get_session_factory()

    def runs(self, suite: str | None, limit: int) -> list[EvalRunOut]:
        from sqlalchemy import select

        from ledgerlens.storage.postgres.models import EvalRunRow

        stmt = select(EvalRunRow)
        if suite is not None:
            stmt = stmt.where(EvalRunRow.suite == suite)
        stmt = stmt.order_by(EvalRunRow.created_at.desc(), EvalRunRow.id).limit(limit)
        with self.sessions() as session:
            return [
                EvalRunOut(
                    id=row.id,
                    suite=row.suite,
                    git_sha=row.git_sha,
                    config=dict(row.config or {}),
                    metrics=dict(row.metrics or {}),
                    created_at=row.created_at,
                )
                for row in session.scalars(stmt)
            ]


def get_eval_store() -> EvalStore:
    return EvalStore()


Store = Annotated[EvalStore, Depends(get_eval_store)]


@router.get("/runs", response_model=list[EvalRunOut])
async def list_runs(
    store: Store,
    suite: Annotated[str | None, Query(max_length=64)] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> list[EvalRunOut]:
    """Evaluation runs, newest first (one suite with ``?suite=``)."""
    return await run_in_threadpool(store.runs, suite, limit)
