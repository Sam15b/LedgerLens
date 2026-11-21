"""Anomaly flags and a person's review of them.

* ``GET /anomalies`` -- flagged invoices, highest score first, with the reasons the scorer
  gave. One score per invoice (``model_version``'s, else its newest). By default only
  **flags** (``score >= 0.25``: a rule hit scores 0.5 or more, the autoencoder alone flags
  from 0.25, see :mod:`ledgerlens.anomaly.scorer`) that nobody has reviewed yet;
  ``?review=confirmed|false_positive|reviewed|all`` shows the others, ``?min_score=0``
  every score.
* ``POST /anomalies/{id}/review`` -- ``{"verdict": "confirmed" | "false_positive",
  "note": "..."}``. Stored on the score (``anomaly_scores.review``) as a label for
  retraining the autoencoder; a later review of the same score replaces it.

``id`` is the score's id (``anomaly_scores.id``), not the invoice's.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

if TYPE_CHECKING:
    from sqlalchemy.orm import Session, sessionmaker

    from ledgerlens.storage.postgres.models import AnomalyScoreRow, InvoiceRow

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/anomalies", tags=["anomalies"])

FLAG_SCORE = 0.25
"""The lowest flagged score (:func:`ledgerlens.anomaly.scorer.combine`)."""

Verdict = Literal["confirmed", "false_positive"]
ReviewFilter = Literal["unreviewed", "reviewed", "confirmed", "false_positive", "all"]


class AnomalyOut(BaseModel):
    id: int
    """The score's id: review it with ``POST /anomalies/{id}/review``."""
    invoice_id: str
    invoice_number: str
    document_id: str | None = None
    vendor_id: str | None = None
    vendor_name: str
    invoice_date: date
    currency: str
    total: Decimal
    model_version: str
    score: float
    flagged: bool
    reasons: list[dict[str, Any]]
    review: Verdict | None = None
    review_note: str | None = None
    reviewed_at: datetime | None = None
    scored_at: datetime

    @classmethod
    def of(cls, score: AnomalyScoreRow, invoice: InvoiceRow) -> AnomalyOut:
        return cls(
            id=score.id,
            invoice_id=invoice.id,
            invoice_number=invoice.invoice_number,
            document_id=invoice.document_id,
            vendor_id=invoice.vendor_id,
            vendor_name=invoice.vendor_name,
            invoice_date=invoice.invoice_date,
            currency=invoice.currency,
            total=invoice.total,
            model_version=score.model_version,
            score=score.score,
            flagged=score.score >= FLAG_SCORE,
            reasons=list(score.reasons),
            review=score.review,  # type: ignore[arg-type]  # CHECKed in the database
            review_note=score.review_note,
            reviewed_at=score.reviewed_at,
            scored_at=score.created_at,
        )


class ReviewIn(BaseModel):
    verdict: Verdict
    note: str | None = Field(default=None, max_length=2000)


class AnomalyStore:
    """Anomaly scores and reviews in Postgres."""

    def __init__(self, sessions: sessionmaker[Session] | None = None) -> None:
        self._sessions = sessions

    @property
    def sessions(self) -> sessionmaker[Session]:
        from ledgerlens.storage.postgres.db import get_session_factory

        return self._sessions or get_session_factory()

    def list(
        self,
        *,
        review: ReviewFilter = "unreviewed",
        min_score: float = FLAG_SCORE,
        vendor_id: str | None = None,
        model_version: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AnomalyOut]:
        from ledgerlens.storage.postgres.repositories import AnomalyScoreRepository

        verdict = review if review in ("confirmed", "false_positive") else None
        reviewed = {"unreviewed": False, "reviewed": True}.get(review)
        with self.sessions() as session:
            rows = AnomalyScoreRepository(session).flags(
                min_score=min_score,
                review=verdict,
                reviewed=reviewed,
                vendor_id=vendor_id,
                model_version=model_version,
                limit=limit,
                offset=offset,
            )
            return [AnomalyOut.of(score, invoice) for score, invoice in rows]

    def review(self, score_id: int, verdict: Verdict, note: str | None) -> AnomalyOut | None:
        """The reviewed score, or ``None`` if there is no such score."""
        from ledgerlens.storage.postgres.db import session_scope
        from ledgerlens.storage.postgres.models import InvoiceRow
        from ledgerlens.storage.postgres.repositories import AnomalyScoreRepository

        with session_scope(self.sessions) as session:
            try:
                score = AnomalyScoreRepository(session).review(score_id, verdict, note)
            except KeyError:
                return None
            invoice = session.get(InvoiceRow, score.invoice_id)
            assert invoice is not None  # the foreign key cascades
            return AnomalyOut.of(score, invoice)


def get_anomaly_store() -> AnomalyStore:
    return AnomalyStore()


Store = Annotated[AnomalyStore, Depends(get_anomaly_store)]


@router.get("", response_model=list[AnomalyOut])
async def list_anomalies(
    store: Store,
    review: ReviewFilter = "unreviewed",
    min_score: Annotated[float, Query(ge=0, le=1)] = FLAG_SCORE,
    vendor_id: str | None = None,
    model_version: str | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[AnomalyOut]:
    """Flagged invoices to review, highest score first."""
    return await run_in_threadpool(
        lambda: store.list(
            review=review,
            min_score=min_score,
            vendor_id=vendor_id,
            model_version=model_version,
            limit=limit,
            offset=offset,
        )
    )


@router.post("/{score_id}/review", response_model=AnomalyOut)
async def review_anomaly(score_id: int, body: ReviewIn, store: Store) -> AnomalyOut:
    """Mark a flag ``confirmed`` or ``false_positive`` (kept for retraining)."""
    reviewed = await run_in_threadpool(store.review, score_id, body.verdict, body.note)
    if reviewed is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no anomaly score {score_id}")
    logger.info("anomaly %s reviewed: %s", score_id, body.verdict)
    return reviewed
