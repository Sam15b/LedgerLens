"""``score_anomaly``: score a stored invoice for anomalies, with reasons.

The same scorer ingestion runs (:mod:`ledgerlens.anomaly.scorer`): every rule against
what was stored before the invoice, the autoencoder on its features, one combined score
and plain-language reasons. The score is written to ``anomaly_scores`` under the model
version (replacing an earlier score from that version), so the dashboard shows the same
verdict the agent was given.
"""

from __future__ import annotations

from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from ledgerlens.mcp_server.services import Services
from ledgerlens.mcp_server.tools import in_thread
from ledgerlens.mcp_server.tools.records import citation, find_invoice

SCORE = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False
)
"""Writes the invoice's score row; scoring again gives the same row."""


class AnomalyReasonOut(BaseModel):
    source: Literal["rule", "autoencoder", "summary"]
    code: str
    text: str
    share: float | None = None
    related_ids: list[str] = Field(default_factory=list)


class AnomalyResult(BaseModel):
    invoice_id: str
    invoice_number: str
    citation: str | None
    model_version: str
    """The autoencoder version, or ``rules-only``."""
    score: float
    """0..1: rule hits 0.5 and up, autoencoder-only flags 0.25 and up."""
    flagged: bool
    hard_hit: bool
    """A rule fired (duplicate, over PO, contract rate, unknown vendor)."""
    reasons: list[AnomalyReasonOut] = Field(default_factory=list)


def score_anomaly(services: Services, invoice: str, vendor: str | None = None) -> AnomalyResult:
    from ledgerlens.anomaly.scorer import score_stored
    from ledgerlens.storage.postgres.db import session_scope

    scorer = services.scorer
    with session_scope(services.sessions) as session:
        row = find_invoice(session, services, invoice, vendor)
        verdict = score_stored(session, row.id, scorer)
        if verdict is None:  # pragma: no cover - found just above
            raise ToolError(f"no invoice {invoice!r}")
        return AnomalyResult(
            invoice_id=row.id,
            invoice_number=row.invoice_number,
            citation=citation(row.invoice_number, row.document_id),
            model_version=scorer.model_version,
            score=round(verdict.score, 4),
            flagged=verdict.flagged,
            hard_hit=verdict.hard_hit,
            reasons=[AnomalyReasonOut(**r.model_dump()) for r in verdict.reasons],
        )


def register(server: FastMCP, services: Services) -> None:
    @server.tool(name="score_anomaly", annotations=SCORE)
    async def score_anomaly_tool(
        invoice: Annotated[str, Field(description="Invoice id, or the number printed on it")],
        vendor: Annotated[
            str | None, Field(description="Narrow a number down to this vendor (id or name)")
        ] = None,
    ) -> AnomalyResult:
        """How unusual an invoice is: duplicate / over-PO / contract-rate / unknown-vendor
        rules plus the autoencoder, as one 0..1 score with plain-language reasons."""
        return await in_thread(score_anomaly, services, invoice, vendor)
