"""Per-call LLM cost and latency ledger.

Every call the model router completes is recorded as an :class:`LLMCallRecord`: alias,
provider, model, tokens, USD cost and latency, plus the request id from the logging
context so a call can be traced back to the API request or job that made it.

The :class:`CostLedger` keeps recent records in memory (for summaries such as the provider
comparison) and forwards each one to its sinks. :class:`JsonlCostSink` appends one JSON
object per line to a file; a Postgres ``llm_calls`` sink joins it in Phase 3. A failing
sink is logged and skipped: recording cost must never break the call it describes.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from ledgerlens.core.config import ModelsConfig, repo_path
from ledgerlens.core.logging import get_request_id

logger = logging.getLogger(__name__)

DEFAULT_MAX_RECORDS = 10_000


class LLMCallRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    request_id: str | None = Field(default_factory=get_request_id)
    alias: str
    provider: str
    model: str
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    cost_usd: float | None
    """``None`` when the model has no price in ``models.yaml``."""
    latency_ms: float = Field(ge=0)
    streamed: bool = False


class ModelCostSummary(BaseModel):
    """Totals for one ``provider/model`` across the recorded calls."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    calls: int
    tokens_in: int
    tokens_out: int
    cost_usd: float | None
    """Sum of the priced calls; ``None`` if none of the calls had a price."""
    unpriced_calls: int
    mean_latency_ms: float


class CostSink(Protocol):
    def write(self, record: LLMCallRecord) -> None: ...


class JsonlCostSink:
    """Append each record as one JSON line to ``path`` (directories are created)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def write(self, record: LLMCallRecord) -> None:
        line = record.model_dump_json() + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)


def read_jsonl(path: Path) -> list[LLMCallRecord]:
    """Load the records a :class:`JsonlCostSink` wrote."""
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [LLMCallRecord.model_validate_json(line) for line in lines if line.strip()]


class CostLedger:
    def __init__(
        self, sinks: Sequence[CostSink] = (), *, max_records: int = DEFAULT_MAX_RECORDS
    ) -> None:
        self.sinks = list(sinks)
        self._records: deque[LLMCallRecord] = deque(maxlen=max_records)
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, config: ModelsConfig) -> CostLedger:
        """A ledger with a JSONL sink at ``cost_log`` (relative to the repo), if set."""
        if config.cost_log is None:
            return cls()
        return cls([JsonlCostSink(repo_path(config.cost_log))])

    def record(self, record: LLMCallRecord) -> None:
        with self._lock:
            self._records.append(record)
        for sink in self.sinks:
            try:
                sink.write(record)
            except Exception:  # a broken sink must not fail the LLM call it records
                logger.exception(
                    "cost sink %s failed; record kept in memory only", type(sink).__name__
                )

    @property
    def records(self) -> list[LLMCallRecord]:
        with self._lock:
            return list(self._records)

    def total_cost_usd(self) -> float:
        """Sum of all priced calls (unpriced calls count as zero)."""
        return round(sum(r.cost_usd or 0.0 for r in self.records), 8)

    def summary(self) -> list[ModelCostSummary]:
        return summarize(self.records)


def summarize(records: Iterable[LLMCallRecord]) -> list[ModelCostSummary]:
    """Per ``provider/model`` totals, in order of first appearance."""
    groups: dict[tuple[str, str], list[LLMCallRecord]] = {}
    for record in records:
        groups.setdefault((record.provider, record.model), []).append(record)
    summaries = []
    for (provider, model), calls in groups.items():
        priced = [c.cost_usd for c in calls if c.cost_usd is not None]
        summaries.append(
            ModelCostSummary(
                provider=provider,
                model=model,
                calls=len(calls),
                tokens_in=sum(c.tokens_in for c in calls),
                tokens_out=sum(c.tokens_out for c in calls),
                cost_usd=round(sum(priced), 8) if priced else None,
                unpriced_calls=len(calls) - len(priced),
                mean_latency_ms=round(sum(c.latency_ms for c in calls) / len(calls), 2),
            )
        )
    return summaries
