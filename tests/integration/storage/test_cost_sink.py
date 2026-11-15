"""LLM call costs persisted to ``llm_calls`` and read back."""

from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session, sessionmaker

from ledgerlens.core.logging import request_context
from ledgerlens.observability.cost import CostLedger, LLMCallRecord, PostgresCostSink, summarize
from ledgerlens.storage.postgres.models import LLMCallRow

pytestmark = pytest.mark.integration


def record(cost: float | None, **changes: object) -> LLMCallRecord:
    data = {
        "alias": "extractor",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "tokens_in": 1200,
        "tokens_out": 300,
        "cost_usd": cost,
        "latency_ms": 812.5,
    }
    return LLMCallRecord.model_validate(data | changes)


def test_ledger_writes_every_call_to_llm_calls(sessions: sessionmaker[Session]) -> None:
    sink = PostgresCostSink(sessions)
    ledger = CostLedger([sink])

    with request_context("req-42"):
        ledger.record(record(0.00036))
    ledger.record(record(None, provider="gemini", model="gemini-3.8-flash", streamed=True))

    stored = sink.read()
    assert stored == ledger.records  # exact round trip, request id and timestamps included
    assert stored[0].request_id == "req-42"
    with sessions() as session:
        costs = [row.cost_usd for row in session.query(LLMCallRow).order_by(LLMCallRow.id)]
    assert [str(c) if c is not None else None for c in costs] == ["0.00036000", None]
    assert [s.cost_usd for s in summarize(stored)] == [0.00036, None]


def test_read_since_filters_by_timestamp(sessions: sessionmaker[Session]) -> None:
    sink = PostgresCostSink(sessions)
    sink.write(record(0.01, ts=datetime(2026, 11, 1, tzinfo=UTC)))
    sink.write(record(0.02, ts=datetime(2026, 11, 14, tzinfo=UTC)))

    recent = sink.read(since=datetime(2026, 11, 10, tzinfo=UTC))

    assert [r.cost_usd for r in recent] == [0.02]
