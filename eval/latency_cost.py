"""Latency and cost as reproducible numbers.

Latency: :func:`latency_summary` gives count, mean, p50, p95 and max of a list of
timings, and :func:`stage_latencies` does it per retrieval stage (``dense``, ``rerank``,
``graph``, ...) from :attr:`RetrievalResult.timings_ms`. Percentiles interpolate
linearly between the two nearest values, so p95 of 40 timings is not just the 38th.

Cost: every priced LLM call is a row in ``llm_calls`` (or a line of the ``cost_log``
JSONL), with the request id it was made under (``observability/cost.py``). Calls are
grouped by request id and each request is

* a **document** -- ``doc:<document id>``: classifying and extracting one ingested file;
* a **query** -- any other request id: a ``/chat`` request, or one evaluation question
  (``eval:<run>:<question>``);
* unattributed -- no request id (a script outside any request).

:func:`cost_report` gives, for documents and for queries, how many there were, their
calls and tokens, and the cost per item (mean, p50, p95, max) and in total; calls to a
model with no price count as $0 and are counted separately.

    python -m eval.latency_cost --results eval/reports/<date>/results.jsonl
    python -m eval.latency_cost --llm-calls db --since 2026-11-01
    python -m eval.latency_cost --llm-calls data/interim/llm_calls.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from ledgerlens.observability.cost import (
    DOCUMENT_REQUEST_PREFIX,
    LLMCallRecord,
    ModelCostSummary,
    summarize,
)


def percentile(values: Iterable[float], pct: float) -> float:
    """The ``pct`` percentile (0-100) by linear interpolation; 0.0 for no values."""
    if not 0 <= pct <= 100:
        raise ValueError(f"pct must be between 0 and 100, not {pct}")
    items = sorted(values)
    if not items:
        return 0.0
    position = (len(items) - 1) * pct / 100
    low = math.floor(position)
    high = min(low + 1, len(items) - 1)
    return items[low] + (items[high] - items[low]) * (position - low)


@dataclass(frozen=True)
class LatencySummary:
    """Timings in milliseconds."""

    n: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 1) if isinstance(v, float) else v for k, v in asdict(self).items()}


def latency_summary(values: Iterable[float | None]) -> LatencySummary:
    """Count, mean, p50, p95 and max of ``values`` (``None`` entries are skipped)."""
    items = [float(v) for v in values if v is not None]
    if not items:
        return LatencySummary(0, 0.0, 0.0, 0.0, 0.0)
    return LatencySummary(
        n=len(items),
        mean_ms=sum(items) / len(items),
        p50_ms=percentile(items, 50),
        p95_ms=percentile(items, 95),
        max_ms=max(items),
    )


def stage_latencies(timings: Iterable[Mapping[str, float]]) -> dict[str, LatencySummary]:
    """Per stage (``total`` last), the summary of that stage's timings across runs. Stages
    that recorded an error marker (``graph_error``) are not timings and are skipped."""
    by_stage: dict[str, list[float]] = {}
    for run in timings:
        for stage, value in run.items():
            if stage.endswith("_error") or not isinstance(value, int | float):
                continue
            by_stage.setdefault(stage, []).append(float(value))
    order = sorted(by_stage, key=lambda s: (s == "total", s))
    return {stage: latency_summary(by_stage[stage]) for stage in order}


def render_latency_table(rows: Mapping[str, LatencySummary]) -> list[str]:
    """Markdown rows: one per named summary."""
    lines = ["| | n | mean ms | p50 ms | p95 ms | max ms |", "|---|---:|---:|---:|---:|---:|"]
    for name, s in rows.items():
        lines.append(
            f"| {name} | {s.n} | {s.mean_ms:.0f} | {s.p50_ms:.0f} | {s.p95_ms:.0f} | "
            f"{s.max_ms:.0f} |"
        )
    return lines


# --------------------------------------------------------------------------- cost

RequestKind = Literal["document", "query"]


def request_kind(request_id: str | None) -> RequestKind | None:
    """What a call was for, from its request id (``None``: unattributed)."""
    if not request_id:
        return None
    return "document" if request_id.startswith(DOCUMENT_REQUEST_PREFIX) else "query"


@dataclass(frozen=True)
class CostGroup:
    """The documents (or the queries) behind a set of LLM calls."""

    kind: RequestKind
    n: int
    """Distinct requests (documents, queries) that made at least one call."""
    calls: int
    tokens_in: int
    tokens_out: int
    unpriced_calls: int
    total_usd: float
    mean_usd: float
    p50_usd: float
    p95_usd: float
    max_usd: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def cost_group(kind: RequestKind, records: Sequence[LLMCallRecord]) -> CostGroup:
    per_request: dict[str, float] = {}
    for r in records:
        key = r.request_id or ""
        per_request[key] = per_request.get(key, 0.0) + (r.cost_usd or 0.0)
    costs = list(per_request.values())
    total = sum(costs)
    return CostGroup(
        kind=kind,
        n=len(per_request),
        calls=len(records),
        tokens_in=sum(r.tokens_in for r in records),
        tokens_out=sum(r.tokens_out for r in records),
        unpriced_calls=sum(r.cost_usd is None for r in records),
        total_usd=round(total, 8),
        mean_usd=round(total / len(costs), 8) if costs else 0.0,
        p50_usd=round(percentile(costs, 50), 8),
        p95_usd=round(percentile(costs, 95), 8),
        max_usd=round(max(costs), 8) if costs else 0.0,
    )


@dataclass(frozen=True)
class CostReport:
    query: CostGroup
    document: CostGroup
    unattributed_calls: int
    unattributed_usd: float
    by_model: list[ModelCostSummary] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query.as_dict(),
            "document": self.document.as_dict(),
            "unattributed_calls": self.unattributed_calls,
            "unattributed_usd": self.unattributed_usd,
            "by_model": [m.model_dump() for m in self.by_model],
        }


def cost_report(records: Iterable[LLMCallRecord]) -> CostReport:
    """Cost per query and per document from LLM call records."""
    items = list(records)
    groups: dict[RequestKind | None, list[LLMCallRecord]] = {"query": [], "document": []}
    groups[None] = []
    for record in items:
        groups[request_kind(record.request_id)].append(record)
    loose = groups[None]
    return CostReport(
        query=cost_group("query", groups["query"]),
        document=cost_group("document", groups["document"]),
        unattributed_calls=len(loose),
        unattributed_usd=round(sum(r.cost_usd or 0.0 for r in loose), 8),
        by_model=summarize(items),
    )


def load_llm_calls(source: str, *, since: datetime | None = None) -> list[LLMCallRecord]:
    """LLM call records from ``db`` (the ``llm_calls`` table) or a ``cost_log`` JSONL
    path; ``since`` keeps calls at or after it."""
    if source == "db":
        from ledgerlens.observability.cost import PostgresCostSink

        return PostgresCostSink().read(since=since)
    from ledgerlens.observability.cost import read_jsonl

    records = read_jsonl(Path(source))
    return [r for r in records if since is None or r.ts >= since]


def render_cost_table(report: CostReport) -> list[str]:
    """Markdown: cost per query and per document, then per model."""
    lines = [
        "| | n | LLM calls | tokens in / out | total $ | mean $ | p50 $ | p95 $ | max $ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in (report.query, report.document):
        lines.append(
            f"| per {group.kind} | {group.n} | {group.calls} | "
            f"{group.tokens_in:,} / {group.tokens_out:,} | {group.total_usd:.4f} | "
            f"{group.mean_usd:.6f} | {group.p50_usd:.6f} | {group.p95_usd:.6f} | "
            f"{group.max_usd:.6f} |"
        )
    unpriced = report.query.unpriced_calls + report.document.unpriced_calls
    notes = []
    if unpriced:
        notes.append(f"{unpriced} calls had no price in models.yaml and count as $0.")
    if report.unattributed_calls:
        notes.append(
            f"{report.unattributed_calls} calls had no request id "
            f"(${report.unattributed_usd:.4f}) and are in neither row."
        )
    if notes:
        lines += ["", *notes]
    if report.by_model:
        lines += [
            "",
            "| provider/model | calls | tokens in / out | cost $ | mean latency ms |",
            "|---|---:|---:|---:|---:|",
        ]
        for m in report.by_model:
            cost = "unpriced" if m.cost_usd is None else f"{m.cost_usd:.4f}"
            lines.append(
                f"| {m.provider}/{m.model} | {m.calls} | {m.tokens_in:,} / {m.tokens_out:,} | "
                f"{cost} | {m.mean_latency_ms:.0f} |"
            )
    return lines


# --------------------------------------------------------------------------- CLI


def read_results(path: Path) -> list[dict[str, Any]]:
    """Per-question results (``eval.benchmark_retrieval --out``, ``eval.runner``)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Latency and cost of LedgerLens runs.")
    parser.add_argument("--results", type=Path, help="per-question JSONL: p50 / p95 latency")
    parser.add_argument(
        "--llm-calls", help="cost per query and per document: 'db' or a cost_log JSONL path"
    )
    parser.add_argument(
        "--since", type=datetime.fromisoformat, help="only calls from this time on (ISO 8601)"
    )
    args = parser.parse_args(argv)
    if not args.results and not args.llm_calls:
        parser.error("give --results, --llm-calls or both")
    if args.results:
        results = read_results(args.results)
        rows = {"retrieval": latency_summary(r.get("latency_ms") for r in results)}
        rows |= {
            f"stage {k}": v
            for k, v in stage_latencies(r.get("timings_ms") or {} for r in results).items()
        }
        print("\n".join(render_latency_table(rows)))
    if args.llm_calls:
        report = cost_report(load_llm_calls(args.llm_calls, since=args.since))
        print("\n".join(render_cost_table(report)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
