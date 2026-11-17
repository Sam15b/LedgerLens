"""GraphRAG retrievers.

**R4** :class:`GraphRAGRetriever` -- vector search finds the seeds, the graph supplies the
relationships (docs/layers/03-retrieval.md, "GraphRAG algorithm")::

    question -> analyze -> filters -> R3 (hybrid + rerank) top k chunks
                                         |
                       seed document_ids + the question's document numbers
                                         |
             graph template chosen from the analysis (retrieval/graph.py) -> facts

The result carries R3's chunks *and* the graph facts; :mod:`ledgerlens.retrieval.context`
fuses them into one cited context. Stages are timed as ``analyze``, R3's own stages and
``graph``.

The graph is an addition, never a dependency: if Neo4j fails, R4 logs it and returns R3's
chunks (``timings_ms["graph_error"]`` marks it), so an answer can still be written.

**R5** :class:`Text2CypherRetriever` -- R4, plus LLM-written Cypher
(:mod:`ledgerlens.retrieval.text2cypher`) for questions the analysis marks as an
aggregation or a relationship question. Its rows come first among the facts, after a
line saying which query produced them. A query that is rejected or fails is regenerated
once with the error message; if that fails too (or the LLM is unavailable), R5 falls back
to the deterministic path -- R3's chunks plus the template facts -- and marks
``timings_ms["text2cypher_error"]``. Other questions cost no LLM call: R5 answers them
exactly as R4 does.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import date
from typing import Any

from ledgerlens.retrieval import (
    GraphFact,
    QueryFilters,
    RetrievalResult,
    Retriever,
    ScoredChunk,
    StageTimer,
)
from ledgerlens.retrieval.analyze import QueryAnalysis, QueryAnalyzer
from ledgerlens.retrieval.graph import GraphExpander
from ledgerlens.retrieval.text2cypher import CypherGenerator, CypherGuard, CypherRun

logger = logging.getLogger(__name__)


class GraphRAGRetriever:
    """R4: a vector retriever's chunks (R3's, normally) expanded through the graph."""

    name = "r4_graphrag"

    def __init__(
        self,
        base: Retriever,
        expander: GraphExpander,
        *,
        analyzer: QueryAnalyzer | None = None,
    ) -> None:
        self.base = base
        self.expander = expander
        self.analyzer = analyzer

    def analyze(self, query: str) -> QueryAnalysis | None:
        return self.analyzer.analyze(query) if self.analyzer is not None else None

    def retrieve(
        self, query: str, *, filters: QueryFilters | None = None, k: int = 5
    ) -> RetrievalResult:
        """``filters`` given by the caller win; otherwise the analysis supplies them."""
        timer = StageTimer()
        with timer.stage("analyze"):
            analysis = self.analyze(query)
        chunks, facts = self.expand(query, analysis, timer, filters=filters, k=k)
        return RetrievalResult.build(query, self.name, chunks, facts, timings_ms=timer.done())

    def expand(
        self,
        query: str,
        analysis: QueryAnalysis | None,
        timer: StageTimer,
        *,
        filters: QueryFilters | None = None,
        k: int = 5,
    ) -> tuple[list[ScoredChunk], list[GraphFact]]:
        """R3's chunks and the template facts around them, timed into ``timer``."""
        if filters is None and analysis is not None:
            filters = analysis.filters()
        seeds = self.base.retrieve(query, filters=filters, k=k)
        timer.timings_ms.update({n: ms for n, ms in seeds.timings_ms.items() if n != "total"})
        with timer.stage("graph"):
            try:
                facts = self.expander.facts(analysis, seeds.document_ids)
            except Exception:  # Neo4j down or a bad query: answer from the chunks alone
                logger.warning("graph expansion failed; using vector results only", exc_info=True)
                facts = []
                timer.timings_ms["graph_error"] = 1.0
        return list(seeds.chunks), facts


# --------------------------------------------------------------------------- R5

NUMBER_COLUMNS = ("invoice_number", "po_number", "payment_reference", "contract_number")
"""Columns whose value, next to a ``document_id`` column, makes a row citable."""


def _value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_value(item) for item in value) + "]"
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{k}: {_value(v)}" for k, v in value.items()) + "}"
    return "null" if value is None else str(value)


def cypher_facts(run: CypherRun) -> list[GraphFact]:
    """A header fact naming the query and its row count, then one fact per row
    (``column: value, ...``). A row with a ``document_id`` and a document number column is
    citable as ``doc:<number>``."""
    one_line = " ".join(run.cypher.split())
    header = GraphFact(
        text=f"Graph query ({len(run.rows)} rows): {one_line}",
        source="text2cypher",
        subject=f"text2cypher:{one_line}",
        data={"cypher": run.cypher, "rows": len(run.rows)},
    )
    facts = [header]
    for n, row in enumerate(run.rows):
        shown = {k: v for k, v in row.items() if k != "document_id"}
        text = ", ".join(f"{column}: {_value(value)}" for column, value in shown.items())
        number = next((row[c] for c in NUMBER_COLUMNS if row.get(c)), None)
        document_id = row.get("document_id")
        citations = {f"doc:{number}": document_id} if number and document_id else {}
        facts.append(
            GraphFact(
                text=f"Row {n + 1}: {text}",
                document_ids=(document_id,) if document_id else (),
                source="text2cypher",
                citations=citations,
                subject=f"text2cypher:{one_line}#{n}",
                data=dict(row),
            )
        )
    return facts


class Text2CypherRetriever:
    """R5: R4, plus LLM-generated Cypher for aggregation and relationship questions."""

    name = "r5_text2cypher"

    def __init__(
        self,
        graphrag: GraphRAGRetriever,
        generator: CypherGenerator,
        guard: CypherGuard,
        *,
        retries: int = 1,
    ) -> None:
        self.graphrag = graphrag
        self.generator = generator
        self.guard = guard
        self.retries = retries

    @staticmethod
    def wants_cypher(analysis: QueryAnalysis | None) -> bool:
        return analysis is not None and (analysis.aggregation or analysis.needs_graph)

    def query_graph(self, question: str, analysis: QueryAnalysis | None) -> CypherRun:
        """Generate and run Cypher, regenerating up to ``retries`` times with the error.
        Raises the last error when every attempt fails."""
        previous: str | None = None
        error: str | None = None
        for attempt in range(self.retries + 1):
            cypher = self.generator.generate(question, analysis, previous=previous, error=error)
            try:
                return self.guard.run(cypher)
            except Exception as exc:  # rejected, syntax error, timeout, ...
                if attempt == self.retries:
                    raise
                previous, error = cypher, f"{type(exc).__name__}: {exc}"
                logger.info("generated Cypher failed (%s); retrying with the error", error)
        raise AssertionError("unreachable")

    def retrieve(
        self, query: str, *, filters: QueryFilters | None = None, k: int = 5
    ) -> RetrievalResult:
        timer = StageTimer()
        with timer.stage("analyze"):
            analysis = self.graphrag.analyze(query)
        chunks, facts = self.graphrag.expand(query, analysis, timer, filters=filters, k=k)
        if self.wants_cypher(analysis):
            with timer.stage("text2cypher"):
                try:
                    facts = [*cypher_facts(self.query_graph(query, analysis)), *facts]
                except Exception:
                    logger.warning(
                        "Text2Cypher failed after %d attempt(s); using GraphRAG results",
                        self.retries + 1,
                        exc_info=True,
                    )
                    timer.timings_ms["text2cypher_error"] = 1.0
        return RetrievalResult.build(query, self.name, chunks, facts, timings_ms=timer.done())
