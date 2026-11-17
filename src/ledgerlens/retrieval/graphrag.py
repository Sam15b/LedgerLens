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
"""

from __future__ import annotations

import logging

from ledgerlens.retrieval import QueryFilters, RetrievalResult, Retriever, StageTimer
from ledgerlens.retrieval.analyze import QueryAnalysis, QueryAnalyzer
from ledgerlens.retrieval.graph import GraphExpander

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
        return RetrievalResult.build(
            query, self.name, seeds.chunks, graph_facts=facts, timings_ms=timer.done()
        )
