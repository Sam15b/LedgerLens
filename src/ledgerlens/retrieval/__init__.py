"""Retrieval: find the evidence for a question (docs/layers/03-retrieval.md).

Every retriever variant -- R1 dense, R2 hybrid, R3 hybrid + rerank, R4 GraphRAG, R5
GraphRAG + Text2Cypher -- implements :class:`Retriever`, so the benchmark
(``eval/benchmark_retrieval.py``) and the agents can swap one for another:

    result = retriever.retrieve("What's the total on INV-889?", filters=filters, k=5)
    for chunk in result.chunks:
        print(chunk.citation, chunk.score, chunk.text)

* :class:`QueryFilters` -- structured constraints pulled out of the question (document
  type, vendor, document number, date range); retrievers turn them into store filters.
* :class:`ScoredChunk` -- a retrieved chunk with its full text and why it matched.
* :class:`GraphFact` -- a fact read from the knowledge graph (R4/R5), rendered as text.
* :class:`RetrievalResult` -- chunks, graph facts, citations and per-stage timings
  (recorded with :class:`StageTimer`).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ledgerlens.core.schemas import DocType


class QueryFilters(BaseModel):
    """Constraints on which documents may answer a question. Empty means "any".

    Values of one field are alternatives (any vendor in ``vendor_ids``); different fields
    must all hold. Dates are inclusive and apply to the document's own date (invoice
    date, order date, payment date, contract start).
    """

    model_config = ConfigDict(frozen=True)

    doc_types: tuple[DocType, ...] = ()
    vendor_ids: tuple[str, ...] = ()
    doc_numbers: tuple[str, ...] = ()
    date_from: date | None = None
    date_to: date | None = None

    @model_validator(mode="after")
    def _ordered_dates(self) -> QueryFilters:
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError(f"date_from {self.date_from} is after date_to {self.date_to}")
        return self

    def is_empty(self) -> bool:
        return not (
            self.doc_types or self.vendor_ids or self.doc_numbers or self.date_from or self.date_to
        )


class ScoredChunk(BaseModel):
    """One retrieved chunk. ``score`` is the retriever's final score (higher is better):
    the vector similarity for R1/R2, the cross-encoder score once reranked."""

    model_config = ConfigDict(frozen=True)

    id: str
    """The Pinecone vector id, ``{document_id}#c{n}`` (= ``chunks.id`` in Postgres)."""
    document_id: str
    text: str
    """Full chunk text from Postgres (the truncated copy from Pinecone if unavailable)."""
    score: float
    doc_type: str | None = None
    doc_number: str | None = None
    vendor_id: str | None = None
    page: int | None = None
    chunk_index: int | None = None
    scores: dict[str, float] = Field(default_factory=dict)
    """Every score the chunk received on the way, by stage (``dense``, ``rerank``, ...)."""

    @property
    def citation(self) -> str:
        """The tag an answer cites this chunk by: ``doc:<number>``, else ``doc:<id prefix>``."""
        return f"doc:{self.doc_number or self.document_id[:8]}"


class GraphFact(BaseModel):
    """A fact from the knowledge graph, e.g. ``Invoice INV-889 (Acme, 4,200.00 GBP, due
    2026-04-13) -> PO-1042 -> no payment``."""

    model_config = ConfigDict(frozen=True)

    text: str
    document_ids: tuple[str, ...] = ()
    """Documents the fact is about, for citations."""
    source: str = "graph"
    """Which query produced it (a template name, or ``text2cypher``)."""
    citations: dict[str, str] = Field(default_factory=dict)
    """Citation tag (``doc:INV-889``) -> document id, for the documents the fact names that
    have one (records loaded without a document have nothing to cite)."""
    data: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """What a retriever found, best first."""

    query: str
    retriever: str
    chunks: list[ScoredChunk] = Field(default_factory=list)
    graph_facts: list[GraphFact] = Field(default_factory=list)
    citations: dict[str, str] = Field(default_factory=dict)
    """Citation tag -> document id, for every chunk and fact."""
    timings_ms: dict[str, float] = Field(default_factory=dict)

    @classmethod
    def build(
        cls,
        query: str,
        retriever: str,
        chunks: list[ScoredChunk],
        graph_facts: list[GraphFact] | None = None,
        timings_ms: dict[str, float] | None = None,
    ) -> RetrievalResult:
        """A result with ``citations`` filled in from the chunks and facts."""
        facts = graph_facts or []
        citations = {chunk.citation: chunk.document_id for chunk in chunks}
        for fact in facts:
            for tag, document_id in fact.citations.items():
                citations.setdefault(tag, document_id)
            cited = set(fact.citations.values())
            for document_id in fact.document_ids:
                if document_id not in cited:
                    citations.setdefault(f"doc:{document_id[:8]}", document_id)
        return cls(
            query=query,
            retriever=retriever,
            chunks=chunks,
            graph_facts=facts,
            citations=citations,
            timings_ms=timings_ms or {},
        )

    @property
    def document_ids(self) -> list[str]:
        """Documents of the chunks, best first, each once (the seeds for graph expansion)."""
        return list(dict.fromkeys(chunk.document_id for chunk in self.chunks))


class StageTimer:
    """Wall time per retrieval stage, in milliseconds, plus ``total`` since creation::

        timer = StageTimer()
        with timer.stage("search"):
            hits = ...
        result = RetrievalResult.build(..., timings_ms=timer.done())

    A stage entered twice adds up.
    """

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.timings_ms: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            self.timings_ms[name] = self.timings_ms.get(name, 0.0) + elapsed

    def done(self) -> dict[str, float]:
        """The stage timings with ``total`` added."""
        return self.timings_ms | {"total": (time.perf_counter() - self.started) * 1000}


@runtime_checkable
class Retriever(Protocol):
    """Every retriever variant: one question in, the best ``k`` chunks (and facts) out."""

    name: str

    def retrieve(
        self, query: str, *, filters: QueryFilters | None = None, k: int = 5
    ) -> RetrievalResult: ...


__all__ = [
    "GraphFact",
    "QueryFilters",
    "RetrievalResult",
    "Retriever",
    "ScoredChunk",
    "StageTimer",
]
