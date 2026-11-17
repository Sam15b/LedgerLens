"""Benchmark retriever variants on the golden set: recall@k, MRR, latency, cost.

Every retriever returns chunks (best first) and, for R4/R5, graph facts. The documents
it *retrieved* are read off that result in rank order -- the chunks' documents, then the
documents the facts cite -- each once (:func:`retrieved_documents`). They are scored
against the question's ``relevant_document_ids``:

* **recall@k** -- the share of relevant documents among the first ``k`` retrieved;
* **MRR** -- 1 / the rank of the first relevant document (0 if none is retrieved);
* **evidence recall** -- the share of relevant documents anywhere in the result, facts
  included: what the answer step gets to see.

Unanswerable questions (no relevant documents) have no recall; they are counted, not
averaged.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ledgerlens.retrieval import RetrievalResult


def unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def retrieved_documents(result: RetrievalResult) -> list[str]:
    """Documents in rank order: chunk documents, then the documents graph facts cite."""
    chunk_documents = (chunk.document_id for chunk in result.chunks)
    fact_documents = (doc for fact in result.graph_facts for doc in fact.document_ids)
    return unique([*chunk_documents, *fact_documents])


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Share of ``relevant`` among the first ``k`` of ``retrieved`` (1.0 when nothing is
    relevant: there was nothing to miss)."""
    if k < 1:
        raise ValueError("k must be at least 1")
    wanted = set(relevant)
    if not wanted:
        return 1.0
    return len(wanted & set(unique(retrieved)[:k])) / len(wanted)


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """1 / rank of the first relevant document in ``retrieved`` (1-based), else 0."""
    wanted = set(relevant)
    for rank, document in enumerate(unique(retrieved), start=1):
        if document in wanted:
            return 1.0 / rank
    return 0.0


def evidence_recall(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Share of ``relevant`` anywhere in ``retrieved``."""
    wanted = list(relevant)
    return recall_at_k(retrieved, wanted, max(len(retrieved), 1))


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def percentile(values: Iterable[float], pct: float) -> float:
    """The ``pct`` percentile (0-100) by linear interpolation; 0.0 for no values."""
    items = sorted(values)
    if not items:
        return 0.0
    position = (len(items) - 1) * pct / 100
    low = int(position)
    high = min(low + 1, len(items) - 1)
    return items[low] + (items[high] - items[low]) * (position - low)
