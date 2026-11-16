"""Vector retrievers over the Pinecone chunk index.

* **R1** :class:`DenseRetriever` -- the baseline: embed the question with the dense model
  (BGE, with its query instruction) and take Pinecone's top ``k`` by dot product.

R1 is what GraphRAG has to beat (docs/layers/03-retrieval.md); it is deliberately plain.
"""

from __future__ import annotations

from typing import Any, Protocol

from ledgerlens.retrieval import QueryFilters, RetrievalResult, ScoredChunk
from ledgerlens.storage.pinecone_store import PineconeStore, VectorHit


class QueryEmbedder(Protocol):
    """The part of :class:`~ledgerlens.ingestion.embedding.DenseEmbedder` a retriever uses."""

    def embed_query(self, text: str) -> list[float]: ...


def _int(value: Any) -> int | None:
    return int(value) if isinstance(value, int | float) else None


def to_scored(hit: VectorHit, stage: str) -> ScoredChunk:
    """A Pinecone match as a retrieved chunk (text: the truncated copy in metadata)."""
    meta = hit.metadata
    return ScoredChunk(
        id=hit.id,
        document_id=hit.document_id,
        text=hit.text,
        score=hit.score,
        doc_type=meta.get("doc_type"),
        doc_number=meta.get("doc_number"),
        vendor_id=meta.get("vendor_id"),
        page=_int(meta.get("page")),
        chunk_index=_int(meta.get("chunk_index")),
        scores={stage: hit.score},
    )


class DenseRetriever:
    """R1: dense-only nearest neighbours of the question."""

    name = "r1_dense"

    def __init__(self, store: PineconeStore, embedder: QueryEmbedder) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(
        self, query: str, *, filters: QueryFilters | None = None, k: int = 5
    ) -> RetrievalResult:
        if filters is not None and not filters.is_empty():
            raise NotImplementedError("metadata filters are not supported yet")
        hits = self.store.query(self.embedder.embed_query(query), None, top_k=k)
        return RetrievalResult.build(query, self.name, [to_scored(h, "dense") for h in hits])
