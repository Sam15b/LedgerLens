"""Vector retrievers over the Pinecone chunk index.

* **R1** :class:`DenseRetriever` -- the baseline: embed the question with the dense model
  (BGE, with its query instruction) and take Pinecone's top ``k`` by dot product, within
  the question's :class:`~ledgerlens.retrieval.QueryFilters` (as metadata filters).

R1 is what GraphRAG has to beat (docs/layers/03-retrieval.md); it is deliberately plain.

Pinecone keeps only the first ~1 KB of each chunk in metadata; the full text lives in the
Postgres ``chunks`` table under the same id. A :class:`ChunkTexts` source fills it in
after the search (one query for all hits); without one -- or for a chunk Postgres does not
have -- the truncated copy is used.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from sqlalchemy.orm import Session, sessionmaker

from ledgerlens.retrieval import QueryFilters, RetrievalResult, ScoredChunk
from ledgerlens.retrieval.filters import to_pinecone_filter
from ledgerlens.storage.pinecone_store import PineconeStore, VectorHit
from ledgerlens.storage.postgres.db import get_session_factory
from ledgerlens.storage.postgres.repositories import ChunkRepository


class QueryEmbedder(Protocol):
    """The part of :class:`~ledgerlens.ingestion.embedding.DenseEmbedder` a retriever uses."""

    def embed_query(self, text: str) -> list[float]: ...


class ChunkTexts(Protocol):
    """Full chunk text by vector id (ids it does not know are left out)."""

    def texts(self, ids: Sequence[str]) -> dict[str, str]: ...


class PostgresChunkTexts:
    """:class:`ChunkTexts` from the Postgres ``chunks`` table."""

    def __init__(self, sessions: sessionmaker[Session] | None = None) -> None:
        self.sessions = sessions or get_session_factory()

    def texts(self, ids: Sequence[str]) -> dict[str, str]:
        with self.sessions() as session:
            return ChunkRepository(session).texts(ids)


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


def with_full_text(chunks: list[ScoredChunk], source: ChunkTexts | None) -> list[ScoredChunk]:
    """``chunks`` with their text replaced by the full text from ``source``."""
    if source is None or not chunks:
        return chunks
    full = source.texts([chunk.id for chunk in chunks])
    return [
        chunk.model_copy(update={"text": full[chunk.id]}) if chunk.id in full else chunk
        for chunk in chunks
    ]


class DenseRetriever:
    """R1: dense-only nearest neighbours of the question."""

    name = "r1_dense"

    def __init__(
        self, store: PineconeStore, embedder: QueryEmbedder, *, texts: ChunkTexts | None = None
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.texts = texts

    def retrieve(
        self, query: str, *, filters: QueryFilters | None = None, k: int = 5
    ) -> RetrievalResult:
        hits = self.store.query(
            self.embedder.embed_query(query),
            None,
            top_k=k,
            filter=to_pinecone_filter(filters) or None,
        )
        chunks = with_full_text([to_scored(h, "dense") for h in hits], self.texts)
        return RetrievalResult.build(query, self.name, chunks)
