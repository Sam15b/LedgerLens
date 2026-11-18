"""Vector retrievers over the Pinecone chunk index.

* **R1** :class:`DenseRetriever` -- the baseline: embed the question with the dense model
  (BGE, with its query instruction) and take Pinecone's top ``k`` by dot product, within
  the question's :class:`~ledgerlens.retrieval.QueryFilters` (as metadata filters).

* **R2** :class:`HybridRetriever` -- the same query plus a BM25 sparse vector of the
  question, weighted ``alpha`` dense to ``1 - alpha`` sparse (``hybrid_alpha`` in
  pinecone.yaml, 0.7). The sparse side matches exact tokens -- ``INV-889``, ``PO-1042``,
  amounts -- that a dense embedding blurs.

* **R3** :class:`RerankRetriever` -- R2's top ``candidates`` (20), re-sorted by a
  cross-encoder (:mod:`ledgerlens.retrieval.rerank`) and cut to ``k`` (5).

Every result carries ``timings_ms`` per stage -- ``embed``, ``search``, ``fetch_text``,
``rerank`` (R3) -- and ``total``, for the latency columns of the benchmark.

R1 is what GraphRAG has to beat (docs/layers/03-retrieval.md); it is deliberately plain.
Note that the Pinecone Local emulator ignores sparse values, so R2 only differs from R1
on the serverless index.

Pinecone keeps only the first ~1 KB of each chunk in metadata; the full text lives in the
Postgres ``chunks`` table under the same id. A :class:`ChunkTexts` source fills it in
after the search (one query for all hits); without one -- or for a chunk Postgres does not
have -- the truncated copy is used.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from sqlalchemy.orm import Session, sessionmaker

from ledgerlens.ingestion.embedding import SparseVector
from ledgerlens.retrieval import (
    QueryFilters,
    RetrievalResult,
    Retriever,
    ScoredChunk,
    StageTimer,
)
from ledgerlens.retrieval.filters import to_pinecone_filter
from ledgerlens.storage.pinecone_store import VectorHit
from ledgerlens.storage.postgres.db import get_session_factory
from ledgerlens.storage.postgres.repositories import ChunkRepository


class VectorIndex(Protocol):
    """What a retriever queries: :class:`~ledgerlens.storage.pinecone_store.PineconeStore`,
    or a stand-in with the same :meth:`query` (``eval.tune_retrieval.ExactHybridStore``)."""

    def query(
        self,
        dense: Sequence[float],
        sparse: SparseVector | None = None,
        *,
        top_k: int | None = None,
        filter: Mapping[str, Any] | None = None,
        namespace: str | None = None,
        alpha: float | None = None,
    ) -> list[VectorHit]: ...


class QueryEmbedder(Protocol):
    """The part of :class:`~ledgerlens.ingestion.embedding.DenseEmbedder` a retriever uses."""

    def embed_query(self, text: str) -> list[float]: ...


class SparseQueryEncoder(Protocol):
    """The part of :class:`~ledgerlens.ingestion.embedding.BM25Encoder` R2 uses (fitted on
    the indexed chunks: load it from ``embedding.sparse_params_path``)."""

    def encode_query(self, text: str) -> SparseVector: ...


class Reranker(Protocol):
    """Re-sorts chunks by relevance (:class:`~ledgerlens.retrieval.rerank.CrossEncoderReranker`)."""

    def rerank(
        self, query: str, chunks: Sequence[ScoredChunk], top_n: int | None = None
    ) -> list[ScoredChunk]: ...


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
    stage = "dense"
    """The key of the vector score in :attr:`ScoredChunk.scores`."""

    def __init__(
        self, store: VectorIndex, embedder: QueryEmbedder, *, texts: ChunkTexts | None = None
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.texts = texts
        self.alpha: float | None = None

    def encode_sparse(self, query: str) -> SparseVector | None:
        """The sparse half of the query: none for R1."""
        return None

    def retrieve(
        self, query: str, *, filters: QueryFilters | None = None, k: int = 5
    ) -> RetrievalResult:
        timer = StageTimer()
        with timer.stage("embed"):
            dense = self.embedder.embed_query(query)
            sparse = self.encode_sparse(query)
        with timer.stage("search"):
            hits = self.store.query(
                dense,
                sparse,
                top_k=k,
                filter=to_pinecone_filter(filters) or None,
                alpha=self.alpha,
            )
        with timer.stage("fetch_text"):
            chunks = with_full_text([to_scored(h, self.stage) for h in hits], self.texts)
        return RetrievalResult.build(query, self.name, chunks, timings_ms=timer.done())


class HybridRetriever(DenseRetriever):
    """R2: dense + BM25 sparse in one Pinecone query, ``alpha``-weighted."""

    name = "r2_hybrid"
    stage = "hybrid"

    def __init__(
        self,
        store: VectorIndex,
        embedder: QueryEmbedder,
        sparse: SparseQueryEncoder,
        *,
        alpha: float | None = None,
        texts: ChunkTexts | None = None,
    ) -> None:
        super().__init__(store, embedder, texts=texts)
        self.sparse = sparse
        self.alpha = alpha  # None: the index config's hybrid_alpha

    def encode_sparse(self, query: str) -> SparseVector | None:
        return self.sparse.encode_query(query)


DEFAULT_RERANK_CANDIDATES = 20


class RerankRetriever:
    """R3: a retriever's top ``candidates`` (R2's, normally), reranked to the best ``k``."""

    name = "r3_hybrid_rerank"

    def __init__(
        self,
        base: Retriever,
        reranker: Reranker,
        *,
        candidates: int = DEFAULT_RERANK_CANDIDATES,
    ) -> None:
        if candidates < 1:
            raise ValueError("candidates must be at least 1")
        self.base = base
        self.reranker = reranker
        self.candidates = candidates

    def retrieve(
        self, query: str, *, filters: QueryFilters | None = None, k: int = 5
    ) -> RetrievalResult:
        timer = StageTimer()
        pool = self.base.retrieve(query, filters=filters, k=max(k, self.candidates))
        timer.timings_ms.update(pool.timings_ms)
        timer.timings_ms.pop("total", None)
        with timer.stage("rerank"):
            ranked = self.reranker.rerank(query, pool.chunks, top_n=k)
        return RetrievalResult.build(query, self.name, ranked, timings_ms=timer.done())
