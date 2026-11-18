"""Build retrievers from ``configs/``: the default one, or any variant by name.

    retriever = build_retriever()          # retrieval.default (R4, GraphRAG)
    result = retriever.retrieve("which payment settled INV-000889?")

Each variant builds on the one before it -- R2 hybrid, R3 = R2 reranked, R4 = R3 seeds
expanded through the graph, R5 = R4 plus Text2Cypher -- and a :class:`RetrieverFactory`
builds every shared part once (embedder, BM25 encoder, cross-encoder, query analyzer, Neo4j
driver), so several variants can be compared without loading the models twice. Sizes come
from the config: ``pinecone.hybrid_alpha`` and ``pinecone.query_top_k`` (rerank
candidates), ``retrieval.rerank_model`` and ``retrieval.graph``.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any

from ledgerlens.core.config import AppConfig, get_config
from ledgerlens.retrieval import Retriever

if TYPE_CHECKING:
    from ledgerlens.llm.router import LLMClient
    from ledgerlens.retrieval.vector import VectorIndex

VARIANTS = ("R1", "R2", "R3", "R4", "R5")


class RetrieverFactory:
    """Builds and caches the retriever variants over the live stores."""

    def __init__(
        self,
        *,
        config: AppConfig | None = None,
        namespace: str | None = None,
        store: VectorIndex | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.config = config or get_config()
        self.namespace = namespace
        self._store = store
        self._llm = llm
        self._built: dict[str, Retriever] = {}

    # ----------------------------------------------------------------- shared parts

    @cached_property
    def store(self) -> VectorIndex:
        if self._store is not None:
            return self._store
        from ledgerlens.storage.pinecone_store import PineconeStore

        return PineconeStore.from_settings(namespace=self.namespace)

    @cached_property
    def embedder(self) -> Any:
        from ledgerlens.ingestion.embedding import DenseEmbedder

        return DenseEmbedder()

    @cached_property
    def sparse(self) -> Any:
        from ledgerlens.ingestion.embedding import BM25Encoder

        return BM25Encoder.load()

    @cached_property
    def texts(self) -> Any:
        from ledgerlens.retrieval.vector import PostgresChunkTexts

        return PostgresChunkTexts()

    @cached_property
    def driver(self) -> Any:
        from ledgerlens.storage.neo4j.driver import get_driver

        return get_driver()

    @cached_property
    def analyzer(self) -> Any:
        from ledgerlens.retrieval.analyze import QueryAnalyzer

        return QueryAnalyzer.from_postgres()

    # ----------------------------------------------------------------- variants

    def get(self, variant: str | None = None) -> Retriever:
        """The retriever for ``variant`` (default: ``retrieval.default``)."""
        name = (variant or self.config.retrieval.default).upper()
        if name not in VARIANTS:
            raise ValueError(f"unknown retriever {name!r}; choose from {', '.join(VARIANTS)}")
        if name not in self._built:
            self._built[name] = self._build(name)
        return self._built[name]

    def _build(self, name: str) -> Retriever:
        from ledgerlens.retrieval.vector import DenseRetriever, HybridRetriever, RerankRetriever

        if name == "R1":
            return DenseRetriever(self.store, self.embedder, texts=self.texts)
        if name == "R2":
            return HybridRetriever(
                self.store,
                self.embedder,
                self.sparse,
                alpha=self.config.pinecone.hybrid_alpha,
                texts=self.texts,
            )
        if name == "R3":
            from ledgerlens.retrieval.rerank import CrossEncoderReranker

            return RerankRetriever(
                self.get("R2"),
                CrossEncoderReranker(self.config.retrieval.rerank_model),
                candidates=self.config.pinecone.query_top_k,
            )
        from ledgerlens.retrieval.graph import GraphExpander
        from ledgerlens.retrieval.graphrag import GraphRAGRetriever, Text2CypherRetriever

        if name == "R4":
            graph = self.config.retrieval.graph
            expander = GraphExpander(
                self.driver, max_hops=graph.max_hops, per_seed=graph.per_seed, limit=graph.limit
            )
            return GraphRAGRetriever(self.get("R3"), expander, analyzer=self.analyzer)
        from ledgerlens.llm.router import LLMClient
        from ledgerlens.retrieval.text2cypher import CypherGenerator, CypherGuard

        r4 = self.get("R4")
        assert isinstance(r4, GraphRAGRetriever)
        generator = CypherGenerator(self._llm or LLMClient())
        return Text2CypherRetriever(r4, generator, CypherGuard(self.driver))


def build_retriever(variant: str | None = None, **options: Any) -> Retriever:
    """One retriever (default: ``retrieval.default``); ``options`` go to
    :class:`RetrieverFactory`."""
    return RetrieverFactory(**options).get(variant)
