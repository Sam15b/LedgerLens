"""R2 hybrid weighting, the cross-encoder reranker and R3, with stub models and index."""

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from ledgerlens.core.config import PineconeConfig
from ledgerlens.core.ids import chunk_vector_id
from ledgerlens.ingestion.embedding import SparseVector
from ledgerlens.retrieval import Retriever, ScoredChunk, StageTimer
from ledgerlens.retrieval.rerank import CrossEncoderReranker
from ledgerlens.retrieval.vector import DenseRetriever, HybridRetriever, RerankRetriever
from ledgerlens.storage.pinecone_store import PineconeStore

DIM = 2
DOC_X = "00000000-0000-4000-8000-0000000000a1"  # semantically close to the question
DOC_Y = "00000000-0000-4000-8000-0000000000b2"  # contains the exact token asked for
TOKEN = 889  # sparse dimension of "inv-889"


class HybridIndex:
    """Scores like Pinecone's dotproduct index: dense dot + sparse dot."""

    def __init__(self) -> None:
        self.vectors: list[dict[str, Any]] = [
            {
                "id": chunk_vector_id(DOC_X, 0),
                "values": [1.0, 0.0],
                "sparse": {},
                "metadata": {"document_id": DOC_X, "text": "invoice for pallet wrap"},
            },
            {
                "id": chunk_vector_id(DOC_Y, 0),
                "values": [0.6, 0.8],
                "sparse": {TOKEN: 1.0},
                "metadata": {
                    "document_id": DOC_Y,
                    "doc_number": "INV-889",
                    "text": "INVOICE INV-889 total 498.00",
                },
            },
        ]
        self.queries: list[dict[str, Any]] = []

    def query(self, **kwargs: Any) -> Any:
        self.queries.append(kwargs)
        sparse = kwargs.get("sparse_vector") or {"indices": [], "values": []}
        weights = dict(zip(sparse["indices"], sparse["values"], strict=True))
        matches = []
        for v in self.vectors:
            dense = sum(a * b for a, b in zip(v["values"], kwargs["vector"], strict=True))
            lexical = sum(w * weights.get(i, 0.0) for i, w in v["sparse"].items())
            matches.append(
                SimpleNamespace(id=v["id"], score=dense + lexical, metadata=v["metadata"])
            )
        matches.sort(key=lambda m: -m.score)
        return SimpleNamespace(matches=matches[: kwargs["top_k"]])

    def upsert(self, **kwargs: Any) -> None:  # pragma: no cover - not used
        raise NotImplementedError

    def delete(self, **kwargs: Any) -> None:  # pragma: no cover - not used
        raise NotImplementedError

    def list(self, *, prefix: str, namespace: str) -> Iterator[Any]:  # pragma: no cover
        raise NotImplementedError

    def describe_index_stats(self) -> Any:  # pragma: no cover - not used
        raise NotImplementedError


class Embedder:
    def embed_query(self, text: str) -> list[float]:
        return [0.9, 0.1]  # closer to DOC_X than to DOC_Y


class BM25:
    def encode_query(self, text: str) -> SparseVector:
        return SparseVector(indices=[TOKEN], values=[2.0])


def store(index: HybridIndex) -> PineconeStore:
    config = PineconeConfig(dimension=DIM, hybrid_alpha=0.7)
    return PineconeStore(index, namespace="eval", config=config)


# --------------------------------------------------------------------------- R2


def test_r2_sends_the_sparse_query_weighted_by_alpha() -> None:
    index = HybridIndex()

    HybridRetriever(store(index), Embedder(), BM25()).retrieve("total on INV-889", k=2)
    HybridRetriever(store(index), Embedder(), BM25(), alpha=0.25).retrieve("q", k=2)

    default, override = index.queries
    assert default["vector"] == pytest.approx([0.9 * 0.7, 0.1 * 0.7])  # hybrid_alpha
    assert default["sparse_vector"] == {"indices": [TOKEN], "values": [pytest.approx(0.6)]}
    assert override["vector"] == pytest.approx([0.9 * 0.25, 0.1 * 0.25])
    assert override["sparse_vector"]["values"] == [pytest.approx(1.5)]


def test_sparse_weight_lifts_the_exact_token_match_above_the_dense_one() -> None:
    def top(alpha: float) -> str:
        retriever = HybridRetriever(store(HybridIndex()), Embedder(), BM25(), alpha=alpha)
        result = retriever.retrieve("total on INV-889", k=2)
        return result.chunks[0].document_id

    dense_only = DenseRetriever(store(HybridIndex()), Embedder()).retrieve("q", k=2)

    assert dense_only.chunks[0].document_id == DOC_X  # R1 misses the exact number
    assert top(1.0) == DOC_X  # alpha 1 = dense only
    assert top(0.7) == DOC_Y  # the default weighting finds INV-889
    hybrid = HybridRetriever(store(HybridIndex()), Embedder(), BM25()).retrieve("q", k=1)
    assert hybrid.retriever == "r2_hybrid" and set(hybrid.chunks[0].scores) == {"hybrid"}


# --------------------------------------------------------------------------- reranker


class PairScorer:
    """Relevance = fixed score per chunk text; records what it was asked."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, inputs: list[tuple[str, str]], /, **kwargs: Any) -> Any:
        self.calls.append(inputs)
        return [self.scores[text] for _, text in inputs]


def chunk(name: str, score: float) -> ScoredChunk:
    return ScoredChunk(
        id=f"{name}#c0", document_id=name, text=name, score=score, scores={"hybrid": score}
    )


def test_rerank_orders_by_cross_encoder_score_and_keeps_earlier_scores() -> None:
    model = PairScorer({"a": 0.10, "b": 0.95, "c": 0.40, "d": 0.40})
    reranker = CrossEncoderReranker(model=model)
    chunks = [chunk("a", 0.9), chunk("b", 0.8), chunk("c", 0.7), chunk("d", 0.6)]

    ranked = reranker.rerank("question", chunks, top_n=3)

    assert [c.id for c in ranked] == ["b#c0", "c#c0", "d#c0"]  # c before d: tie, input order
    assert ranked[0].score == 0.95
    assert ranked[0].scores == {"hybrid": 0.8, "rerank": 0.95}
    assert model.calls == [[("question", t) for t in "abcd"]]
    assert [c.id for c in reranker.rerank("question", chunks)] == ["b#c0", "c#c0", "d#c0", "a#c0"]


def test_rerank_of_nothing_does_not_load_or_call_the_model() -> None:
    reranker = CrossEncoderReranker(model=PairScorer({}))
    assert reranker.rerank("question", []) == []
    assert reranker.model.calls == []  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- R3


def test_r3_reranks_a_wider_pool_and_records_every_stage() -> None:
    index = HybridIndex()
    base = HybridRetriever(store(index), Embedder(), BM25(), alpha=1.0)  # dense: X first
    model = PairScorer({"invoice for pallet wrap": 0.02, "INVOICE INV-889 total 498.00": 0.99})
    r3 = RerankRetriever(base, CrossEncoderReranker(model=model), candidates=20)
    assert isinstance(r3, Retriever)

    result = r3.retrieve("total on INV-889", k=1)

    assert index.queries[0]["top_k"] == 20  # the pool, not k
    assert [c.document_id for c in result.chunks] == [DOC_Y]  # rerank fixes the order
    assert result.retriever == "r3_hybrid_rerank"
    assert result.citations == {"doc:INV-889": DOC_Y}
    assert set(result.timings_ms) == {"embed", "search", "fetch_text", "rerank", "total"}
    stages = sum(v for name, v in result.timings_ms.items() if name != "total")
    assert result.timings_ms["total"] >= result.timings_ms["rerank"]
    assert stages > 0


def test_r3_needs_a_candidate_pool() -> None:
    base = DenseRetriever(store(HybridIndex()), Embedder())
    with pytest.raises(ValueError, match="candidates"):
        RerankRetriever(base, CrossEncoderReranker(model=PairScorer({})), candidates=0)


def test_r1_and_r2_time_their_stages() -> None:
    result = DenseRetriever(store(HybridIndex()), Embedder()).retrieve("q", k=1)
    assert set(result.timings_ms) == {"embed", "search", "fetch_text", "total"}


def test_stage_timer_adds_up_repeated_stages() -> None:
    timer = StageTimer()
    with timer.stage("search"):
        pass
    first = timer.timings_ms["search"]
    with timer.stage("search"):
        sum(range(1000))
    done = timer.done()
    assert done["search"] >= first and done["total"] >= done["search"]
