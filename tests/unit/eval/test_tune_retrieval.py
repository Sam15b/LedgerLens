"""Retrieval tuning: exact hybrid scoring, the pick rule and the sweep report."""

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from eval.benchmark_retrieval import Summary
from eval.golden_set.schema import GoldenQuestion
from eval.tune_retrieval import (
    BM25Documents,
    ExactHybridStore,
    SweepRow,
    choose,
    render_report,
    sparse_dot,
    sweep,
)
from ledgerlens.ingestion.embedding import BM25Encoder, SparseVector
from ledgerlens.retrieval import QueryFilters, RetrievalResult, ScoredChunk
from ledgerlens.storage.pinecone_store import VectorHit

A, B, C = (f"{n}0000000-0000-4000-8000-000000000000" for n in "abc")


def hit(document_id: str, score: float) -> VectorHit:
    return VectorHit(id=f"{document_id}#c0", score=score, metadata={"document_id": document_id})


class DenseIndex:
    """Returns a fixed dense ranking and counts the queries it gets."""

    def __init__(self, hits: list[VectorHit]) -> None:
        self.hits = hits
        self.calls: list[dict[str, Any]] = []

    def query(
        self,
        dense: Sequence[float],
        sparse: SparseVector | None = None,
        *,
        top_k: int | None = None,
        filter: Mapping[str, Any] | None = None,
        namespace: str | None = None,
        alpha: float | None = None,
    ) -> list[VectorHit]:
        self.calls.append({"sparse": sparse, "top_k": top_k, "filter": filter})
        return self.hits[: top_k or len(self.hits)]


class Texts:
    def __init__(self, texts: dict[str, str]) -> None:
        self.data = texts
        self.asked: list[list[str]] = []

    def texts(self, ids: Sequence[str]) -> dict[str, str]:
        self.asked.append(list(ids))
        return {i: self.data[i] for i in ids if i in self.data}


@pytest.fixture
def encoder() -> BM25Encoder:
    return BM25Encoder().fit(["pallet wrap invoice", "invoice inv-889 freight", "contract terms"])


@pytest.fixture
def store(encoder: BM25Encoder) -> ExactHybridStore:
    # A reads like the question (best dense); B holds the exact token asked for.
    index = DenseIndex([hit(A, 0.9), hit(B, 0.6), hit(C, 0.5)])
    texts = Texts(
        {
            f"{A}#c0": "pallet wrap invoice",
            f"{B}#c0": "invoice inv-889 freight",
            f"{C}#c0": "contract terms",
        }
    )
    return ExactHybridStore(index, BM25Documents(encoder, texts), alpha=0.7, top_k=2)


def test_sparse_dot() -> None:
    a = SparseVector(indices=[1, 2, 5], values=[0.5, 1.0, 2.0])
    b = SparseVector(indices=[2, 5, 9], values=[3.0, 0.25, 7.0])
    assert sparse_dot(a, b) == pytest.approx(3.5)
    assert sparse_dot(a, SparseVector()) == 0.0


def test_alpha_one_or_no_sparse_keeps_the_dense_ranking(
    store: ExactHybridStore, encoder: BM25Encoder
) -> None:
    query = encoder.encode_query("inv-889")
    assert [h.id for h in store.query([1.0], None)] == [f"{A}#c0", f"{B}#c0"]
    dense = store.query([1.0], query, alpha=1.0)
    assert [(h.id, h.score) for h in dense] == [(f"{A}#c0", 0.9), (f"{B}#c0", 0.6)]


def test_low_alpha_lifts_the_keyword_match(store: ExactHybridStore, encoder: BM25Encoder) -> None:
    query = encoder.encode_query("inv-889")
    hits = store.query([1.0], query, alpha=0.2, top_k=3)
    assert hits[0].id == f"{B}#c0"
    doc_b = BM25Documents(encoder, Texts({"b": "invoice inv-889 freight"})).vectors(["b"])["b"]
    assert hits[0].score == pytest.approx(0.2 * 0.6 + 0.8 * sparse_dot(query, doc_b))
    # no sparse overlap: ties at 0 sparse are broken by the dense score
    assert [h.id for h in hits[1:]] == [f"{A}#c0", f"{C}#c0"]


def test_dense_ranking_is_fetched_once_per_query_and_filter(
    store: ExactHybridStore, encoder: BM25Encoder
) -> None:
    index = store.store
    assert isinstance(index, DenseIndex)
    query = encoder.encode_query("inv-889")
    for alpha in (0.0, 0.5, 1.0):
        store.query([1.0], query, alpha=alpha)
    store.query([1.0], query, filter={"doc_type": "invoice"})
    assert len(index.calls) == 2
    assert all(call["sparse"] is None and call["top_k"] == store.pool for call in index.calls)
    assert index.calls[1]["filter"] == {"doc_type": "invoice"}


def test_alpha_out_of_range(store: ExactHybridStore) -> None:
    with pytest.raises(ValueError, match="alpha"):
        store.query([1.0], None, alpha=1.5)


def test_documents_are_encoded_once_and_missing_text_has_no_terms(encoder: BM25Encoder) -> None:
    texts = Texts({"x": "pallet wrap"})
    documents = BM25Documents(encoder, texts)
    first = documents.vectors(["x", "gone"])
    again = documents.vectors(["gone", "x"])
    assert first["gone"] == SparseVector() == again["gone"]
    assert len(first["x"]) == 2
    assert texts.asked == [["x", "gone"]]


# --------------------------------------------------------------------------- picking


def summary(recall: float, mrr: float, p50: float = 100.0, evidence: float = 0.0) -> Summary:
    return Summary(10, 10, recall, mrr, evidence, p50, p50, 0, 0)


def test_choose_by_metric_then_mrr_then_cost() -> None:
    rows = [
        SweepRow(0.3, summary(0.6, 0.5), 0.0),
        SweepRow(0.5, summary(0.7, 0.6, p50=300), 0.0),
        SweepRow(0.7, summary(0.7, 0.6, p50=200), 0.0),
        SweepRow(0.9, summary(0.7, 0.55), 0.0),
    ]
    assert choose(rows, "recall") == 0.7  # tied with 0.5 on recall and MRR, cheaper
    assert choose(rows, "recall", cost=lambda r: abs(r.value - 0.5)) == 0.5
    assert choose(rows, "mrr") == 0.7
    with pytest.raises(ValueError):
        choose([], "recall")


def test_equal_cost_goes_to_the_smaller_value() -> None:
    rows = [SweepRow(v, summary(0.5, 0.5), 0.0) for v in (30, 10, 20)]
    assert choose(rows, "recall", cost=lambda r: 0.0) == 10


# --------------------------------------------------------------------------- sweep


class FixedRetriever:
    """Returns ``documents`` as chunks, whatever the question."""

    name = "stub"

    def __init__(self, documents: list[str]) -> None:
        self.documents = documents

    def retrieve(
        self, query: str, *, filters: QueryFilters | None = None, k: int = 5
    ) -> RetrievalResult:
        chunks = [
            ScoredChunk(id=f"{d}#c0", document_id=d, text="t", score=1.0)
            for d in self.documents[:k]
        ]
        return RetrievalResult.build(query, self.name, chunks, timings_ms={"total": 1.0})


QUESTIONS = [
    GoldenQuestion.model_validate(
        {
            "id": "q001",
            "question": "total on INV-000001?",
            "ground_truth_answer": "100.00",
            "difficulty": "easy",
            "category": "lookup",
            "requires_graph": False,
            "relevant_document_ids": [B],
            "expected": {},
        }
    )
]


def test_sweep_scores_every_value_and_renders_the_pick() -> None:
    by_value = {0.0: [A, C], 0.5: [A, B], 1.0: [B]}
    result = sweep(
        QUESTIONS,
        "alpha",
        "R2",
        lambda value: FixedRetriever(by_value[value]),
        list(by_value),
        metric="mrr",
        current=0.5,
        k=2,
        cost=lambda row: abs(row.value - 0.5),
    )
    assert [row.summary.mrr for row in result.rows] == [0.0, 0.5, 1.0]
    assert result.best == 1.0
    report = render_report([result], k=2, setup={"k": "2"})
    assert "| alpha | R2 | 0.5 | 1 |" in report
    assert "| **1** (picked) | 1.000 | 1.000 |" in report
    assert "| 0.5 (was) | 1.000 | 0.500 |" in report
