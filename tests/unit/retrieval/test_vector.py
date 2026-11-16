"""R1 against a stubbed Pinecone index: ranking, filters, full text from Postgres."""

from collections.abc import Iterator, Sequence
from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest

from ledgerlens.core.config import PineconeConfig
from ledgerlens.core.ids import chunk_vector_id
from ledgerlens.core.schemas import DocType
from ledgerlens.retrieval import QueryFilters, RetrievalResult, Retriever
from ledgerlens.retrieval.filters import to_pinecone_filter
from ledgerlens.retrieval.vector import DenseRetriever
from ledgerlens.storage.pinecone_store import PineconeStore

DIM = 3
DOC_A = "00000000-0000-4000-8000-00000000000a"
DOC_B = "00000000-0000-4000-8000-00000000000b"
DOC_C = "00000000-0000-4000-8000-00000000000c"


def holds(value: Any, op: str, arg: Any) -> bool:
    if op == "$eq":
        return bool(value == arg)
    if op == "$in":
        return value in arg
    if value is None:
        return False
    return bool(value >= arg) if op == "$gte" else bool(value <= arg)


def matches(metadata: dict[str, Any], flt: dict[str, Any]) -> bool:
    """Pinecone's filter semantics for the operators retrieval uses."""
    return all(
        holds(metadata.get(key), op, arg) for key, cond in flt.items() for op, arg in cond.items()
    )


class StubIndex:
    """Dot-product search over a fixed set of vectors, honouring metadata filters."""

    def __init__(self, vectors: list[dict[str, Any]]) -> None:
        self.vectors = vectors
        self.queries: list[dict[str, Any]] = []

    def query(self, **kwargs: Any) -> Any:
        self.queries.append(kwargs)
        flt = kwargs.get("filter") or {}
        scored = [
            SimpleNamespace(
                id=v["id"],
                score=sum(a * b for a, b in zip(v["values"], kwargs["vector"], strict=True)),
                metadata=v["metadata"],
            )
            for v in self.vectors
            if matches(v["metadata"], flt)
        ]
        scored.sort(key=lambda m: -m.score)
        return SimpleNamespace(matches=scored[: kwargs["top_k"]])

    def upsert(self, **kwargs: Any) -> None:  # pragma: no cover - not used
        raise NotImplementedError

    def delete(self, **kwargs: Any) -> None:  # pragma: no cover - not used
        raise NotImplementedError

    def list(self, *, prefix: str, namespace: str) -> Iterator[Any]:  # pragma: no cover
        raise NotImplementedError

    def describe_index_stats(self) -> Any:  # pragma: no cover - not used
        raise NotImplementedError


def vector(doc: str, n: int, values: list[float], **metadata: Any) -> dict[str, Any]:
    return {
        "id": chunk_vector_id(doc, n),
        "values": values,
        "metadata": {"document_id": doc, "chunk_index": n, "text": f"short {doc[-1]}{n}"}
        | metadata,
    }


VECTORS = [
    vector(DOC_A, 0, [1.0, 0.0, 0.0], doc_type="invoice", vendor_id="v_acme",
           doc_number="INV-889", doc_date=20260314, page=1),
    vector(DOC_A, 1, [0.8, 0.6, 0.0], doc_type="invoice", vendor_id="v_acme",
           doc_number="INV-889", doc_date=20260314, page=1),
    vector(DOC_B, 0, [0.6, 0.8, 0.0], doc_type="purchase_order", vendor_id="v_acme",
           doc_number="PO-1042", doc_date=20260301),
    vector(DOC_C, 0, [0.0, 0.0, 1.0], doc_type="invoice", vendor_id="v_bolt",
           doc_number="INV-12", doc_date=20260520),
]  # fmt: skip


class FakeEmbedder:
    """Maps known questions to fixed directions."""

    QUESTIONS = {
        "acme invoice": [1.0, 0.1, 0.0],
        "purchase order": [0.5, 1.0, 0.0],
        "bolt": [0.0, 0.0, 1.0],
    }

    def __init__(self) -> None:
        self.seen: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.seen.append(text)
        return self.QUESTIONS[text]


class FakeTexts:
    def __init__(self, texts: dict[str, str]) -> None:
        self.full = texts
        self.asked: list[list[str]] = []

    def texts(self, ids: Sequence[str]) -> dict[str, str]:
        self.asked.append(list(ids))
        return {i: self.full[i] for i in ids if i in self.full}


@pytest.fixture
def index() -> StubIndex:
    return StubIndex(VECTORS)


def r1(index: StubIndex, texts: FakeTexts | None = None) -> DenseRetriever:
    store = PineconeStore(index, namespace="eval", config=PineconeConfig(dimension=DIM))
    return DenseRetriever(store, FakeEmbedder(), texts=texts)


# --------------------------------------------------------------------------- R1


def test_r1_is_a_retriever_returning_the_nearest_chunks_best_first(index: StubIndex) -> None:
    retriever = r1(index)
    assert isinstance(retriever, Retriever)

    result = retriever.retrieve("acme invoice", k=2)

    assert isinstance(result, RetrievalResult) and result.retriever == "r1_dense"
    assert [c.id for c in result.chunks] == [chunk_vector_id(DOC_A, 0), chunk_vector_id(DOC_A, 1)]
    first = result.chunks[0]
    assert (first.document_id, first.doc_type, first.doc_number, first.page) == (
        DOC_A,
        "invoice",
        "INV-889",
        1,
    )
    assert first.score == pytest.approx(1.0) and first.scores == {"dense": pytest.approx(1.0)}
    assert result.citations == {"doc:INV-889": DOC_A}
    assert result.document_ids == [DOC_A]
    assert index.queries[0]["top_k"] == 2
    assert "sparse_vector" not in index.queries[0]  # dense only
    assert "filter" not in index.queries[0]


def test_r1_returns_the_full_chunk_text_from_postgres(index: StubIndex) -> None:
    full = {chunk_vector_id(DOC_A, 0): "the whole first chunk, longer than 1 KB in real life"}
    texts = FakeTexts(full)

    result = r1(index, texts).retrieve("acme invoice", k=2)

    assert [c.text for c in result.chunks] == [
        "the whole first chunk, longer than 1 KB in real life",
        "short a1",  # not in Postgres (yet): falls back to the truncated copy
    ]
    assert texts.asked == [[chunk_vector_id(DOC_A, 0), chunk_vector_id(DOC_A, 1)]]


def test_r1_without_hits_does_not_query_postgres(index: StubIndex) -> None:
    texts = FakeTexts({})
    filters = QueryFilters(doc_numbers=("INV-404",))

    result = r1(index, texts).retrieve("acme invoice", filters=filters)

    assert result.chunks == [] and texts.asked == []


def test_r1_applies_query_filters_in_pinecone(index: StubIndex) -> None:
    retriever = r1(index)

    po = retriever.retrieve(
        "acme invoice", filters=QueryFilters(doc_types=(DocType.PURCHASE_ORDER,))
    )
    bolt = retriever.retrieve("acme invoice", filters=QueryFilters(vendor_ids=("v_bolt",)))
    march = retriever.retrieve(
        "bolt",
        filters=QueryFilters(date_from=date(2026, 3, 1), date_to=date(2026, 3, 31)),
        k=10,
    )

    assert [c.doc_number for c in po.chunks] == ["PO-1042"]
    assert [c.doc_number for c in bolt.chunks] == ["INV-12"]
    assert {c.document_id for c in march.chunks} == {DOC_A, DOC_B}  # not the May invoice
    assert index.queries[0]["filter"] == {"doc_type": {"$eq": "purchase_order"}}


# --------------------------------------------------------------------------- filters


def test_empty_filters_mean_no_filter() -> None:
    assert to_pinecone_filter(None) == {}
    assert to_pinecone_filter(QueryFilters()) == {}
    assert QueryFilters().is_empty()


def test_one_value_is_eq_several_are_in_and_dates_are_int_ranges() -> None:
    filters = QueryFilters(
        doc_types=(DocType.INVOICE, DocType.RECEIPT),
        vendor_ids=("v_acme", "v_acme"),
        doc_numbers=("INV-889",),
        date_from=date(2026, 1, 1),
        date_to=date(2026, 3, 31),
    )

    assert to_pinecone_filter(filters) == {
        "doc_type": {"$in": ["invoice", "receipt"]},
        "vendor_id": {"$eq": "v_acme"},  # repeated values collapse
        "doc_number": {"$eq": "INV-889"},
        "doc_date": {"$gte": 20260101, "$lte": 20260331},
    }
    assert to_pinecone_filter(QueryFilters(date_to=date(2026, 2, 28))) == {
        "doc_date": {"$lte": 20260228}
    }


def test_a_backwards_date_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="after date_to"):
        QueryFilters(date_from=date(2026, 3, 1), date_to=date(2026, 2, 1))
