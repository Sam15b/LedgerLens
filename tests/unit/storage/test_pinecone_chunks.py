"""PineconeStore against an in-memory fake index (no server)."""

from collections.abc import Iterator, Sequence
from datetime import date
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from ledgerlens.core.config import PineconeConfig
from ledgerlens.core.ids import chunk_vector_id, new_document_id
from ledgerlens.core.schemas import Contract, Invoice, Payment, PurchaseOrder, Receipt
from ledgerlens.ingestion.embedding import EmbeddedChunk, SparseVector
from ledgerlens.storage.pinecone_store import (
    DocumentKeys,
    PineconeStore,
    date_filter,
    date_key,
    hybrid_scale,
    truncate_utf8,
    vector_metadata,
)

DIM = 4
CONFIG = PineconeConfig(dimension=DIM, upsert_batch_size=3, metadata_text_max_bytes=16)


class FakeIndex:
    """Stores vectors per namespace; queries return them by dense dot product."""

    def __init__(self) -> None:
        self.namespaces: dict[str, dict[str, dict[str, Any]]] = {}
        self.upserts: list[int] = []
        self.queries: list[dict[str, Any]] = []
        self.deletes: list[list[str]] = []
        self.response: Any = {"matches": []}

    def upsert(self, *, vectors: list[dict[str, Any]], namespace: str) -> None:
        self.upserts.append(len(vectors))
        self.namespaces.setdefault(namespace, {}).update({v["id"]: v for v in vectors})

    def query(self, **kwargs: Any) -> Any:
        self.queries.append(kwargs)
        return self.response

    def delete(self, *, ids: Sequence[str], namespace: str) -> None:
        self.deletes.append(list(ids))
        for vector_id in ids:
            self.namespaces.get(namespace, {}).pop(vector_id, None)

    def list(self, *, prefix: str, namespace: str) -> Iterator[Any]:
        ids = sorted(i for i in self.namespaces.get(namespace, {}) if i.startswith(prefix))
        for start in range(0, len(ids), 2):  # small pages, like the real paginated listing
            yield SimpleNamespace(vectors=[SimpleNamespace(id=i) for i in ids[start : start + 2]])

    def describe_index_stats(self) -> Any:
        return SimpleNamespace(
            dimension=DIM,
            total_vector_count=sum(len(v) for v in self.namespaces.values()),
            namespaces={
                name: SimpleNamespace(vector_count=len(v)) for name, v in self.namespaces.items()
            },
        )


def embedded(
    document_id: str, n: int, text: str = "body", pages: list[str] | None = None
) -> EmbeddedChunk:
    return EmbeddedChunk(
        id=chunk_vector_id(document_id, n),
        values=[0.1] * DIM,
        sparse=SparseVector(indices=[1, 5], values=[0.5, 0.2]),
        metadata={
            "document_id": document_id,
            "chunk_index": n,
            "doc_type": "invoice",
            "kind": "header",
            "pages": pages or ["2", "3"],
            "text": text,
            "filename": "inv.pdf",
        },
    )


def test_date_key_is_a_sortable_int() -> None:
    assert date_key(date(2026, 3, 14)) == 20260314
    assert date_key(date(2025, 12, 31)) < date_key(date(2026, 1, 1))


def test_truncate_utf8_never_splits_a_character() -> None:
    assert truncate_utf8("short", 16) == "short"
    assert truncate_utf8("ééé", 5) == "éé"  # 2 bytes each: the third would be cut in half
    assert len(truncate_utf8("x" * 5000, 1024).encode()) == 1024


def test_metadata_keeps_filter_fields_and_truncated_text() -> None:
    document_id = new_document_id()
    keys = DocumentKeys(vendor_id="V001", doc_number="INV-889", doc_date=date(2026, 3, 14))

    metadata = vector_metadata(
        embedded(document_id, 0, text="x" * 100).metadata, keys, max_text_bytes=16
    )

    assert metadata == {
        "document_id": document_id,
        "doc_type": "invoice",
        "kind": "header",
        "chunk_index": 0,
        "page": 2,
        "vendor_id": "V001",
        "doc_number": "INV-889",
        "doc_date": 20260314,
        "text": "x" * 16,
    }


def test_metadata_leaves_out_unknown_values() -> None:
    metadata = vector_metadata(embedded(new_document_id(), 1).metadata, max_text_bytes=64)

    assert not {"vendor_id", "doc_number", "doc_date", "title", "filename"} & set(metadata)


@pytest.mark.parametrize(
    ("record", "number", "day"),
    [
        (
            Invoice(
                invoice_number="INV-1",
                vendor_name="Acme",
                invoice_date=date(2026, 1, 2),
                total=Decimal(1),
            ),
            "INV-1",
            date(2026, 1, 2),
        ),
        (
            Receipt(vendor_name="Kopi", receipt_date=date(2026, 1, 3), total=Decimal(1)),
            None,
            date(2026, 1, 3),
        ),
        (
            PurchaseOrder(
                po_number="PO-1", vendor_name="Acme", order_date=date(2026, 1, 4), total=Decimal(1)
            ),
            "PO-1",
            date(2026, 1, 4),
        ),
        (
            Payment(
                payment_reference="PAY-1",
                vendor_name="Acme",
                paid_at=date(2026, 1, 5),
                amount=Decimal(1),
            ),
            "PAY-1",
            date(2026, 1, 5),
        ),
        (
            Contract(contract_number="MSA-1", vendor_name="Acme", start_date=date(2026, 1, 6)),
            "MSA-1",
            date(2026, 1, 6),
        ),
        (None, None, None),
    ],
)
def test_document_keys_from_each_record_type(
    record: Any, number: str | None, day: date | None
) -> None:
    keys = DocumentKeys.from_record(record, vendor_id="V001")

    assert (keys.vendor_id, keys.doc_number, keys.doc_date) == ("V001", number, day)


def test_upsert_writes_in_batches_with_mapped_metadata() -> None:
    index, document_id = FakeIndex(), new_document_id()
    store = PineconeStore(index, namespace="dev", config=CONFIG)

    written = store.upsert_chunks(
        [embedded(document_id, n) for n in range(7)], keys=DocumentKeys(vendor_id="V001")
    )

    assert written == 7
    assert index.upserts == [3, 3, 1]
    stored = index.namespaces["dev"][chunk_vector_id(document_id, 0)]
    assert stored["sparse_values"] == {"indices": [1, 5], "values": [0.5, 0.2]}
    assert stored["metadata"]["vendor_id"] == "V001"
    assert "filename" not in stored["metadata"]


def test_upsert_to_another_namespace() -> None:
    index = FakeIndex()

    PineconeStore(index, namespace="dev", config=CONFIG).upsert_chunks(
        [embedded(new_document_id(), 0)], namespace="eval"
    )

    assert set(index.namespaces) == {"eval"}


def test_upsert_rejects_vectors_of_the_wrong_dimension() -> None:
    chunk = embedded(new_document_id(), 0).model_copy(update={"values": [0.1] * (DIM + 1)})

    with pytest.raises(ValueError, match="5-dim vector, index is 4"):
        PineconeStore(FakeIndex(), namespace="dev", config=CONFIG).upsert_chunks([chunk])


def test_upserting_nothing_makes_no_calls() -> None:
    index = FakeIndex()

    assert PineconeStore(index, namespace="dev", config=CONFIG).upsert_chunks([]) == 0
    assert index.upserts == []


# --------------------------------------------------------------------------- query


def store(index: FakeIndex | None = None, **config: Any) -> PineconeStore:
    return PineconeStore(
        index or FakeIndex(), namespace="dev", config=CONFIG.model_copy(update=config)
    )


def test_hybrid_scale_is_a_convex_combination() -> None:
    dense, sparse = hybrid_scale([1.0, 2.0], SparseVector(indices=[3], values=[0.5]), 0.75)

    assert dense == [0.75, 1.5]
    assert sparse == {"indices": [3], "values": [0.125]}


@pytest.mark.parametrize("sparse", [None, SparseVector()])
def test_hybrid_scale_without_sparse_terms_keeps_dense_as_is(sparse: SparseVector | None) -> None:
    assert hybrid_scale([1.0, 2.0], sparse, 0.3) == ([1.0, 2.0], None)


def test_alpha_one_is_pure_dense() -> None:
    assert hybrid_scale([1.0], SparseVector(indices=[1], values=[1.0]), 1.0) == ([1.0], None)


@pytest.mark.parametrize("alpha", [-0.1, 1.5])
def test_alpha_out_of_range_is_rejected(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        hybrid_scale([1.0], None, alpha)


def test_date_filter_bounds() -> None:
    assert date_filter(date(2026, 1, 1), date(2026, 3, 31)) == {
        "doc_date": {"$gte": 20260101, "$lte": 20260331}
    }
    assert date_filter(end=date(2026, 3, 31)) == {"doc_date": {"$lte": 20260331}}
    assert date_filter() == {}


def test_query_sends_weighted_vectors_with_config_defaults() -> None:
    index = FakeIndex()

    store(index, hybrid_alpha=0.5, query_top_k=7).query(
        [1.0] * DIM, SparseVector(indices=[2], values=[1.0])
    )

    sent = index.queries[0]
    assert sent["vector"] == [0.5] * DIM
    assert sent["sparse_vector"] == {"indices": [2], "values": [0.5]}
    assert (sent["top_k"], sent["namespace"], sent["include_metadata"]) == (7, "dev", True)
    assert "filter" not in sent


def test_query_overrides_and_filters() -> None:
    index = FakeIndex()
    where = {"doc_type": "invoice", **date_filter(start=date(2026, 1, 1))}

    store(index).query([1.0] * DIM, top_k=3, filter=where, namespace="eval", alpha=1.0)

    sent = index.queries[0]
    assert (sent["top_k"], sent["namespace"], sent["filter"]) == (3, "eval", where)
    assert "sparse_vector" not in sent


def test_query_returns_hits_best_first() -> None:
    document_id = new_document_id()
    index = FakeIndex()
    index.response = SimpleNamespace(
        matches=[
            SimpleNamespace(
                id=chunk_vector_id(document_id, 1),
                score=0.9,
                metadata={"document_id": document_id, "text": "Total 120.00"},
            ),
            SimpleNamespace(id=chunk_vector_id(document_id, 0), score=0.4, metadata=None),
        ]
    )

    hits = store(index).query([1.0] * DIM)

    assert [(h.score, h.document_id) for h in hits] == [(0.9, document_id), (0.4, document_id)]
    assert (hits[0].text, hits[1].text) == ("Total 120.00", "")


def test_query_rejects_the_wrong_dimension() -> None:
    with pytest.raises(ValueError, match="3 dims, index is 4"):
        store().query([1.0] * 3)


# --------------------------------------------------------------------------- delete + stats


def test_delete_document_removes_only_that_documents_vectors() -> None:
    index, keep, drop = FakeIndex(), new_document_id(), new_document_id()
    pinecone = store(index)
    pinecone.upsert_chunks([embedded(drop, n) for n in range(5)] + [embedded(keep, 0)])

    deleted = pinecone.delete_document(drop)

    assert deleted == 5
    assert list(index.namespaces["dev"]) == [chunk_vector_id(keep, 0)]
    assert pinecone.delete_document(drop) == 0  # already gone: a no-op
    assert pinecone.document_ids(keep) == [chunk_vector_id(keep, 0)]


def test_delete_document_batches_large_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ledgerlens.storage.pinecone_store.DELETE_BATCH_SIZE", 2)
    index, document_id = FakeIndex(), new_document_id()
    pinecone = store(index)
    pinecone.upsert_chunks([embedded(document_id, n) for n in range(5)])

    pinecone.delete_document(document_id)

    assert [len(batch) for batch in index.deletes] == [2, 2, 1]


def test_delete_document_only_touches_its_namespace() -> None:
    index, document_id = FakeIndex(), new_document_id()
    pinecone = store(index)
    pinecone.upsert_chunks([embedded(document_id, 0)])
    pinecone.upsert_chunks([embedded(document_id, 0)], namespace="eval")

    assert pinecone.delete_document(document_id, namespace="eval") == 1
    assert pinecone.count("dev") == 1


@pytest.mark.parametrize("bad", ["", "not-a-uuid", "3F2C0000-0000-4000-8000-000000000000"])
def test_delete_document_refuses_anything_but_a_document_id(bad: str) -> None:
    with pytest.raises(ValueError, match="document id"):
        store().delete_document(bad)


def test_stats_counts_vectors_per_namespace() -> None:
    index = FakeIndex()
    pinecone = store(index)
    pinecone.upsert_chunks([embedded(new_document_id(), n) for n in range(3)])
    pinecone.upsert_chunks([embedded(new_document_id(), 0)], namespace="eval")

    assert pinecone.stats() == {
        "dimension": DIM,
        "total_vector_count": 4,
        "namespaces": {"dev": 3, "eval": 1},
    }
    assert (pinecone.count(), pinecone.count("eval"), pinecone.count("prod")) == (3, 1, 0)


def test_stats_reads_plain_dict_responses() -> None:
    class DictIndex(FakeIndex):
        def describe_index_stats(self) -> Any:
            return {
                "dimension": DIM,
                "total_vector_count": 2,
                "namespaces": {"dev": {"vector_count": 2}},
            }

    assert store(DictIndex()).stats()["namespaces"] == {"dev": 2}
