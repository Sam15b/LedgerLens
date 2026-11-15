"""PineconeStore against an in-memory fake index (no server)."""

from collections.abc import Iterator, Sequence
from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from ledgerlens.core.config import PineconeConfig
from ledgerlens.core.ids import chunk_vector_id, new_document_id
from ledgerlens.core.schemas import Contract, Invoice, Payment, PurchaseOrder, Receipt
from ledgerlens.ingestion.embedding import EmbeddedChunk, SparseVector
from ledgerlens.storage.pinecone_store import (
    DocumentKeys,
    PineconeStore,
    date_key,
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

    def upsert(self, *, vectors: list[dict[str, Any]], namespace: str) -> None:
        self.upserts.append(len(vectors))
        self.namespaces.setdefault(namespace, {}).update({v["id"]: v for v in vectors})

    def query(self, **kwargs: Any) -> Any:
        self.queries.append(kwargs)
        return {"matches": []}

    def delete(self, *, ids: Sequence[str], namespace: str) -> None:
        self.deletes.append(list(ids))
        for vector_id in ids:
            self.namespaces.get(namespace, {}).pop(vector_id, None)

    def list(self, *, prefix: str, namespace: str) -> Iterator[Any]:
        yield from ()

    def describe_index_stats(self) -> Any:
        return {}


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
