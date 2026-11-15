"""PineconeStore against the Pinecone Local emulator (``PINECONE_HOST``).

    docker compose up -d --wait pinecone-local
    PINECONE_HOST=http://localhost:5080 pytest -m integration tests/integration/storage

Each module run creates its own small index and deletes it afterwards. The emulator is
dense-only: it accepts sparse values but neither stores nor scores them.
"""

import uuid
from collections.abc import Iterator
from datetime import date

import pytest

from ledgerlens.core.config import PineconeConfig, get_settings
from ledgerlens.core.ids import chunk_vector_id, new_document_id
from ledgerlens.ingestion.embedding import EmbeddedChunk, SparseVector
from ledgerlens.storage.pinecone_store import (
    DocumentKeys,
    LocalIndexAdmin,
    PineconeStore,
    create_client,
    date_filter,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not get_settings().pinecone_host,
        reason="creates and deletes an index: only run against Pinecone Local (PINECONE_HOST)",
    ),
]

DIM = 8
CONFIG = PineconeConfig(dimension=DIM, upsert_batch_size=100, metadata_text_max_bytes=32)


def unit(axis: int) -> list[float]:
    return [1.0 if i == axis else 0.0 for i in range(DIM)]


def chunk(
    document_id: str, n: int, *, axis: int = 0, terms: list[int] | None = None
) -> EmbeddedChunk:
    terms = terms or []
    return EmbeddedChunk(
        id=chunk_vector_id(document_id, n),
        values=unit(axis),
        sparse=SparseVector(indices=terms, values=[1.0] * len(terms)),
        metadata={
            "document_id": document_id,
            "chunk_index": n,
            "doc_type": "invoice",
            "kind": "body",
            "pages": ["1"],
            "text": f"chunk {n} of {document_id} " + "x" * 64,
        },
    )


@pytest.fixture(scope="module")
def pinecone() -> Iterator[PineconeStore]:
    settings = get_settings()
    admin = LocalIndexAdmin(settings.pinecone_host)
    name = f"it-store-{uuid.uuid4().hex[:8]}"
    info = admin.create(name, dimension=DIM, metric="dotproduct", cloud="aws", region="us-east-1")
    assert info.host
    try:
        index = create_client(settings.pinecone_api_key, settings.pinecone_host).Index(
            host=info.host
        )
        yield PineconeStore(index, namespace="eval", config=CONFIG)  # type: ignore[arg-type]
    finally:
        admin.delete(name)


@pytest.fixture
def store(pinecone: PineconeStore) -> Iterator[PineconeStore]:
    yield pinecone
    for namespace in pinecone.stats()["namespaces"]:  # empty the index between tests
        pinecone.index.delete(delete_all=True, namespace=namespace)  # type: ignore[call-arg]


def test_upsert_in_batches_then_reupsert_is_idempotent(store: PineconeStore) -> None:
    document_id = new_document_id()
    chunks = [chunk(document_id, n) for n in range(250)]

    assert store.upsert_chunks(chunks) == 250
    assert store.upsert_chunks(chunks) == 250  # same ids: overwritten, not duplicated

    assert store.count() == 250
    assert len(store.document_ids(document_id)) == 250


def test_metadata_is_mapped_and_filterable(store: PineconeStore) -> None:
    march, april = new_document_id(), new_document_id()
    store.upsert_chunks(
        [chunk(march, 0)],
        keys=DocumentKeys(vendor_id="V001", doc_number="INV-3", doc_date=date(2026, 3, 14)),
    )
    store.upsert_chunks(
        [chunk(april, 0)],
        keys=DocumentKeys(vendor_id="V002", doc_number="INV-4", doc_date=date(2026, 4, 2)),
    )

    hits = store.query(unit(0), top_k=5, filter=date_filter(date(2026, 4, 1)))

    assert [h.document_id for h in hits] == [april]
    assert hits[0].metadata == {
        "document_id": april,
        "doc_type": "invoice",
        "kind": "body",
        "chunk_index": 0,
        "page": 1,
        "vendor_id": "V002",
        "doc_number": "INV-4",
        "doc_date": 20260402,
        "text": f"chunk 0 of {april}"[:32],
    }
    by_vendor = store.query(unit(0), top_k=5, filter={"vendor_id": "V001"})
    assert [h.document_id for h in by_vendor] == [march]


def test_hybrid_query_scales_the_dense_score_by_alpha(store: PineconeStore) -> None:
    """Pinecone Local accepts sparse values but does not store or score them (a fetched
    vector comes back with ``sparse_values=None``), so here only the dense side of the
    weighting is observable. The sparse side is covered by the unit tests; ranking with
    real sparse scores is checked on the serverless index in eval runs."""
    document_id = new_document_id()
    semantic = chunk(document_id, 0, axis=0)
    keyword = chunk(document_id, 1, axis=1, terms=[42])
    store.upsert_chunks([semantic, keyword])
    sparse = SparseVector(indices=[42], values=[3.0])

    for alpha in (0.9, 0.3):
        hits = store.query(unit(0), sparse, top_k=2, alpha=alpha)
        assert hits[0].id == semantic.id
        assert hits[0].score == pytest.approx(alpha)  # dense dot product 1.0, times alpha


def test_delete_document_removes_only_its_vectors(store: PineconeStore) -> None:
    gone, kept = new_document_id(), new_document_id()
    store.upsert_chunks([chunk(gone, n) for n in range(3)] + [chunk(kept, 0)])

    assert store.delete_document(gone) == 3
    assert store.delete_document(gone) == 0

    assert store.document_ids(gone) == []
    assert store.document_ids(kept) == [chunk_vector_id(kept, 0)]
    assert [h.document_id for h in store.query(unit(0), top_k=10)] == [kept]


def test_namespaces_are_isolated(store: PineconeStore) -> None:
    document_id = new_document_id()
    store.upsert_chunks([chunk(document_id, 0)])
    store.upsert_chunks([chunk(document_id, 0), chunk(document_id, 1)], namespace="dev")

    assert store.delete_document(document_id) == 1  # the store's own namespace: eval
    assert store.stats()["namespaces"].get("dev") == 2
    assert store.query(unit(0), top_k=5) == []
    assert len(store.query(unit(0), top_k=5, namespace="dev")) == 2
