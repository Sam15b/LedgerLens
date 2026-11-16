"""Rebuild the derived stores from PostgreSQL, the system of record.

Pinecone vectors and the Neo4j graph are both projections of what Postgres holds, so
either can be thrown away and rebuilt from it (docs/layers/02-storage.md, "Definition of
done"):

* :func:`reindex_pinecone` -- re-embed every stored chunk and upsert it, with the same
  vector ids and metadata the write path produces; vectors of documents that are no
  longer in Postgres are pruned.

Only documents whose status is ``ready`` are rebuilt: a document that is still queued or
failed is retried by ingestion, which writes all three stores itself.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import InstrumentedAttribute, Session, sessionmaker

from ledgerlens.core.ids import parse_vector_id
from ledgerlens.core.schemas import DocType, DocumentStatus
from ledgerlens.ingestion.chunking import Chunk
from ledgerlens.ingestion.embedding import BM25Encoder, ChunkEmbedder, DenseEmbedder
from ledgerlens.storage.pinecone_store import DELETE_BATCH_SIZE, DocumentKeys, PineconeStore
from ledgerlens.storage.postgres.models import (
    ChunkRow,
    ContractRow,
    DocumentRow,
    InvoiceRow,
    PaymentRow,
    PurchaseOrderRow,
)

logger = logging.getLogger(__name__)

Progress = Callable[[int, int], None]
"""``progress(done, total)``, called after each document."""


def ready_document_ids(session: Session) -> list[str]:
    """Documents that finished ingestion, oldest first."""
    stmt = (
        select(DocumentRow.id)
        .where(DocumentRow.status == DocumentStatus.READY.value)
        .order_by(DocumentRow.created_at, DocumentRow.id)
    )
    return list(session.scalars(stmt))


# --------------------------------------------------------------------------- Pinecone


def row_to_chunk(row: ChunkRow, doc_type: DocType | str) -> Chunk:
    """A stored chunk as the embedder takes it. Only the embedded ``text`` is stored, so it
    stands in for the body (the body is not embedded, only the text is)."""
    return Chunk(
        index=row.chunk_index,
        doc_type=DocType(doc_type),
        kind=row.kind,  # type: ignore[arg-type]  # written from ChunkKind
        text=row.text,
        body=row.text,
        pages=list(row.pages) or [1],
        title=row.title,
    )


_KEY_COLUMNS: tuple[tuple[InstrumentedAttribute[Any], ...], ...] = (
    (
        InvoiceRow.document_id,
        InvoiceRow.vendor_id,
        InvoiceRow.invoice_number,
        InvoiceRow.invoice_date,
    ),
    (
        PurchaseOrderRow.document_id,
        PurchaseOrderRow.vendor_id,
        PurchaseOrderRow.po_number,
        PurchaseOrderRow.order_date,
    ),
    (
        PaymentRow.document_id,
        PaymentRow.vendor_id,
        PaymentRow.payment_reference,
        PaymentRow.paid_at,
    ),
    (
        ContractRow.document_id,
        ContractRow.vendor_id,
        ContractRow.contract_number,
        ContractRow.start_date,
    ),
)
"""Per record table: document id, vendor id, the record's number and its date."""


def document_keys(session: Session) -> dict[str, DocumentKeys]:
    """The vendor, number and date of the record each document produced -- what the write
    path copies into its chunks' metadata (``DocumentKeys.from_record``)."""
    keys: dict[str, DocumentKeys] = {}
    for columns in _KEY_COLUMNS:
        for document_id, vendor_id, number, day in session.execute(select(*columns)).tuples():
            if document_id is not None:
                keys[document_id] = DocumentKeys(
                    vendor_id=vendor_id, doc_number=number, doc_date=day
                )
    return keys


def fit_bm25(session: Session, document_ids: Iterable[str]) -> BM25Encoder:
    """BM25 statistics over the chunks of ``document_ids`` -- the corpus being indexed."""
    wanted = list(document_ids)
    texts: list[str] = []
    for start in range(0, len(wanted), 1000):
        batch = wanted[start : start + 1000]
        texts.extend(session.scalars(select(ChunkRow.text).where(ChunkRow.document_id.in_(batch))))
    return BM25Encoder().fit(texts)


@dataclass
class ReindexResult:
    documents: int = 0
    vectors: int = 0
    pruned: int = 0
    """Vectors deleted because their document is not ``ready`` in Postgres."""
    skipped: list[str] = field(default_factory=list)
    """Ready documents without chunks (nothing to index)."""


def reindex_pinecone(
    sessions: sessionmaker[Session],
    store: PineconeStore,
    dense: DenseEmbedder,
    *,
    sparse: BM25Encoder | None = None,
    prune: bool = True,
    progress: Progress | None = None,
) -> ReindexResult:
    """Re-embed the chunks of every ready document and replace its vectors.

    ``sparse`` is a fitted BM25 encoder; without one, BM25 is fit on the chunks being
    indexed (save it with :meth:`BM25Encoder.save` so queries use the same statistics).
    With ``prune``, vectors in the namespace whose document is not ready are deleted, so
    the namespace ends up holding exactly what Postgres says it should.
    """
    with sessions() as session:
        document_ids = ready_document_ids(session)
        doc_types = dict(
            session.execute(
                select(DocumentRow.id, DocumentRow.doc_type).where(DocumentRow.id.in_(document_ids))
            ).tuples()
        )
        keys = document_keys(session)
        encoder = sparse if sparse is not None else fit_bm25(session, document_ids)
    embedder = ChunkEmbedder(dense, encoder)
    result = ReindexResult()
    for done, document_id in enumerate(document_ids, start=1):
        with sessions() as session:
            rows = list(
                session.scalars(
                    select(ChunkRow)
                    .where(ChunkRow.document_id == document_id)
                    .order_by(ChunkRow.chunk_index)
                )
            )
        store.delete_document(document_id)
        if rows:
            chunks = [row_to_chunk(row, doc_types[document_id]) for row in rows]
            embedded = embedder.embed(document_id, chunks)
            result.vectors += store.upsert_chunks(embedded, keys=keys.get(document_id))
            result.documents += 1
        else:
            result.skipped.append(document_id)
        if progress is not None:
            progress(done, len(document_ids))
    if prune:
        result.pruned = prune_vectors(store, set(document_ids))
    return result


def prune_vectors(store: PineconeStore, keep: set[str]) -> int:
    """Delete every vector in the store's namespace whose document is not in ``keep``."""
    stale = [vector_id for vector_id in store.all_ids() if _document_of(vector_id) not in keep]
    for start in range(0, len(stale), DELETE_BATCH_SIZE):
        store.index.delete(ids=stale[start : start + DELETE_BATCH_SIZE], namespace=store.namespace)
    if stale:
        logger.info("pruned %d vectors of documents not ready in Postgres", len(stale))
    return len(stale)


def _document_of(vector_id: str) -> str | None:
    try:
        return parse_vector_id(vector_id)[0]
    except ValueError:  # not a chunk id: nothing the write path made, so it goes too
        return None
