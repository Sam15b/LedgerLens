"""Rebuild the derived stores from PostgreSQL, the system of record.

Pinecone vectors and the Neo4j graph are both projections of what Postgres holds, so
either can be thrown away and rebuilt from it (docs/layers/02-storage.md, "Definition of
done"):

* :func:`reindex_pinecone` -- re-embed every stored chunk and upsert it, with the same
  vector ids and metadata the write path produces; vectors of documents that are no
  longer in Postgres are pruned.
* :func:`rebuild_graph` -- MERGE the vendors (with their aliases) and every record back
  into Neo4j through the same :class:`~ledgerlens.storage.neo4j.linker.GraphLinker` the
  write path uses, so the edges and unresolved links come out the same. Linking order
  does not matter to the linker; contracts, POs, invoices, then payments is just the
  order that leaves the fewest unresolved links along the way.

Only documents whose status is ``ready`` are rebuilt: a document that is still queued or
failed is retried by ingestion, which writes all three stores itself.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from neo4j import Driver
from sqlalchemy import select
from sqlalchemy.orm import InstrumentedAttribute, Session, selectinload, sessionmaker

from ledgerlens.core.ids import parse_vector_id
from ledgerlens.core.schemas import (
    DocType,
    DocumentStatus,
    ExtractedRecord,
    Invoice,
    PurchaseOrder,
)
from ledgerlens.ingestion.chunking import Chunk
from ledgerlens.ingestion.embedding import BM25Encoder, ChunkEmbedder, DenseEmbedder
from ledgerlens.storage.neo4j.driver import apply_schema, run_read, run_write
from ledgerlens.storage.neo4j.linker import GraphLinker, VendorMatch
from ledgerlens.storage.pinecone_store import DELETE_BATCH_SIZE, DocumentKeys, PineconeStore
from ledgerlens.storage.postgres.models import (
    ChunkRow,
    ContractRow,
    DocumentRow,
    InvoiceRow,
    PaymentRow,
    PurchaseOrderRow,
)
from ledgerlens.storage.postgres.repositories import (
    VendorRepository,
    to_contract,
    to_invoice,
    to_payment,
    to_purchase_order,
    to_receipt,
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
        for document_id, vendor_id, number, day in session.execute(select(*columns)):
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
        typed = session.execute(
            select(DocumentRow.id, DocumentRow.doc_type).where(
                DocumentRow.status == DocumentStatus.READY.value
            )
        )
        doc_types = {row.id: row.doc_type for row in typed}
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


# --------------------------------------------------------------------------- Neo4j


@dataclass(frozen=True)
class GraphCounts:
    """Nodes per label and relationships per type."""

    nodes: dict[str, int]
    relationships: dict[str, int]


def graph_counts(driver: Driver, *, database: str | None = None) -> GraphCounts:
    nodes = run_read(
        driver,
        "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS n ORDER BY label",
        database=database,
    )
    rels = run_read(
        driver,
        "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n ORDER BY type",
        database=database,
    )
    return GraphCounts(
        nodes={row["label"]: row["n"] for row in nodes},
        relationships={row["type"]: row["n"] for row in rels},
    )


def wipe_graph(driver: Driver, *, database: str | None = None, batch_size: int = 10_000) -> int:
    """Delete every node and relationship, in batches. Returns how many nodes went."""
    removed = 0
    while True:
        rows = run_write(
            driver,
            "MATCH (n) WITH n LIMIT $limit DETACH DELETE n RETURN count(*) AS removed",
            {"limit": batch_size},
            database=database,
        )
        count = int(rows[0]["removed"]) if rows else 0
        removed += count
        if count < batch_size:
            return removed


@dataclass(frozen=True)
class StoredRecord:
    """A business record read back from Postgres, with the keys it was written under."""

    record: ExtractedRecord
    id: str
    document_id: str | None
    vendor_id: str | None


def stored_records(session: Session) -> list[StoredRecord]:
    """Every record of a ready document, plus records not from any document (imported or
    synthetic data): contracts, purchase orders, invoices (receipts as receipts), payments."""
    ready = set(ready_document_ids(session))
    receipts = set(
        session.scalars(select(DocumentRow.id).where(DocumentRow.doc_type == DocType.RECEIPT.value))
    )

    def invoice(row: InvoiceRow) -> ExtractedRecord:
        return to_receipt(row) if row.document_id in receipts else to_invoice(row)

    pairs: list[tuple[RecordRow, ExtractedRecord]] = []
    pairs += [
        (row, to_contract(row))
        for row in session.scalars(select(ContractRow).order_by(ContractRow.id))
    ]
    pairs += [
        (row, to_purchase_order(row))
        for row in session.scalars(
            select(PurchaseOrderRow)
            .options(selectinload(PurchaseOrderRow.line_items))
            .order_by(PurchaseOrderRow.id)
        )
    ]
    pairs += [
        (row, invoice(row))
        for row in session.scalars(
            select(InvoiceRow).options(selectinload(InvoiceRow.line_items)).order_by(InvoiceRow.id)
        )
    ]
    pairs += [
        (row, to_payment(row))
        for row in session.scalars(
            select(PaymentRow).options(selectinload(PaymentRow.allocations)).order_by(PaymentRow.id)
        )
    ]
    return [
        StoredRecord(record, row.id, row.document_id, row.vendor_id)
        for row, record in pairs
        if row.document_id is None or row.document_id in ready
    ]


RecordRow = ContractRow | PurchaseOrderRow | InvoiceRow | PaymentRow


def _tax_id(record: ExtractedRecord) -> str | None:
    return record.vendor_tax_id if isinstance(record, Invoice | PurchaseOrder) else None


@dataclass
class GraphRebuildResult:
    wiped: int = 0
    vendors: int = 0
    records: dict[str, int] = field(default_factory=dict)
    """Records linked, by node label."""
    unresolved: int = 0
    """References that found no target (stored as ``UnresolvedLink`` nodes)."""


def rebuild_graph(
    sessions: sessionmaker[Session],
    linker: GraphLinker,
    *,
    wipe: bool = False,
    progress: Progress | None = None,
) -> GraphRebuildResult:
    """MERGE every vendor and record in Postgres into the graph.

    Idempotent: on a graph that already holds them nothing changes. With ``wipe`` the
    graph is emptied first, so it ends up holding exactly what Postgres says.
    """
    driver, database = linker.driver, linker.database
    result = GraphRebuildResult()
    if wipe:
        result.wiped = wipe_graph(driver, database=database)
    apply_schema(driver, database=database)
    with sessions() as session:
        vendors = VendorRepository(session).all()
        records = stored_records(session)
    for vendor in vendors:
        linker.resolver.upsert(
            vendor.id, vendor.canonical_name, tax_id=vendor.tax_id, aliases=vendor.aliases
        )
    result.vendors = len(vendors)
    for done, stored in enumerate(records, start=1):
        record = stored.record
        if stored.vendor_id is not None:
            # The printed spelling becomes an alias, as it did when the record was ingested.
            given = VendorMatch(stored.vendor_id, "given", 100.0)
            linker.resolver.remember(given, record.vendor_name, _tax_id(record))
        linked = linker.link(
            record, id=stored.id, document_id=stored.document_id, vendor_id=stored.vendor_id
        )
        result.records[linked.label] = result.records.get(linked.label, 0) + 1
        if progress is not None:
            progress(done, len(records))
    result.unresolved = len(linker.unresolved())
    return result
