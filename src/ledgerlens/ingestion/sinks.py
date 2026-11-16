"""Write an ingested document to all three stores, safely and repeatably.

:class:`StoresSink` is the production :class:`~ledgerlens.ingestion.pipeline.StorageSink`.
For one document it runs, in order (docs/layers/02-storage.md, "Write path"):

0. **vendor** -- resolve the printed vendor against the graph (read only), so Postgres and
   Neo4j agree on its id;
1. **postgres** -- one transaction: the document row (``processing``), its chunks, the
   vendor row if new, and the extracted record with its line items;
2. **pinecone** -- delete every vector of the document, then upsert its chunks;
3. **neo4j** -- remember the vendor (new node / alias), MERGE the record and its edges;
4. mark the document ``ready``.

Every step is idempotent (upserts, delete-then-upsert, MERGE), so running the write again
-- after a crash, a failure, or for a document that was already stored -- converges on the
same final state. A failure at any step marks the document ``failed`` with the step and
the error; nothing is rolled back across stores, because the retry overwrites it all.

If re-extraction changed a document's type, records of its old type are removed from
Postgres and the graph before the new one is written.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from ledgerlens.core.schemas import (
    Document,
    DocumentStatus,
    ExtractedRecord,
    Invoice,
    PurchaseOrder,
)
from ledgerlens.storage.neo4j.linker import GraphLinker, VendorMatch, record_label
from ledgerlens.storage.pinecone_store import DocumentKeys, PineconeStore
from ledgerlens.storage.postgres.db import get_session_factory, session_scope
from ledgerlens.storage.postgres.repositories import (
    DocumentRepository,
    Repositories,
    record_model,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session, sessionmaker

    from ledgerlens.ingestion.pipeline import IngestResult

logger = logging.getLogger(__name__)

T = TypeVar("T")


class StoreWriteError(RuntimeError):
    """A write step failed; ``step`` names it (``vendor``, ``postgres``, ...)."""

    def __init__(self, step: str, cause: BaseException) -> None:
        super().__init__(f"{step}: {type(cause).__name__}: {cause}")
        self.step = step


def _tax_id(record: ExtractedRecord) -> str | None:
    return record.vendor_tax_id if isinstance(record, Invoice | PurchaseOrder) else None


class StoresSink:
    """Postgres + Pinecone + Neo4j. ``pinecone`` / ``linker`` may be ``None`` to skip a
    store (tests, or a deployment without the graph)."""

    def __init__(
        self,
        *,
        sessions: sessionmaker[Session] | None = None,
        pinecone: PineconeStore | None = None,
        linker: GraphLinker | None = None,
    ) -> None:
        self.sessions = sessions or get_session_factory()
        self.pinecone = pinecone
        self.linker = linker

    @classmethod
    def from_settings(cls) -> StoresSink:
        """All three stores from ``DATABASE_URL``, ``PINECONE_*`` and ``NEO4J_*``."""
        from ledgerlens.storage.neo4j.driver import get_driver

        return cls(pinecone=PineconeStore.from_settings(), linker=GraphLinker(get_driver()))

    # ------------------------------------------------------------------ StorageSink

    def write(self, result: IngestResult) -> Document:
        document = result.document
        if result.duplicate:
            return self.get(document.id) or document
        if document.status is DocumentStatus.FAILED:  # a pipeline step failed: record it
            self.mark(document)
            return document
        try:
            self._write(result)
        except StoreWriteError as exc:
            logger.warning("storing %s failed at %s", document.id, exc)
            failed = document.with_status(DocumentStatus.FAILED, error=str(exc))
            self.mark(failed)
            return failed
        ready = document.with_status(DocumentStatus.READY)
        self.mark(ready)
        return ready

    def close(self) -> None:
        """Nothing to release: the engine, index client and driver are shared."""

    # ------------------------------------------------------------------ status

    def mark(self, document: Document) -> bool:
        """Store ``document`` with its current status. A failure here is logged, not
        raised: if Postgres is down there is nowhere to record it, and the document is
        retried on the next run either way (it is not ``ready``)."""
        try:
            with session_scope(self.sessions) as session:
                DocumentRepository(session).upsert(document)
        except Exception:
            logger.exception("could not record %s as %s", document.id, document.status.value)
            return False
        return True

    def get(self, document_id: str) -> Document | None:
        with self.sessions() as session:
            return DocumentRepository(session).get(document_id)

    # ------------------------------------------------------------------ the write path

    def _write(self, result: IngestResult) -> None:
        document = result.document
        record = result.extraction.record if result.extraction else None
        record_id = self.record_id(result)
        vendor = self._step("vendor", lambda: self.resolve_vendor(result, record))
        vendor_id = vendor.vendor_id if vendor else None
        self._step("postgres", lambda: self._postgres(result, record, record_id, vendor_id))
        if self.pinecone is not None:
            pinecone = self.pinecone
            self._step("pinecone", lambda: self._vectors(pinecone, result, record, vendor_id))
        if self.linker is not None:
            linker = self.linker
            self._step("neo4j", lambda: self._graph(linker, document, record, record_id, vendor))

    @staticmethod
    def _step(name: str, run: Callable[[], T]) -> T:
        try:
            return run()
        except Exception as exc:
            raise StoreWriteError(name, exc) from exc

    # Two hooks a subclass can override when it knows more than the document says (a
    # bulk load of records whose ids and vendors are already known).

    def record_id(self, result: IngestResult) -> str:
        """The id the document's record is stored under: the ``document_id``."""
        return result.document.id

    def resolve_vendor(
        self, result: IngestResult, record: ExtractedRecord | None
    ) -> VendorMatch | None:
        """Which vendor the record is from: resolved against the graph (read only)."""
        if record is None or self.linker is None:
            return None
        return self.linker.resolver.resolve(record.vendor_name, _tax_id(record))

    def _postgres(
        self,
        result: IngestResult,
        record: ExtractedRecord | None,
        record_id: str,
        vendor_id: str | None,
    ) -> None:
        document = result.document
        with session_scope(self.sessions) as session:
            repos = Repositories(session)
            repos.documents.upsert(document.with_status(DocumentStatus.PROCESSING))
            repos.chunks.replace(document.id, result.chunks)
            repos.delete_records(document.id, keep=record_model(record) if record else None)
            if record is None:
                return
            if vendor_id is not None:
                repos.vendors.ensure(vendor_id, record.vendor_name, tax_id=_tax_id(record))
            repos.upsert_record(record, id=record_id, document_id=document.id, vendor_id=vendor_id)

    @staticmethod
    def _vectors(
        pinecone: PineconeStore,
        result: IngestResult,
        record: ExtractedRecord | None,
        vendor_id: str | None,
    ) -> None:
        document_id = result.document.id
        if result.chunks and len(result.embedded) != len(result.chunks):
            raise ValueError(
                f"{len(result.chunks)} chunks but {len(result.embedded)} embedded: "
                "embed before storing (is the BM25 encoder fitted?)"
            )
        pinecone.delete_document(document_id)
        keys = DocumentKeys.from_record(record, vendor_id=vendor_id)
        pinecone.upsert_chunks(result.embedded, keys=keys)

    @staticmethod
    def _graph(
        linker: GraphLinker,
        document: Document,
        record: ExtractedRecord | None,
        record_id: str,
        vendor: VendorMatch | None,
    ) -> None:
        linker.unlink_document(document.id, keep=record_label(record) if record else None)
        if record is None or vendor is None:
            return
        linker.resolver.remember(vendor, record.vendor_name, _tax_id(record))
        linker.link(record, id=record_id, document_id=document.id, vendor_id=vendor.vendor_id)
