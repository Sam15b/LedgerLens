"""Idempotent reads and writes on the PostgreSQL system of record.

Every write is an upsert -- ``INSERT ... ON CONFLICT DO UPDATE`` keyed by the row's id --
whose update only fires when a value actually changed. Writing the same record twice
leaves one row, untouched by the second write (``updated_at`` included), so a re-run of
ingestion is safe at any point.

Repositories wrap a :class:`~sqlalchemy.orm.Session` and never commit: the caller owns
the transaction, usually with :func:`~ledgerlens.storage.postgres.db.session_scope`::

    with session_scope() as session:
        documents = DocumentRepository(session)
        documents.upsert(document)
        ChunkRepository(session).replace(document.id, chunks)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from sqlalchemy import Table, delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from ledgerlens.core.schemas import DocType, Document, DocumentStatus
from ledgerlens.storage.postgres.models import Base, ChunkRow, DocumentRow

if TYPE_CHECKING:
    from ledgerlens.ingestion.chunking import Chunk


def upsert(
    session: Session,
    model: type[Base],
    rows: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    conflict: Sequence[str] = ("id",),
    keep: Iterable[str] = (),
) -> None:
    """``INSERT rows ON CONFLICT (conflict) DO UPDATE`` the other columns given.

    Columns in ``keep`` are written on insert but never overwritten (e.g. a workflow
    status). The update only runs for rows whose values differ, so an identical upsert
    is a no-op and does not bump ``updated_at``.
    """
    batch = [rows] if isinstance(rows, Mapping) else list(rows)
    if not batch:
        return
    table: Table = model.__table__  # type: ignore[assignment]
    stmt = pg_insert(table).values(batch)
    skip = {*conflict, *keep, "created_at"}
    columns = [c for c in batch[0] if c not in skip]
    if not columns:
        session.execute(stmt.on_conflict_do_nothing(index_elements=list(conflict)))
        return
    changes: dict[str, Any] = {c: stmt.excluded[c] for c in columns}
    if "updated_at" in table.c:
        changes["updated_at"] = func.now()
    changed = or_(*(table.c[c].is_distinct_from(stmt.excluded[c]) for c in columns))
    session.execute(
        stmt.on_conflict_do_update(index_elements=list(conflict), set_=changes, where=changed)
    )


def clean_text(text: str) -> str:
    """PostgreSQL ``TEXT`` cannot hold NUL bytes, which OCR output occasionally contains."""
    return text.replace("\x00", "")


# --------------------------------------------------------------------------- documents


def document_values(document: Document) -> dict[str, Any]:
    return {
        "id": document.id,
        "sha256": document.sha256,
        "filename": document.filename,
        "mime_type": document.mime_type,
        "doc_type": document.doc_type.value,
        "status": document.status.value,
        "error": document.error,
        "pages": document.pages,
        "ocr_engine": document.ocr_engine,
        "created_at": document.created_at,
    }


def to_document(row: DocumentRow) -> Document:
    return Document(
        id=row.id,
        sha256=row.sha256,
        filename=row.filename,
        mime_type=row.mime_type,
        doc_type=DocType(row.doc_type),
        status=DocumentStatus(row.status),
        error=row.error,
        pages=row.pages,
        ocr_engine=row.ocr_engine,  # type: ignore[arg-type]  # CHECKed in the database
        created_at=row.created_at,
    )


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert(self, document: Document) -> None:
        """Insert or update the document by ``id``; ``created_at`` keeps its first value."""
        upsert(self.session, DocumentRow, document_values(document))

    def claim(self, document: Document) -> str | None:
        """Insert ``document`` unless its ``sha256`` is already stored.

        Returns ``None`` if this call inserted it, else the ``id`` of the document that
        already has these bytes. The unique index on ``sha256`` makes check-and-insert
        atomic, so two concurrent uploads of the same file cannot both win.
        """
        table: Table = DocumentRow.__table__  # type: ignore[assignment]
        stmt = (
            pg_insert(table)
            .values(document_values(document))
            .on_conflict_do_nothing(index_elements=["sha256"])
            .returning(table.c.id)
        )
        if self.session.execute(stmt).scalar_one_or_none() is not None:
            return None
        existing = self.id_for_sha256(document.sha256)
        if existing is None:  # pragma: no cover - the conflicting row was deleted meanwhile
            raise RuntimeError(f"sha256 {document.sha256} conflicted but no row has it")
        return existing

    def get(self, document_id: str) -> Document | None:
        row = self.session.get(DocumentRow, document_id)
        return None if row is None else to_document(row)

    def get_by_sha256(self, sha256: str) -> Document | None:
        row = self.session.scalars(
            select(DocumentRow).where(DocumentRow.sha256 == sha256.lower())
        ).one_or_none()
        return None if row is None else to_document(row)

    def id_for_sha256(self, sha256: str) -> str | None:
        return self.session.scalars(
            select(DocumentRow.id).where(DocumentRow.sha256 == sha256.lower())
        ).one_or_none()

    def set_status(
        self, document_id: str, status: DocumentStatus, *, error: str | None = None
    ) -> bool:
        """Move a document to ``status`` (``error`` is kept only for ``failed``).

        Returns whether the document exists.
        """
        stmt = (
            update(DocumentRow)
            .where(DocumentRow.id == document_id)
            .values(
                status=status.value,
                error=error if status is DocumentStatus.FAILED else None,
                updated_at=func.now(),
            )
        )
        return bool(self.session.execute(stmt).rowcount)  # type: ignore[attr-defined]

    def list_by_status(self, *statuses: DocumentStatus) -> list[Document]:
        rows = self.session.scalars(
            select(DocumentRow)
            .where(DocumentRow.status.in_([s.value for s in statuses]))
            .order_by(DocumentRow.created_at)
        )
        return [to_document(row) for row in rows]

    def delete(self, document_id: str) -> bool:
        """Delete a document; its chunks and extracted records cascade with it."""
        result = self.session.execute(delete(DocumentRow).where(DocumentRow.id == document_id))
        return bool(result.rowcount)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- chunks


class ChunkRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def replace(
        self,
        document_id: str,
        chunks: Sequence[Chunk],
        *,
        token_counts: Sequence[int | None] | None = None,
    ) -> int:
        """Make ``chunks`` the document's chunks: upsert each by its vector id and delete
        any left over from a previous, longer version. Returns how many there are now."""
        if token_counts is not None and len(token_counts) != len(chunks):
            raise ValueError("token_counts must have one entry per chunk")
        counts = token_counts if token_counts is not None else [None] * len(chunks)
        upsert(
            self.session,
            ChunkRow,
            [
                {
                    "id": chunk.vector_id(document_id),
                    "document_id": document_id,
                    "chunk_index": chunk.index,
                    "kind": chunk.kind,
                    "title": chunk.title,
                    "text": clean_text(chunk.text),
                    "token_count": count,
                    "pages": list(chunk.pages),
                }
                for chunk, count in zip(chunks, counts, strict=True)
            ],
        )
        keep = [chunk.vector_id(document_id) for chunk in chunks]
        self.session.execute(
            delete(ChunkRow).where(ChunkRow.document_id == document_id, ChunkRow.id.not_in(keep))
        )
        return len(chunks)

    def for_document(self, document_id: str) -> list[ChunkRow]:
        return list(
            self.session.scalars(
                select(ChunkRow)
                .where(ChunkRow.document_id == document_id)
                .order_by(ChunkRow.chunk_index)
            )
        )

    def texts(self, ids: Iterable[str]) -> dict[str, str]:
        """Full text by vector id -- Pinecone hits carry only a truncated copy."""
        wanted = list(dict.fromkeys(ids))
        if not wanted:
            return {}
        rows = self.session.execute(
            select(ChunkRow.id, ChunkRow.text).where(ChunkRow.id.in_(wanted))
        )
        return {row.id: row.text for row in rows}
