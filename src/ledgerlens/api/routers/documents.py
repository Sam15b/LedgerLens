"""Documents: upload a file for ingestion, then follow it and read what was extracted.

* ``POST /documents`` (multipart, field ``file``) -- the loader hashes and sniffs the
  upload; the same bytes uploaded before (under any name) are a **duplicate**: 200 with the
  existing ``document_id`` and its status, nothing queued. A new file is claimed (its
  ``queued`` row is inserted), kept under ``loader.upload_dir`` and queued for the worker:
  202 with the ``document_id`` and the job id. A file whose earlier run never reached
  ``ready`` is queued again under its old id.
* ``GET /documents`` -- newest first, filtered by ``status`` / ``doc_type``, paged with
  ``limit`` / ``offset``;
* ``GET /documents/{id}`` -- the document, its status (and ``error`` if it failed), how
  many chunks it was split into, the record extracted from it (``record_type``,
  ``record_id`` and the typed fields, line items included) once it is stored, and
  whether its original file is kept (``has_file``);
* ``GET /documents/{id}/file`` -- that original file, served inline with its sniffed
  type (the dashboard shows it next to the extracted fields). Only files uploaded here
  are kept; a document ingested from a folder by the CLI has none (404).

The raw file is kept at ``<upload_dir>/<document_id>/<original name>``: the worker ingests
it by path (so API and worker must share the folder) and the document keeps its original
file name. If the queue is down the document stays ``queued`` and the response is 503;
uploading the same file again queues it.

Limits (``loader`` in ``configs/ingestion.yaml``): a body over ``max_file_mb`` is refused
with 413 before it is read (:class:`UploadLimitRoute`); the type is sniffed from the bytes
and must be one of ``allowed_mime_types`` (415 otherwise); an empty, corrupt or
password-protected file is 422.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Awaitable, Callable, Coroutine
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.types import Message

from ledgerlens.core.config import get_config, repo_path
from ledgerlens.core.ids import is_document_id
from ledgerlens.core.schemas import DocType, Document, DocumentStatus
from ledgerlens.ingestion.loaders import (
    DedupeRegistry,
    FileLoader,
    FileTooLargeError,
    LoaderError,
    LoadResult,
    UnsupportedFileError,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)

_UNSAFE = re.compile(r"[^A-Za-z0-9._ -]+")
MAX_NAME_CHARS = 150
MULTIPART_OVERHEAD_BYTES = 64 * 1024
"""Room for the multipart boundaries and part headers around the file itself."""


# --------------------------------------------------------------------------- limits


def max_upload_bytes() -> int:
    return int(get_config().ingestion.loader.max_file_mb * 1024 * 1024)


def too_large(limit: int) -> HTTPException:
    mb = limit / 1024 / 1024
    return HTTPException(
        status.HTTP_413_CONTENT_TOO_LARGE, f"the upload is over the {mb:g} MB limit"
    )


class UploadLimitRoute(APIRoute):
    """Refuses a request body over the upload limit before it is parsed.

    FastAPI reads the whole multipart body (spooling the file to disk) before the endpoint
    runs, so the limit is enforced here: a declared ``Content-Length`` over it is refused
    at once, and a body sent without one is counted as it arrives and cut off at the limit.
    """

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def limited(request: Request) -> Response:
            if request.method != "POST":
                return await handler(request)
            file_limit = max_upload_bytes()
            limit = file_limit + MULTIPART_OVERHEAD_BYTES
            declared = request.headers.get("content-length", "")
            if declared.isdigit() and int(declared) > limit:
                raise too_large(file_limit)
            received = 0
            receive = request.receive

            async def counted() -> Message:
                nonlocal received
                message = await receive()
                received += len(message.get("body", b""))
                if received > limit:
                    raise too_large(file_limit)
                return message

            return await handler(Request(request.scope, counted))

        return limited


router = APIRouter(prefix="/documents", tags=["documents"], route_class=UploadLimitRoute)


class UploadOut(BaseModel):
    document_id: str
    status: DocumentStatus
    duplicate: bool
    """The same bytes were uploaded before: ``document_id`` is that document's."""
    job_id: str | None = None
    """The ingestion job queued for this upload (``None`` for a duplicate)."""


class DocumentOut(BaseModel):
    id: str
    filename: str
    mime_type: str | None = None
    doc_type: DocType
    status: DocumentStatus
    error: str | None = None
    """Why the last run failed (``failed`` only)."""
    pages: int | None = None
    ocr_engine: str | None = None
    sha256: str
    created_at: datetime

    @classmethod
    def of(cls, document: Document) -> DocumentOut:
        return cls.model_validate(document.model_dump())


class DocumentDetail(DocumentOut):
    chunks: int = 0
    record_type: str | None = None
    """``invoice``, ``receipt``, ``purchase_order``, ``payment`` or ``contract``."""
    record_id: str | None = None
    record: dict[str, Any] | None = None
    """The extracted fields, as stored (``None`` until the document is stored)."""
    has_file: bool = False
    """The uploaded file is kept: ``GET /documents/{id}/file`` serves it."""


# --------------------------------------------------------------------------- raw files


def safe_filename(name: str | None) -> str:
    """The upload's base name with anything but letters, digits, ``._ -`` replaced."""
    base = re.split(r"[\\/]", name or "")[-1]
    cleaned = _UNSAFE.sub("_", base).strip(" .")
    return cleaned[-MAX_NAME_CHARS:] or "upload"


class RawFileStore:
    """Uploaded files on a shared folder: ``<root>/<document_id>/<filename>``."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def save(self, document_id: str, filename: str, data: bytes) -> Path:
        """Write the file (replacing an earlier copy of the document) and return its path."""
        folder = self.root / document_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / filename
        partial = folder / f".{filename}.part"
        partial.write_bytes(data)
        os.replace(partial, path)  # the worker never sees half a file
        return path.resolve()

    def find(self, document_id: str, filename: str) -> Path | None:
        """The kept upload of a document, or ``None`` (never uploaded here, or removed).

        ``filename`` comes from the database, but a name that would leave the document's
        folder is refused all the same.
        """
        folder = (self.root / document_id).resolve()
        path = (folder / filename).resolve()
        if path.parent != folder or not path.is_file():
            return None
        return path


# --------------------------------------------------------------------------- dependencies

Enqueue = Callable[[Path], Awaitable[str | None]]


async def enqueue_ingestion(path: Path) -> str | None:
    """Queue one ingestion job for ``path`` (arq on ``REDIS_URL``); returns the job id."""
    from ledgerlens.workers.ingest_worker import enqueue

    ids = await enqueue([path])
    return ids[0] if ids else None


def get_enqueue() -> Enqueue:
    return enqueue_ingestion


@lru_cache
def get_registry() -> DedupeRegistry:
    """Dedupe on ``documents.sha256`` (a won claim inserts the ``queued`` row)."""
    from ledgerlens.ingestion.loaders import PostgresRegistry

    return PostgresRegistry()


def get_raw_store() -> RawFileStore:
    return RawFileStore(repo_path(get_config().ingestion.loader.upload_dir))


class DocumentStore:
    """Reads documents and what was extracted from them, from Postgres."""

    def __init__(self, sessions: sessionmaker[Session] | None = None) -> None:
        self._sessions = sessions

    @property
    def sessions(self) -> sessionmaker[Session]:
        from ledgerlens.storage.postgres.db import get_session_factory

        return self._sessions or get_session_factory()

    def get(self, document_id: str) -> Document | None:
        from ledgerlens.storage.postgres.repositories import DocumentRepository

        with self.sessions() as session:
            return DocumentRepository(session).get(document_id)

    def list(
        self,
        *,
        status: DocumentStatus | None = None,
        doc_type: DocType | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Document]:
        from ledgerlens.storage.postgres.repositories import DocumentRepository

        with self.sessions() as session:
            return DocumentRepository(session).find(
                status=status, doc_type=doc_type, limit=limit, offset=offset
            )

    def detail(self, document_id: str) -> DocumentDetail | None:
        from ledgerlens.storage.postgres.repositories import Repositories

        with self.sessions() as session:
            repos = Repositories(session)
            document = repos.documents.get(document_id)
            if document is None:
                return None
            found = repos.record_for_document(document_id, document.doc_type)
            detail = DocumentDetail.model_validate(
                DocumentOut.of(document).model_dump() | {"chunks": repos.chunks.count(document_id)}
            )
        if found is not None:
            record_id, record = found
            detail.record_type = record.doc_type
            detail.record_id = record_id
            detail.record = record.model_dump(mode="json", exclude={"doc_type"})
        return detail


def get_document_store() -> DocumentStore:
    return DocumentStore()


Registry = Annotated[DedupeRegistry, Depends(get_registry)]
RawStore = Annotated[RawFileStore, Depends(get_raw_store)]
Store = Annotated[DocumentStore, Depends(get_document_store)]
Queue = Annotated[Enqueue, Depends(get_enqueue)]


# --------------------------------------------------------------------------- upload


def loader_error_status(exc: LoaderError) -> int:
    """413 over the size limit, 415 for a type ingestion does not take (sniffed from the
    bytes, whatever the name or declared type says), 422 for an empty or unreadable file."""
    if isinstance(exc, FileTooLargeError):
        return status.HTTP_413_CONTENT_TOO_LARGE
    if isinstance(exc, UnsupportedFileError):
        return status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    return status.HTTP_422_UNPROCESSABLE_CONTENT


def load_upload(registry: DedupeRegistry, data: bytes, filename: str) -> LoadResult:
    """Inspect the bytes and claim their hash (blocking: hashing, page count, Postgres)."""
    return FileLoader(get_config().ingestion.loader, registry=registry).load_bytes(data, filename)


@router.post(
    "",
    response_model=UploadOut,
    status_code=status.HTTP_202_ACCEPTED,
    responses={200: {"model": UploadOut, "description": "Already uploaded: nothing queued"}},
)
async def upload_document(
    file: UploadFile,
    response: Response,
    registry: Registry,
    raw: RawStore,
    store: Store,
    enqueue: Queue,
) -> UploadOut:
    """Upload a PDF or image; it is ingested in the background (poll ``GET
    /documents/{id}`` until ``ready`` or ``failed``)."""
    filename = safe_filename(file.filename)
    data = await file.read()
    try:
        loaded = await run_in_threadpool(load_upload, registry, data, filename)
    except LoaderError as exc:
        raise HTTPException(loader_error_status(exc), str(exc)) from exc

    if loaded.duplicate:
        existing = await run_in_threadpool(store.get, loaded.document_id)
        response.status_code = status.HTTP_200_OK
        return UploadOut(
            document_id=loaded.document_id,
            status=existing.status if existing else DocumentStatus.QUEUED,
            duplicate=True,
        )

    path = await run_in_threadpool(raw.save, loaded.document_id, filename, data)
    try:
        job_id = await enqueue(path)
    except Exception as exc:
        logger.warning("could not queue %s: %s", loaded.document_id, exc)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"document {loaded.document_id} is stored but the ingestion queue is unavailable; "
            "upload the file again to queue it",
        ) from exc
    log: dict[str, Any] = {"document_id": loaded.document_id, "job_id": job_id}
    logger.info("queued %s%s", filename, " (retry)" if loaded.retry else "", extra=log)
    return UploadOut(
        document_id=loaded.document_id,
        status=DocumentStatus.QUEUED,
        duplicate=False,
        job_id=job_id,
    )


# --------------------------------------------------------------------------- read


@router.get("", response_model=list[DocumentOut])
async def list_documents(
    store: Store,
    status_filter: Annotated[DocumentStatus | None, Query(alias="status")] = None,
    doc_type: DocType | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[DocumentOut]:
    """Documents, newest first."""
    documents = await run_in_threadpool(
        lambda: store.list(status=status_filter, doc_type=doc_type, limit=limit, offset=offset)
    )
    return [DocumentOut.of(document) for document in documents]


def not_found(document_id: str) -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, f"no document {document_id!r}")


@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(document_id: str, store: Store, raw: RawStore) -> DocumentDetail:
    """One document with its status and the fields extracted from it."""
    detail = None
    if is_document_id(document_id.lower()):
        detail = await run_in_threadpool(store.detail, document_id.lower())
    if detail is None:
        raise not_found(document_id)
    detail.has_file = await run_in_threadpool(raw.find, detail.id, detail.filename) is not None
    return detail


FILE_TYPES: dict[str, dict[str, Any]] = {
    t: {} for t in ("application/pdf", "image/png", "image/jpeg", "image/tiff")
}


@router.get(
    "/{document_id}/file",
    response_class=FileResponse,
    responses={200: {"content": FILE_TYPES, "description": "The uploaded file"}},
)
async def get_document_file(document_id: str, store: Store, raw: RawStore) -> FileResponse:
    """The document's original file, as uploaded (404 when it is not kept)."""
    document = None
    if is_document_id(document_id.lower()):
        document = await run_in_threadpool(store.get, document_id.lower())
    if document is None:
        raise not_found(document_id)
    path = await run_in_threadpool(raw.find, document.id, document.filename)
    if path is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"the original file of document {document.id} is not kept "
            "(only files uploaded through POST /documents are)",
        )
    return FileResponse(
        path,
        media_type=document.mime_type or "application/octet-stream",
        filename=document.filename,
        content_disposition_type="inline",
        headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "private, max-age=300"},
    )
