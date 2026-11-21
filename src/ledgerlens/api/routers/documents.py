"""Document upload: store the file, record it ``queued`` and hand it to the ingestion worker.

* ``POST /documents`` (multipart, field ``file``) -- the loader hashes and sniffs the
  upload; the same bytes uploaded before (under any name) are a **duplicate**: 200 with the
  existing ``document_id`` and its status, nothing queued. A new file is claimed (its
  ``queued`` row is inserted), kept under ``loader.upload_dir`` and queued for the worker:
  202 with the ``document_id`` and the job id. A file whose earlier run never reached
  ``ready`` is queued again under its old id.

The raw file is kept at ``<upload_dir>/<document_id>/<original name>``: the worker ingests
it by path (so API and worker must share the folder) and the document keeps its original
file name. If the queue is down the document stays ``queued`` and the response is 503;
uploading the same file again queues it.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Awaitable, Callable
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile, status
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ledgerlens.core.config import get_config, repo_path
from ledgerlens.core.schemas import Document, DocumentStatus
from ledgerlens.ingestion.loaders import DedupeRegistry, FileLoader, LoaderError, LoadResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

_UNSAFE = re.compile(r"[^A-Za-z0-9._ -]+")
MAX_NAME_CHARS = 150


class UploadOut(BaseModel):
    document_id: str
    status: DocumentStatus
    duplicate: bool
    """The same bytes were uploaded before: ``document_id`` is that document's."""
    job_id: str | None = None
    """The ingestion job queued for this upload (``None`` for a duplicate)."""


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
    """Reads documents from Postgres."""

    def get(self, document_id: str) -> Document | None:
        from ledgerlens.storage.postgres.db import get_session_factory
        from ledgerlens.storage.postgres.repositories import DocumentRepository

        with get_session_factory()() as session:
            return DocumentRepository(session).get(document_id)


def get_document_store() -> DocumentStore:
    return DocumentStore()


Registry = Annotated[DedupeRegistry, Depends(get_registry)]
RawStore = Annotated[RawFileStore, Depends(get_raw_store)]
Store = Annotated[DocumentStore, Depends(get_document_store)]
Queue = Annotated[Enqueue, Depends(get_enqueue)]


# --------------------------------------------------------------------------- upload


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
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

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
