"""Load an uploaded file safely: identify it, count its pages, and never process it twice.

:class:`FileLoader` is the first step of ingestion. For a path (or the bytes of an upload)
it

1. hashes the content with SHA-256 in fixed-size blocks (large files are never read into
   memory just to be hashed);
2. sniffs the real type from the leading bytes, not the file name;
3. counts pages (PyMuPDF for PDFs, Pillow for images; multi-page TIFFs count frames);

and returns a :class:`LoadedFile`. :meth:`FileLoader.load` also checks the hash against a
:class:`DedupeRegistry`, so the same bytes uploaded twice -- under any name -- map to one
``document_id`` and the second upload is skipped.
"""

from __future__ import annotations

import hashlib
import io
import logging
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import BinaryIO, Protocol

import pymupdf
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from ledgerlens.core.config import LoaderConfig, get_config
from ledgerlens.core.ids import new_document_id
from ledgerlens.core.schemas import Document
from ledgerlens.core.schemas.document import Sha256

logger = logging.getLogger(__name__)

HASH_BLOCK_BYTES = 1024 * 1024
_SNIFF_BYTES = 16

# Leading bytes ("magic numbers") of the formats ingestion understands.
_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", "application/pdf"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
)


class LoadedFile(BaseModel):
    """What ingestion knows about a file before reading its contents."""

    model_config = ConfigDict(frozen=True)

    filename: str
    sha256: Sha256
    size_bytes: int = Field(ge=1)
    mime_type: str
    pages: int = Field(ge=1)


class LoadResult(BaseModel):
    """A loaded file and the ``document_id`` it belongs to.

    ``duplicate`` is true when these bytes were registered before: ``document_id`` is then
    the earlier document's, and the file should not be processed again.
    """

    model_config = ConfigDict(frozen=True)

    file: LoadedFile
    document_id: str
    duplicate: bool

    def to_document(self) -> Document:
        """A new ``queued`` :class:`Document` for this file."""
        return Document(
            id=self.document_id,
            sha256=self.file.sha256,
            filename=self.file.filename,
            mime_type=self.file.mime_type,
            pages=self.file.pages,
        )


# --------------------------------------------------------------------------- errors


class LoaderError(ValueError):
    """The file cannot be ingested; the message says why and is safe to show the user."""


class UnreadableFileError(LoaderError):
    """Truncated, corrupt or password-protected."""


# --------------------------------------------------------------------------- helpers


def sniff_mime_type(head: bytes) -> str | None:
    """MIME type from a file's first bytes, or ``None`` if it is not a known format."""
    for signature, mime_type in _SIGNATURES:
        if head.startswith(signature):
            return mime_type
    return None


def sha256_of(blocks: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for block in blocks:
        digest.update(block)
    return digest.hexdigest()


def _read_blocks(stream: BinaryIO) -> Iterable[bytes]:
    while block := stream.read(HASH_BLOCK_BYTES):
        yield block


def count_pages(source: Path | bytes, mime_type: str) -> int:
    """Pages in a PDF, or frames in an image (1 for PNG/JPEG)."""
    if mime_type == "application/pdf":
        return _pdf_pages(source)
    return _image_frames(source)


def _pdf_pages(source: Path | bytes) -> int:
    try:
        doc = (
            pymupdf.open(source)
            if isinstance(source, Path)
            else pymupdf.open(stream=source, filetype="pdf")
        )
    except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
        raise UnreadableFileError(f"not a readable PDF: {exc}") from exc
    with doc:
        if doc.needs_pass:
            raise UnreadableFileError("the PDF is password-protected")
        if doc.page_count < 1:
            raise UnreadableFileError("the PDF has no pages")
        return int(doc.page_count)


def _image_frames(source: Path | bytes) -> int:
    try:
        with Image.open(source if isinstance(source, Path) else io.BytesIO(source)) as image:
            frames = int(getattr(image, "n_frames", 1))
            image.verify()  # catches truncated / corrupt image data
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise UnreadableFileError(f"not a readable image: {exc}") from exc
    return frames


# --------------------------------------------------------------------------- dedupe


class DedupeRegistry(Protocol):
    """Remembers which ``document_id`` each content hash belongs to.

    In memory for now; the Postgres version (a unique index on ``documents.sha256``)
    replaces it in Phase 3 so duplicates are caught across processes and restarts.
    """

    def claim(self, sha256: str, document_id: str) -> str | None:
        """Register ``document_id`` for ``sha256`` unless the hash is already known.

        Returns ``None`` if this call registered it, else the existing ``document_id``.
        Check-and-register is atomic, so two concurrent uploads of the same bytes cannot
        both win.
        """
        ...

    def get(self, sha256: str) -> str | None: ...


class InMemoryRegistry:
    def __init__(self) -> None:
        self._ids: dict[str, str] = {}
        self._lock = threading.Lock()

    def claim(self, sha256: str, document_id: str) -> str | None:
        with self._lock:
            existing = self._ids.get(sha256)
            if existing is None:
                self._ids[sha256] = document_id
            return existing

    def get(self, sha256: str) -> str | None:
        with self._lock:
            return self._ids.get(sha256)

    def __len__(self) -> int:
        return len(self._ids)


# --------------------------------------------------------------------------- loader


class FileLoader:
    def __init__(
        self,
        config: LoaderConfig | None = None,
        *,
        registry: DedupeRegistry | None = None,
    ) -> None:
        self.config = config or get_config().ingestion.loader
        self.registry: DedupeRegistry = registry if registry is not None else InMemoryRegistry()

    def inspect(self, path: Path | str) -> LoadedFile:
        """Identify the file at ``path`` (no dedupe check). Raises :class:`LoaderError`."""
        path = Path(path)
        try:
            size = path.stat().st_size
            with path.open("rb") as f:
                mime_type = self._check_type(path.name, f.read(_SNIFF_BYTES))
                f.seek(0)
                digest = sha256_of(_read_blocks(f))
        except FileNotFoundError as exc:
            raise LoaderError(f"{path.name}: file not found") from exc
        return LoadedFile(
            filename=path.name,
            sha256=digest,
            size_bytes=size,
            mime_type=mime_type,
            pages=self._count_pages(path.name, path, mime_type),
        )

    def inspect_bytes(self, data: bytes, filename: str) -> LoadedFile:
        """Identify an upload held in memory (no dedupe check). Raises :class:`LoaderError`."""
        mime_type = self._check_type(filename, data[:_SNIFF_BYTES])
        return LoadedFile(
            filename=filename,
            sha256=sha256_of([data]),
            size_bytes=len(data),
            mime_type=mime_type,
            pages=self._count_pages(filename, data, mime_type),
        )

    def load(self, path: Path | str) -> LoadResult:
        """Inspect ``path`` and claim its hash; a file seen before comes back ``duplicate``."""
        return self._claim(self.inspect(path))

    def load_bytes(self, data: bytes, filename: str) -> LoadResult:
        return self._claim(self.inspect_bytes(data, filename))

    # ------------------------------------------------------------------ internals

    def _claim(self, file: LoadedFile) -> LoadResult:
        new_id = new_document_id()
        existing = self.registry.claim(file.sha256, new_id)
        if existing is None:
            return LoadResult(file=file, document_id=new_id, duplicate=False)
        logger.info(
            "skipping %s: same content as document %s",
            file.filename,
            existing,
            extra={"sha256": file.sha256, "document_id": existing},
        )
        return LoadResult(file=file, document_id=existing, duplicate=True)

    def _check_type(self, filename: str, head: bytes) -> str:
        mime_type = sniff_mime_type(head)
        if mime_type is None:
            raise LoaderError(f"{filename}: unrecognised file type")
        return mime_type

    @staticmethod
    def _count_pages(filename: str, source: Path | bytes, mime_type: str) -> int:
        try:
            return count_pages(source, mime_type)
        except UnreadableFileError as exc:
            raise UnreadableFileError(f"{filename}: {exc}") from exc
