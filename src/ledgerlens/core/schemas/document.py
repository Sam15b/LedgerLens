"""The ``Document`` record: one uploaded file and where it is in the ingestion pipeline.

Mirrors the ``documents`` table (docs/layers/02-storage.md). The extracted business data
(an invoice, PO, ...) lives in its own record that points back here by ``document_id``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BeforeValidator,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from ledgerlens.core.ids import is_document_id, new_document_id
from ledgerlens.core.schemas.common import OptionalText, SchemaModel, Text

Sha256 = Annotated[
    str,
    BeforeValidator(lambda v: v.strip().lower() if isinstance(v, str) else v),
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]
"""Hex SHA-256 digest of the file bytes (normalised to lower case)."""

OCREngineName = Literal["tesseract", "paddle"]


class DocType(StrEnum):
    """What a document is; decides its chunking strategy and extraction schema."""

    INVOICE = "invoice"
    RECEIPT = "receipt"
    PURCHASE_ORDER = "purchase_order"
    PAYMENT = "payment"
    CONTRACT = "contract"
    UNKNOWN = "unknown"
    """Not classified yet, or not a supported kind of document."""


class DocumentStatus(StrEnum):
    """Pipeline state. ``ready`` and ``failed`` are final until the document is re-run."""

    QUEUED = "queued"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"

    @property
    def is_final(self) -> bool:
        return self in (DocumentStatus.READY, DocumentStatus.FAILED)


class Document(SchemaModel):
    id: str = Field(default_factory=new_document_id)
    """``document_id`` (UUID4) shared by Postgres, Pinecone vector ids and Neo4j."""
    sha256: Sha256
    filename: Text
    mime_type: OptionalText = None
    doc_type: DocType = DocType.UNKNOWN
    status: DocumentStatus = DocumentStatus.QUEUED
    error: OptionalText = None
    """Why the last run failed; required when ``status`` is ``failed``."""
    pages: int | None = Field(default=None, ge=1)
    ocr_engine: OCREngineName | None = None
    """``None`` for digital PDFs read from their text layer."""
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("id")
    @classmethod
    def _canonical_id(cls, value: str) -> str:
        if not is_document_id(value):
            raise ValueError(f"not a canonical document id (lower-case UUID4): {value!r}")
        return value

    @model_validator(mode="after")
    def _failed_needs_error(self) -> Document:
        if self.status is DocumentStatus.FAILED and not self.error:
            raise ValueError("a failed document must say why (error)")
        return self

    def with_status(self, status: DocumentStatus, *, error: str | None = None) -> Document:
        """Return a copy moved to ``status``; ``error`` is cleared unless it failed."""
        return self.model_validate(
            self.model_dump() | {"status": status, "error": error if error else None}
        )
