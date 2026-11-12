"""What every extractor (rules, LLM, fine-tuned) takes and returns.

An extractor reads a document's text and returns an :class:`ExtractionResult`: the typed
record (:data:`~ledgerlens.core.schemas.ExtractedDocument`) if every required field was
found, and in any case the fields it did find and the required ones it did not. Keeping
the partial fields lets evaluation score a near-miss and lets the pipeline show a
reviewer what is left to fill in.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ledgerlens.core.schemas import DocType, ExtractedRecord, extraction_schema, parse_extracted


class ExtractionError(RuntimeError):
    """The extractor could not run at all (as opposed to finding few fields)."""


class ExtractionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    extractor: str
    doc_type: DocType
    record: ExtractedRecord | None = None
    fields: dict[str, Any] = Field(default_factory=dict)  # everything found, JSON-ready
    missing: list[str] = Field(default_factory=list)  # required fields not found
    errors: list[str] = Field(default_factory=list)  # validation problems with found values

    @property
    def ok(self) -> bool:
        return self.record is not None

    @classmethod
    def from_fields(
        cls, extractor: str, doc_type: DocType | str, fields: dict[str, Any]
    ) -> ExtractionResult:
        """Validate ``fields`` into the record for ``doc_type``.

        ``None`` values are dropped first, so a field that was not found falls back to the
        schema default (or is reported in ``missing`` if it is required).
        """
        kind = DocType(doc_type)
        found = {k: v for k, v in fields.items() if v is not None and v != []}
        schema = extraction_schema(kind)
        required = [
            name
            for name, info in schema.model_fields.items()
            if info.is_required() and name != "doc_type"
        ]
        missing = [name for name in required if name not in found]
        if missing:
            return cls(extractor=extractor, doc_type=kind, fields=found, missing=missing)
        try:
            record = parse_extracted({**found, "doc_type": kind.value})
        except ValidationError as exc:
            errors = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
            return cls(extractor=extractor, doc_type=kind, fields=found, errors=errors)
        return cls(extractor=extractor, doc_type=kind, record=record, fields=found)


@runtime_checkable
class Extractor(Protocol):
    """Turns a document's text into a typed record."""

    name: str

    def extract(self, text: str, doc_type: DocType | str) -> ExtractionResult:
        """Extract the record for ``doc_type`` from ``text`` (the document's full text)."""
        ...
