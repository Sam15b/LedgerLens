import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ledgerlens.core.ids import is_document_id
from ledgerlens.core.schemas import DocType, Document, DocumentStatus

SHA = "9F86D081884C7D659A2FEAA0C55AD015A3BF4F1B2B0B822CD15D6C15B0F00A08"


def document(**overrides: object) -> Document:
    return Document.model_validate({"sha256": SHA, "filename": "inv-001.pdf"} | overrides)


def test_defaults_for_a_new_upload() -> None:
    doc = document()

    assert is_document_id(doc.id)
    assert doc.sha256 == SHA.lower()
    assert (doc.doc_type, doc.status) == (DocType.UNKNOWN, DocumentStatus.QUEUED)
    assert doc.error is None and doc.pages is None and doc.ocr_engine is None
    assert doc.created_at.tzinfo is not None


def test_round_trips_through_json() -> None:
    doc = document(
        doc_type="invoice", status="ready", pages=2, ocr_engine="tesseract", mime_type="image/png"
    )

    data = json.loads(doc.model_dump_json())

    assert data["doc_type"] == "invoice" and data["status"] == "ready"
    assert Document.model_validate_json(doc.model_dump_json()) == doc


@pytest.mark.parametrize("sha", ["abc", "g" * 64, SHA + "0"])
def test_rejects_a_malformed_sha256(sha: str) -> None:
    with pytest.raises(ValidationError, match="sha256"):
        document(sha256=sha)


@pytest.mark.parametrize("doc_id", ["not-a-uuid", "9F86D081-884C-4D65-9A2F-EAA0C55AD015"])
def test_rejects_a_non_canonical_id(doc_id: str) -> None:
    with pytest.raises(ValidationError, match="canonical document id"):
        document(id=doc_id)


@pytest.mark.parametrize(
    "field,value",
    [
        ("pages", 0),
        ("status", "done"),
        ("doc_type", "memo"),
        ("ocr_engine", "easyocr"),
        ("filename", "   "),
        ("created_at", "2026-11-11T09:00:00"),  # naive timestamps are ambiguous
        ("colour", "blue"),  # unknown fields are rejected
    ],
)
def test_rejects_invalid_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        document(**{field: value})


def test_failed_document_must_say_why() -> None:
    with pytest.raises(ValidationError, match="must say why"):
        document(status="failed")
    with pytest.raises(ValidationError, match="must say why"):
        document(status="failed", error="  ")

    assert document(status="failed", error="OCR timed out").error == "OCR timed out"


def test_with_status_moves_through_the_pipeline() -> None:
    doc = document()

    failed = doc.with_status(DocumentStatus.FAILED, error="OCR timed out")
    retried = failed.with_status(DocumentStatus.PROCESSING)

    assert failed.status is DocumentStatus.FAILED and failed.error == "OCR timed out"
    assert retried.status is DocumentStatus.PROCESSING and retried.error is None
    assert retried.id == doc.id and doc.status is DocumentStatus.QUEUED  # copies, not edits


def test_documents_are_immutable() -> None:
    with pytest.raises(ValidationError):
        document().status = DocumentStatus.READY


def test_final_statuses() -> None:
    assert [s for s in DocumentStatus if s.is_final] == [
        DocumentStatus.READY,
        DocumentStatus.FAILED,
    ]


def test_created_at_keeps_the_given_time() -> None:
    when = datetime(2026, 11, 11, 9, 0, tzinfo=UTC)

    assert document(created_at=when).created_at == when
