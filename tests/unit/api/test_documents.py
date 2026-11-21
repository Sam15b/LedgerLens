"""POST /documents, GET /documents and GET /documents/{id} over stand-in stores: the
upload is stored and queued once, duplicates are not, limits answer 413/415/422, and the
reads pass their filters through."""

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ledgerlens.api.main import create_app
from ledgerlens.api.routers import documents
from ledgerlens.api.routers.documents import (
    MULTIPART_OVERHEAD_BYTES,
    DocumentDetail,
    RawFileStore,
    safe_filename,
)
from ledgerlens.core.config import get_config
from ledgerlens.core.schemas import DocType, Document, DocumentStatus
from ledgerlens.ingestion.loaders import InMemoryRegistry

FIXTURES = Path(__file__).parents[2] / "fixtures" / "documents"
PDF = (FIXTURES / "invoice_digital.pdf").read_bytes()
PNG_HEAD = b"\x89PNG\r\n\x1a\n"
DOC_ID = "6f1c2a8e-4b7d-4c1e-9a3f-0d2b5e7c9a10"


def document(**fields: Any) -> Document:
    values: dict[str, Any] = {
        "id": DOC_ID,
        "sha256": "a" * 64,
        "filename": "inv.pdf",
        "mime_type": "application/pdf",
        "doc_type": DocType.INVOICE,
        "status": DocumentStatus.READY,
        "pages": 1,
        "created_at": datetime(2026, 11, 21, tzinfo=UTC),
    }
    return Document.model_validate(values | fields)


class FakeStore:
    def __init__(self, *docs: Document) -> None:
        self.docs = {d.id: d for d in docs}
        self.listed: dict[str, Any] = {}

    def get(self, document_id: str) -> Document | None:
        return self.docs.get(document_id)

    def list(self, **filters: Any) -> list[Document]:
        self.listed = filters
        return list(self.docs.values())

    def detail(self, document_id: str) -> DocumentDetail | None:
        found = self.docs.get(document_id)
        if found is None:
            return None
        out = DocumentDetail.model_validate(found.model_dump() | {"chunks": 2})
        out.record_type, out.record_id = "invoice", "INV-1"
        out.record = {"invoice_number": "INV-1", "total": "120.00", "line_items": []}
        return out


class RetryRegistry(InMemoryRegistry):
    """Every file was claimed before by DOC_ID, whose run never finished."""

    def claim(self, document: Document) -> str | None:
        return DOC_ID

    def needs_retry(self, document_id: str) -> bool:
        return True


class Env:
    def __init__(self, tmp: Path) -> None:
        self.registry: InMemoryRegistry = InMemoryRegistry()
        self.store = FakeStore()
        self.raw = RawFileStore(tmp)
        self.queued: list[Path] = []
        self.queue_down = False
        app = create_app()
        app.dependency_overrides[documents.get_registry] = lambda: self.registry
        app.dependency_overrides[documents.get_document_store] = lambda: self.store
        app.dependency_overrides[documents.get_raw_store] = lambda: self.raw
        app.dependency_overrides[documents.get_enqueue] = lambda: self.enqueue
        self.client = TestClient(app)

    async def enqueue(self, path: Path) -> str | None:
        if self.queue_down:
            raise ConnectionError("redis is down")
        self.queued.append(path)
        return f"job-{len(self.queued)}"

    def upload(self, data: bytes, name: str = "inv.pdf", **kwargs: Any) -> Any:
        files = {"file": (name, data, "application/octet-stream")}
        return self.client.post("/documents", files=files, **kwargs)


@pytest.fixture
def env(tmp_path: Path) -> Iterator[Env]:
    yield Env(tmp_path)


@pytest.fixture
def small_limit(monkeypatch: pytest.MonkeyPatch) -> int:
    """Uploads limited to 50 KB; returns the limit in bytes."""
    config = get_config()
    loader = config.ingestion.loader.model_copy(update={"max_file_mb": 50 / 1024})
    small = config.model_copy(
        update={"ingestion": config.ingestion.model_copy(update={"loader": loader})}
    )
    monkeypatch.setattr(documents, "get_config", lambda: small)
    return 50 * 1024


# --------------------------------------------------------------------------- upload


def test_a_new_upload_is_stored_and_queued(env: Env) -> None:
    response = env.upload(PDF, "Acme invoice #7.pdf")
    assert response.status_code == 202
    body = response.json()
    assert body == {
        "document_id": body["document_id"],
        "status": "queued",
        "duplicate": False,
        "job_id": "job-1",
    }
    (path,) = env.queued
    assert path == env.raw.root.resolve() / body["document_id"] / "Acme invoice _7.pdf"
    assert path.read_bytes() == PDF
    assert not list(path.parent.glob(".*.part"))


def test_the_same_bytes_again_are_a_duplicate_and_not_queued(env: Env) -> None:
    first = env.upload(PDF).json()
    env.store.docs[first["document_id"]] = document(id=first["document_id"])
    again = env.upload(PDF, "renamed.pdf")
    assert again.status_code == 200
    assert again.json() == {
        "document_id": first["document_id"],
        "status": "ready",
        "duplicate": True,
        "job_id": None,
    }
    assert len(env.queued) == 1


def test_an_unfinished_document_is_queued_again_under_its_id(env: Env) -> None:
    env.registry = RetryRegistry()
    response = env.upload(PDF)
    assert response.status_code == 202
    assert response.json()["document_id"] == DOC_ID
    assert env.queued[0].parent.name == DOC_ID


def test_queue_down_is_503_and_the_file_is_kept(env: Env) -> None:
    env.queue_down = True
    response = env.upload(PDF)
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "unavailable" and "upload the file again" in error["message"]
    assert len(list(env.raw.root.glob("*/inv.pdf"))) == 1


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("invoice.pdf", "invoice.pdf"),
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\me\\scan 01.png", "scan 01.png"),
        ("<script>.pdf", "_script_.pdf"),
        ("...", "upload"),
        (None, "upload"),
        ("a" * 300 + ".pdf", "a" * 146 + ".pdf"),
    ],
)
def test_safe_filename(name: str | None, expected: str) -> None:
    assert safe_filename(name) == expected


# --------------------------------------------------------------------------- limits


def test_an_unsupported_type_is_415_whatever_its_name(env: Env) -> None:
    response = env.upload(b"just some text", "looks-like.pdf")
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"
    assert not env.queued


@pytest.mark.parametrize("data", [b"", PNG_HEAD + b"truncated"])
def test_an_empty_or_unreadable_file_is_422(env: Env, data: bytes) -> None:
    response = env.upload(data, "scan.png")
    assert response.status_code == 422
    assert not env.queued


def test_a_file_over_the_limit_is_413(env: Env, small_limit: int) -> None:
    response = env.upload(PNG_HEAD + b"0" * (small_limit + 1), "big.png")
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "too_large"
    assert "MB limit" in response.json()["error"]["message"]


def test_a_large_declared_body_is_refused_before_it_is_read(
    env: Env, small_limit: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    def never(*args: Any) -> None:
        raise AssertionError("the body was parsed")

    monkeypatch.setattr(documents, "load_upload", never)
    big = PNG_HEAD + b"0" * (small_limit + MULTIPART_OVERHEAD_BYTES)
    response = env.upload(big, "huge.png")
    assert response.status_code == 413
    assert "the upload is over" in response.json()["error"]["message"]


def test_a_streamed_body_is_cut_off_at_the_limit(env: Env, small_limit: int) -> None:
    boundary = "xyz"
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="s.png"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode()
    body = head + PNG_HEAD + b"0" * (small_limit + MULTIPART_OVERHEAD_BYTES)
    body += f"\r\n--{boundary}--\r\n".encode()

    def chunks() -> Iterator[bytes]:  # no Content-Length: sent chunked
        for start in range(0, len(body), 16 * 1024):
            yield body[start : start + 16 * 1024]

    response = env.client.post(
        "/documents",
        content=chunks(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    assert response.status_code == 413
    assert not env.queued


# --------------------------------------------------------------------------- reads


def test_list_passes_filters_and_returns_documents(env: Env) -> None:
    env.store = FakeStore(document(), document(id=DOC_ID.replace("6f", "7f", 1)))
    response = env.client.get(
        "/documents",
        params={"status": "ready", "doc_type": "invoice", "limit": 5, "offset": 10},
    )
    assert response.status_code == 200
    assert [d["filename"] for d in response.json()] == ["inv.pdf", "inv.pdf"]
    assert env.store.listed == {
        "status": DocumentStatus.READY,
        "doc_type": DocType.INVOICE,
        "limit": 5,
        "offset": 10,
    }


@pytest.mark.parametrize(
    "params", [{"status": "done"}, {"doc_type": "memo"}, {"limit": 0}, {"offset": -1}]
)
def test_list_rejects_bad_filters(env: Env, params: dict[str, Any]) -> None:
    assert env.client.get("/documents", params=params).status_code == 422


def test_get_returns_status_and_extracted_fields(env: Env) -> None:
    env.store = FakeStore(document(status=DocumentStatus.FAILED, error="ocr: no text"))
    body = env.client.get(f"/documents/{DOC_ID.upper()}").json()
    assert body["id"] == DOC_ID and body["status"] == "failed" and body["error"] == "ocr: no text"
    assert body["chunks"] == 2 and body["record_type"] == "invoice"
    assert body["record_id"] == "INV-1" and body["record"]["total"] == "120.00"


@pytest.mark.parametrize("document_id", ["not-a-uuid", "00000000-0000-4000-8000-000000000000"])
def test_get_unknown_is_404(env: Env, document_id: str) -> None:
    response = env.client.get(f"/documents/{document_id}")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
