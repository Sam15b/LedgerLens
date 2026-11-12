import hashlib
import io
import threading
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from ledgerlens.core.config import LoaderConfig
from ledgerlens.core.ids import is_document_id
from ledgerlens.core.schemas import (
    Contract,
    DocumentStatus,
    ExtractedModel,
    Invoice,
    Payment,
    PurchaseOrder,
    Receipt,
    parse_extracted,
)
from ledgerlens.ingestion.loaders import (
    FileLoader,
    FileTooLargeError,
    InMemoryRegistry,
    LoaderError,
    UnreadableFileError,
    UnsupportedFileError,
    sniff_mime_type,
)

FIXTURES = Path(__file__).parents[2] / "fixtures" / "documents"
CONFIG = LoaderConfig(
    max_file_mb=1, allowed_mime_types=["application/pdf", "image/png", "image/jpeg", "image/tiff"]
)

# fixture -> (MIME type, pages, record type of the expected JSON)
SAMPLES: dict[str, tuple[str, int, type[ExtractedModel]]] = {
    "invoice_digital.pdf": ("application/pdf", 1, Invoice),
    "purchase_order.pdf": ("application/pdf", 1, PurchaseOrder),
    "contract.pdf": ("application/pdf", 2, Contract),
    "payment_advice.pdf": ("application/pdf", 1, Payment),
    "receipt_scanned.png": ("image/png", 1, Receipt),
}


def loader(**config: object) -> FileLoader:
    return FileLoader(CONFIG.model_copy(update=config))


def write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


# --------------------------------------------------------------------------- fixtures


@pytest.mark.parametrize("name", list(SAMPLES))
def test_sample_documents_load_with_hash_type_and_pages(name: str) -> None:
    path = FIXTURES / name
    mime_type, pages, _ = SAMPLES[name]

    file = loader().inspect(path)

    assert file.filename == name
    assert file.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert file.size_bytes == path.stat().st_size
    assert (file.mime_type, file.pages) == (mime_type, pages)


@pytest.mark.parametrize("name", list(SAMPLES))
def test_each_sample_has_a_valid_expected_record(name: str) -> None:
    expected = (FIXTURES / name).with_suffix(".json").read_text(encoding="utf-8")

    record = parse_extracted(expected)

    assert isinstance(record, SAMPLES[name][2])
    assert record.consistency_issues() == []


def test_sample_records_are_linked_for_reconciliation() -> None:
    def load(name: str) -> object:
        return parse_extracted((FIXTURES / name).read_text(encoding="utf-8"))

    contract, po, invoice, payment = (
        load(f"{n}.json")
        for n in ("contract", "purchase_order", "invoice_digital", "payment_advice")
    )
    assert isinstance(contract, Contract) and isinstance(po, PurchaseOrder)
    assert isinstance(invoice, Invoice) and isinstance(payment, Payment)
    assert po.contract_number == contract.contract_number
    assert invoice.po_number == po.po_number and invoice.total == po.total
    assert payment.invoice_numbers == [invoice.invoice_number]


def test_inspect_bytes_matches_inspect_path() -> None:
    path = FIXTURES / "contract.pdf"

    assert loader().inspect_bytes(path.read_bytes(), "contract.pdf") == loader().inspect(path)


def test_type_comes_from_content_not_file_name(tmp_path: Path) -> None:
    disguised = write(tmp_path, "scan.png", (FIXTURES / "invoice_digital.pdf").read_bytes())

    assert loader().inspect(disguised).mime_type == "application/pdf"


def test_jpeg_and_multi_page_tiff(tmp_path: Path) -> None:
    jpeg, tiff = io.BytesIO(), io.BytesIO()
    Image.new("L", (40, 30), 255).save(jpeg, format="JPEG")
    frames = [Image.new("L", (40, 30), shade) for shade in (255, 200, 150)]
    frames[0].save(tiff, format="TIFF", save_all=True, append_images=frames[1:])

    assert loader().inspect_bytes(jpeg.getvalue(), "a.jpg").pages == 1
    tiff_file = loader().inspect(write(tmp_path, "fax.tiff", tiff.getvalue()))
    assert (tiff_file.mime_type, tiff_file.pages) == ("image/tiff", 3)


# --------------------------------------------------------------------------- dedupe


def test_loading_the_same_file_twice_skips_the_second_load(tmp_path: Path) -> None:
    files = loader()
    first = files.load(FIXTURES / "invoice_digital.pdf")
    renamed = write(
        tmp_path, "copy of invoice.pdf", (FIXTURES / "invoice_digital.pdf").read_bytes()
    )

    second = files.load(FIXTURES / "invoice_digital.pdf")
    third = files.load(renamed)
    upload = files.load_bytes(renamed.read_bytes(), "upload.pdf")

    assert not first.duplicate and is_document_id(first.document_id)
    assert all(r.duplicate for r in (second, third, upload))
    assert {r.document_id for r in (second, third, upload)} == {first.document_id}
    assert third.file.filename == "copy of invoice.pdf"


def test_different_files_get_different_documents() -> None:
    registry = InMemoryRegistry()
    files = FileLoader(CONFIG, registry=registry)

    ids = {files.load(FIXTURES / name).document_id for name in SAMPLES}

    assert len(ids) == len(registry) == len(SAMPLES)


def test_registry_can_be_shared_between_loaders() -> None:
    registry = InMemoryRegistry()
    first = FileLoader(CONFIG, registry=registry).load(FIXTURES / "contract.pdf")

    again = FileLoader(CONFIG, registry=registry).load(FIXTURES / "contract.pdf")

    assert again.duplicate and again.document_id == first.document_id


def test_to_document_starts_a_queued_document() -> None:
    result = loader().load(FIXTURES / "contract.pdf")

    doc = result.to_document()

    assert doc.id == result.document_id
    assert (doc.sha256, doc.filename) == (result.file.sha256, "contract.pdf")
    assert (doc.mime_type, doc.pages, doc.status) == ("application/pdf", 2, DocumentStatus.QUEUED)


def test_concurrent_claims_of_one_hash_have_a_single_winner() -> None:
    registry = InMemoryRegistry()
    results: list[str | None] = []
    start = threading.Barrier(8)

    def claim(n: int) -> None:
        start.wait()
        results.append(registry.claim("a" * 64, f"doc-{n}"))

    threads = [threading.Thread(target=claim, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results.count(None) == 1
    winner = registry.get("a" * 64)
    assert {r for r in results if r is not None} == {winner}


# --------------------------------------------------------------------------- rejections


@pytest.mark.parametrize(
    "name,data",
    [
        ("notes.txt", b"Invoice INV-1 total 12.00\n"),
        ("invoice.docx", b"PK\x03\x04\x14\x00\x06\x00" + b"\x00" * 30),
        ("invoice.pdf", b"<html><body>not a pdf</body></html>"),
    ],
)
def test_unsupported_types_are_rejected(tmp_path: Path, name: str, data: bytes) -> None:
    registry = InMemoryRegistry()

    with pytest.raises(UnsupportedFileError, match="an unrecognised format is not supported"):
        FileLoader(CONFIG, registry=registry).load(write(tmp_path, name, data))
    assert len(registry) == 0  # rejected files are never registered


def test_types_outside_the_configured_list_are_rejected() -> None:
    pdf_only = loader(allowed_mime_types=["application/pdf"])

    with pytest.raises(
        UnsupportedFileError, match=r"image/png is not supported \(allowed: application/pdf\)"
    ):
        pdf_only.inspect(FIXTURES / "receipt_scanned.png")


def test_oversized_files_are_rejected_before_reading() -> None:
    tiny = loader(max_file_mb=0.001)  # ~1 KB

    with pytest.raises(FileTooLargeError, match=r"contract.pdf: 0.0 MB is over the 0.001 MB limit"):
        tiny.inspect(FIXTURES / "contract.pdf")
    with pytest.raises(FileTooLargeError):
        tiny.inspect_bytes((FIXTURES / "contract.pdf").read_bytes(), "contract.pdf")


def test_default_limits_come_from_ingestion_yaml() -> None:
    files = FileLoader()

    assert files.max_bytes == 25 * 1024 * 1024
    assert "application/pdf" in files.config.allowed_mime_types


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(UnreadableFileError, match="empty"):
        loader().inspect(write(tmp_path, "empty.pdf", b""))


@pytest.mark.parametrize(
    "name,data,message",
    [
        ("broken.pdf", b"%PDF-1.7\n this is not really a pdf", "not a readable PDF"),
        ("cut.png", (FIXTURES / "receipt_scanned.png").read_bytes()[:200], "not a readable image"),
    ],
)
def test_corrupt_files_are_rejected(tmp_path: Path, name: str, data: bytes, message: str) -> None:
    with pytest.raises(UnreadableFileError, match=f"{name}: {message}"):
        loader().inspect(write(tmp_path, name, data))


def test_password_protected_pdf_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "locked.pdf"
    with pymupdf.open() as doc:
        doc.new_page()
        doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="secret")

    with pytest.raises(UnreadableFileError, match="password-protected"):
        loader().inspect(path)


def test_missing_file_is_a_loader_error(tmp_path: Path) -> None:
    with pytest.raises(LoaderError, match="file not found"):
        loader().inspect(tmp_path / "nope.pdf")


def test_sniff_mime_type() -> None:
    assert sniff_mime_type(b"%PDF-1.4\n") == "application/pdf"
    assert sniff_mime_type(b"\xff\xd8\xff\xe0") == "image/jpeg"
    assert sniff_mime_type(b"MM\x00*") == sniff_mime_type(b"II*\x00") == "image/tiff"
    assert sniff_mime_type(b"GIF89a") is None
