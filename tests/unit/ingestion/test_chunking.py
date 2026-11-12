from pathlib import Path

import pytest

from ledgerlens.core.config import ChunkingConfig, ChunkSpec
from ledgerlens.core.ids import new_document_id
from ledgerlens.core.schemas import DocType, ExtractedRecord, parse_extracted
from ledgerlens.ingestion.chunking import (
    Chunk,
    approx_token_count,
    chunk_document,
    context_header,
    is_clause_heading,
    split_text,
)
from ledgerlens.ingestion.layout import BBox, PageContent, Word, layout_pages

FIXTURES = Path(__file__).parents[2] / "fixtures" / "documents"
NO_HEADER = ChunkingConfig(context_header=False)


def pdf_pages(name: str) -> list[PageContent]:
    pages = layout_pages(FIXTURES / f"{name}.pdf", "application/pdf")
    assert all(isinstance(p, PageContent) for p in pages)
    return [p for p in pages if isinstance(p, PageContent)]


def record(name: str) -> ExtractedRecord:
    return parse_extracted((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def chunks_of(name: str, config: ChunkingConfig | None = None) -> list[Chunk]:
    rec = record(name)
    return chunk_document(
        pdf_pages(name),
        rec.doc_type,
        record=rec,
        config=config or ChunkingConfig(),
        count_tokens=approx_token_count,
    )


def receipt_page() -> PageContent:
    """The scanned receipt as OCR words (pixel boxes), the way Tesseract returns them."""
    rows = [
        ["KEDAI", "KOPI", "SENTOSA"],
        ["12", "Jalan", "Bunga,", "50450", "Kuala", "Lumpur"],
        ["Receipt:", "R-000732", "Date:", "14/03/2026"],
        ["2", "x", "Kopi", "O", "|4.40"],
        ["1", "x", "Roti", "bakar", "|5.10"],
        ["TOTAL", "MYR", "|9.50"],
        ["incl.", "SST", "6%", "|0.54"],
        ["Terima", "kasih!"],
    ]
    words = []
    for r, row in enumerate(rows):
        x, y = 40.0, 40.0 + r * 22  # line pitch of a narrow receipt
        for text in row:
            if text.startswith("|"):  # right-hand amount column
                text, x = text[1:], 340.0
            width = 11.0 * len(text)
            words.append(Word(text=text, bbox=BBox(x, y, x + width, y + 18), conf=90))
            x += width + 6
    return PageContent.from_words(words, page_number=1, width=460, height=700, source="ocr")


# --------------------------------------------------------------------------- golden boundaries


@pytest.mark.parametrize("name", ["invoice_digital", "purchase_order"])
def test_invoice_and_po_are_one_header_and_one_line_item_chunk(name: str) -> None:
    header, items = chunks_of(name, NO_HEADER)

    assert (header.kind, items.kind) == ("header", "line_items")
    assert header.text.splitlines()[0] in {"INVOICE", "PURCHASE ORDER"}
    assert "Total (GBP)  498.00" in header.text  # totals stay with the header
    assert "Pallet wrap" not in header.text
    assert items.text == (
        "| Description | Qty | Unit price | Amount |\n"
        "| --- | --- | --- | --- |\n"
        "| Pallet wrap 500m | 2 | 45.00 | 90.00 |\n"
        "| Shipping labels (box of 500) | 10 | 32.50 | 325.00 |"
    )
    assert [c.index for c in (header, items)] == [0, 1]


def test_contract_is_one_chunk_per_clause_plus_preamble() -> None:
    chunks = chunks_of("contract", NO_HEADER)

    assert [c.title for c in chunks] == [
        None,
        "1. Term",
        "2. Supply",
        "3. Prices",
        "4. Payment",
        "5. Invoices",
        "Schedule 1 - Rate card",
    ]
    assert {c.kind for c in chunks} == {"clause"}
    assert chunks[0].text.startswith("MASTER SUPPLY AGREEMENT\nContract number: MSA-2026-07")
    assert chunks[4].text == (
        "4. Payment\nThe Customer will pay each valid invoice within 30 days of the invoice date."
    )
    assert [c.pages for c in chunks] == [[1]] * 6 + [[2]]


def test_payment_advice_is_a_single_chunk() -> None:
    (chunk,) = chunks_of("payment_advice", NO_HEADER)

    assert chunk.kind == "body"
    assert chunk.text.startswith("REMITTANCE ADVICE")
    assert chunk.text.endswith("Total paid  498.00")


def test_ocr_receipt_splits_into_header_and_items() -> None:
    header, items = chunk_document([receipt_page()], DocType.RECEIPT, config=NO_HEADER)

    assert header.text.splitlines() == [
        "KEDAI KOPI SENTOSA",
        "12 Jalan Bunga, 50450 Kuala Lumpur",
        "Receipt: R-000732 Date: 14/03/2026",
        "",  # the item rows were here
        "TOTAL MYR  9.50",
        "incl. SST 6%  0.54",
        "Terima kasih!",
    ]
    assert items.text.splitlines()[2:] == ["| 1 x Roti bakar | 5.10 |"]
    assert items.text.splitlines()[0] == "| 2 x Kopi O | 4.40 |"


def test_unknown_documents_are_one_body_chunk() -> None:
    (chunk,) = chunk_document(pdf_pages("invoice_digital"), "unknown", config=NO_HEADER)

    assert chunk.kind == "body" and chunk.text.startswith("INVOICE")


def test_chunk_vector_ids() -> None:
    document_id = new_document_id()
    chunks = chunks_of("invoice_digital")

    assert [c.vector_id(document_id) for c in chunks] == [
        f"{document_id}#c0",
        f"{document_id}#c1",
    ]


# --------------------------------------------------------------------------- context header


@pytest.mark.parametrize(
    "name,expected",
    [
        (
            "invoice_digital",
            "[INVOICE INV-2026-0417 | Vendor: Acme Supplies Ltd | Date: 2026-10-02]",
        ),
        (
            "purchase_order",
            "[PURCHASE ORDER PO-88231 | Vendor: Acme Supplies Ltd | Date: 2026-09-20]",
        ),
        ("payment_advice", "[PAYMENT RA-5531 | Vendor: Acme Supplies Ltd | Date: 2026-10-30]"),
    ],
)
def test_every_chunk_starts_with_the_context_header(name: str, expected: str) -> None:
    chunks = chunks_of(name)

    for chunk in chunks:
        assert chunk.header == expected
        assert chunk.text == f"{expected}\n{chunk.body}"


def test_contract_headers_name_the_clause() -> None:
    chunks = chunks_of("contract")

    assert chunks[4].header == (
        "[CONTRACT MSA-2026-07 | Vendor: Acme Supplies Ltd | Date: 2026-01-01 | 4. Payment]"
    )
    assert (
        chunks[0].header == "[CONTRACT MSA-2026-07 | Vendor: Acme Supplies Ltd | Date: 2026-01-01]"
    )


def test_header_without_a_record_names_only_the_type() -> None:
    assert context_header(DocType.RECEIPT) == "[RECEIPT]"
    assert context_header("unknown") == "[DOCUMENT]"


def test_header_can_be_switched_off() -> None:
    chunk = chunks_of("payment_advice", NO_HEADER)[0]

    assert chunk.header is None and chunk.text == chunk.body


# --------------------------------------------------------------------------- token splitting


def text_page(lines: list[str]) -> PageContent:
    """A text-layer page with one line of words per entry, single-spaced."""
    words = []
    for row, line in enumerate(lines):
        x, y = 10.0, 10.0 + 16 * row
        for text in line.split():
            width = 6.0 * len(text)
            words.append(Word(text=text, bbox=BBox(x, y, x + width, y + 12)))
            x += width + 3
    return PageContent.from_words(words, page_number=1, width=600, height=900, source="text_layer")


def test_long_clauses_are_split_within_the_limit_with_overlap() -> None:
    sentences = [f"The supplier shall deliver batch {i} on time." for i in range(40)]  # 9 tokens
    page = text_page(["7. Delivery", *sentences])
    config = ChunkingConfig(contract=ChunkSpec(max_tokens=60, overlap=15))

    chunks = chunk_document([page], "contract", config=config, count_tokens=approx_token_count)

    assert len(chunks) > 5
    assert all(c.title == "7. Delivery" for c in chunks)
    assert all(approx_token_count(c.text) <= 60 for c in chunks)  # header included
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.body.splitlines()[0] == prev.body.splitlines()[-1]  # one sentence repeated
    assert "batch 39" in chunks[-1].body


def test_split_text_prefers_paragraphs_then_sentences() -> None:
    paragraph = " ".join(["word"] * 8) + "."  # 9 tokens
    text = "\n\n".join([paragraph] * 4)

    assert split_text(text, 20, 0, approx_token_count) == [f"{paragraph}\n\n{paragraph}"] * 2
    assert split_text("One two. Three four.", 3, 0, approx_token_count) == [
        "One two.",
        "Three four.",
    ]
    assert split_text("short", 20, 5, approx_token_count) == ["short"]
    assert split_text("", 20, 5, approx_token_count) == []


def test_an_overlong_word_is_cut_by_characters() -> None:
    parts = split_text("x" * 50, 4, 0, lambda text: len(text) // 5)

    assert parts == ["x" * 20, "x" * 20, "x" * 10]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1. Term", True),
        ("4.2 Fees and expenses", True),
        ("Section 3", True),
        ("ARTICLE IV - Liability", True),
        ("Schedule 1 - Rate card", True),
        ("2 x Kopi O", False),
        ("4.40", False),
        ("The section 3 above applies", False),
    ],
)
def test_clause_headings(text: str, expected: bool) -> None:
    assert is_clause_heading(text) is expected
