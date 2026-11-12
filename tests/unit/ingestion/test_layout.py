import io
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from ledgerlens.ingestion.layout import (
    BBox,
    PageContent,
    ScannedPage,
    Table,
    Word,
    extract_words,
    find_tables,
    group_blocks,
    group_lines,
    has_text_layer,
    layout_pages,
    pages_needing_ocr,
    rasterize,
    read_text_page,
    text_layer_chars,
)

FIXTURES = Path(__file__).parents[2] / "fixtures" / "documents"
RECEIPT = FIXTURES / "receipt_scanned.png"


def open_pdf(name: str) -> pymupdf.Document:
    return pymupdf.open(FIXTURES / name)


def scanned_pdf() -> bytes:
    """The receipt fixture saved as a one-page, image-only PDF (what a scanner produces)."""
    doc = pymupdf.open()
    with Image.open(RECEIPT) as image:
        page = doc.new_page(width=image.width * 72 / 150, height=image.height * 72 / 150)
    page.insert_image(page.rect, filename=str(RECEIPT))
    data: bytes = doc.tobytes()
    return data


def word(text: str, x0: float, y0: float, x1: float, y1: float) -> Word:
    return Word(text=text, bbox=BBox(x0, y0, x1, y1))


# --------------------------------------------------------------------------- text layer


@pytest.mark.parametrize(
    "name", ["invoice_digital.pdf", "purchase_order.pdf", "contract.pdf", "payment_advice.pdf"]
)
def test_digital_fixtures_have_a_text_layer_on_every_page(name: str) -> None:
    with open_pdf(name) as doc:
        assert all(has_text_layer(doc[i]) for i in range(doc.page_count))
        assert pages_needing_ocr(doc) == []


def test_scanned_pdf_has_no_text_layer() -> None:
    with pymupdf.open(stream=scanned_pdf(), filetype="pdf") as doc:
        assert text_layer_chars(doc[0]) == 0
        assert not has_text_layer(doc[0])
        assert pages_needing_ocr(doc) == [1]


def test_threshold_counts_non_whitespace_characters() -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Signed:   J. Smith")  # 14 visible characters

    assert text_layer_chars(page) == 14
    assert has_text_layer(page, min_chars=14)
    assert not has_text_layer(page, min_chars=15)
    assert not has_text_layer(page)  # default threshold from ingestion.yaml is 50


# --------------------------------------------------------------------------- words


def test_words_keep_reading_order_and_boxes() -> None:
    with open_pdf("invoice_digital.pdf") as doc:
        page = doc[0]
        words = extract_words(page)

        assert [w.text for w in words[:4]] == ["INVOICE", "Acme", "Supplies", "Ltd,"]
        assert all(w.conf is None for w in words)
        for w in words:
            assert 0 <= w.bbox.x0 < w.bbox.x1 <= page.rect.width
            assert 0 <= w.bbox.y0 < w.bbox.y1 <= page.rect.height


def test_invoice_page_lines_blocks_and_tables() -> None:
    with open_pdf("invoice_digital.pdf") as doc:
        content = read_text_page(doc[0])

    assert content.source == "text_layer" and content.page_number == 1
    assert [line.text for line in content.lines[3:5]] == [
        "Invoice number: INV-2026-0417",
        "Invoice date: 2 October 2026",
    ]
    assert content.text.split("\n\n")[0] == (
        "INVOICE\nAcme Supplies Ltd, 4 Canal Wharf, Leeds LS11 5PS\nVAT reg. no. GB123456789"
    )
    assert len(content.blocks) == 5
    items, totals = content.tables
    assert items.to_markdown() == (
        "| Description | Qty | Unit price | Amount |\n"
        "| --- | --- | --- | --- |\n"
        "| Pallet wrap 500m | 2 | 45.00 | 90.00 |\n"
        "| Shipping labels (box of 500) | 10 | 32.50 | 325.00 |"
    )
    assert totals.rows == [["Subtotal", "415.00"], ["VAT 20%", "83.00"], ["Total (GBP)", "498.00"]]


def test_contract_rate_card_is_a_three_column_table_on_page_two() -> None:
    with open_pdf("contract.pdf") as doc:
        first, second = (read_text_page(doc[i]) for i in range(2))

    assert first.tables == []  # clause text is never mistaken for a table
    (rate_card,) = second.tables
    assert rate_card.n_columns == 3
    assert rate_card.rows[0] == ["Item", "Unit", "Price"]
    assert rate_card.rows[-1] == ["Courier", "hour", "38.00"]


def test_grouping_uses_geometry_not_input_order() -> None:
    # OCR returns words in any order, with slightly uneven baselines.
    words = [
        word("12.00", 300, 33, 340, 43),
        word("Total", 10, 32, 45, 42),
        word("Coffee", 10, 20, 50, 30),
        word("due", 48, 32, 70, 42),
        word("4.00", 300, 21, 330, 31),
    ]

    lines = group_lines(words)

    assert [line.text for line in lines] == ["Coffee  4.00", "Total due  12.00"]
    assert [[c.text for c in line.cells] for line in lines][1] == ["Total due", "12.00"]
    assert len(group_blocks(lines)) == 1
    assert find_tables(lines)[0].rows == [["Coffee", "4.00"], ["Total due", "12.00"]]


def test_right_aligned_amounts_form_one_column() -> None:
    lines = group_lines(
        [
            word("Item", 10, 0, 40, 10),
            word("Amount", 200, 0, 250, 10),
            word("Tea", 10, 12, 30, 22),
            word("4.00", 226, 12, 250, 22),
            word("Cake", 10, 24, 40, 34),
            word("120.00", 214, 24, 250, 34),
        ]
    )

    (table,) = find_tables(lines)

    assert table.rows == [["Item", "Amount"], ["Tea", "4.00"], ["Cake", "120.00"]]


def test_a_single_multi_cell_line_is_not_a_table() -> None:
    lines = group_lines([word("Total", 10, 0, 40, 10), word("9.50", 200, 0, 230, 10)])

    assert find_tables(lines) == []


def test_markdown_escapes_pipes() -> None:
    table = Table(rows=[["a|b", "c"], ["1", "2"]], bbox=BBox(0, 0, 1, 1))

    assert table.to_markdown().splitlines()[0] == r"| a\|b | c |"


# --------------------------------------------------------------------------- OCR path


def test_rasterize_uses_the_requested_dpi() -> None:
    with open_pdf("payment_advice.pdf") as doc:
        image = rasterize(doc[0], dpi=100)

    assert image.mode == "L"
    assert image.size == pytest.approx((595 * 100 / 72, 842 * 100 / 72), abs=1)


def test_layout_pages_routes_digital_pages_past_ocr() -> None:
    pages = layout_pages(FIXTURES / "contract.pdf", "application/pdf")

    assert [type(p) for p in pages] == [PageContent, PageContent]
    assert [p.page_number for p in pages] == [1, 2]


def test_layout_pages_sends_scanned_pdf_pages_to_ocr_at_configured_dpi() -> None:
    (page,) = layout_pages(scanned_pdf(), "application/pdf")

    assert isinstance(page, ScannedPage)
    assert page.dpi == 300  # ocr.dpi in ingestion.yaml
    assert page.image.width == pytest.approx(500 * 300 / 150, abs=2)
    assert page.scale == pytest.approx(72 / 300)


def test_layout_pages_sends_images_and_every_tiff_frame_to_ocr() -> None:
    (receipt,) = layout_pages(RECEIPT, "image/png")
    tiff = io.BytesIO()
    frames = [Image.new("L", (40, 30), shade) for shade in (255, 128)]
    frames[0].save(tiff, format="TIFF", save_all=True, append_images=frames[1:], dpi=(200, 200))

    pages = layout_pages(tiff.getvalue(), "image/tiff")

    assert isinstance(receipt, ScannedPage) and receipt.scale == 1.0
    assert [(p.page_number, p.dpi, p.image.size) for p in pages if isinstance(p, ScannedPage)] == [
        (1, 200, (40, 30)),
        (2, 200, (40, 30)),
    ]


def test_ocr_words_are_scaled_to_pdf_points() -> None:
    page = ScannedPage(1, Image.new("L", (600, 300)), dpi=300, scale=72 / 300)

    content = page.to_page_content([Word(text="Total", bbox=BBox(100, 100, 200, 140), conf=91.5)])

    assert content.source == "ocr"
    assert (content.width, content.height) == pytest.approx((144, 72))
    assert content.words[0].bbox == pytest.approx(BBox(24, 24, 48, 33.6))
    assert content.words[0].conf == 91.5
