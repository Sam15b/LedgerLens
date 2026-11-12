import shutil
from pathlib import Path
from typing import Any

import pytesseract
import pytest
from PIL import Image

from eval.ocr_cer import cer
from ledgerlens.core.config import OCRConfig
from ledgerlens.ingestion.layout import BBox, ScannedPage, layout_pages
from ledgerlens.ingestion.ocr.base import OCREngine, OCRError, OCRUnavailableError, OCRWord
from ledgerlens.ingestion.ocr.tesseract import TesseractEngine, upscale_factor

RECEIPT = Path(__file__).parents[3] / "fixtures" / "documents" / "receipt_scanned.png"
# What receipt_scanned.png says (see make_fixtures.py), without the dashed rules.
RECEIPT_TEXT = """\
KEDAI KOPI SENTOSA
12 Jalan Bunga, 50450 Kuala Lumpur
Receipt: R-000732 Date: 14/03/2026
2 x Kopi O 4.40
1 x Roti bakar 5.10
TOTAL MYR 9.50
incl. SST 6% 0.54
Terima kasih!
"""
# Agreed bar for the speckled, tilted 150-DPI receipt fixture (after normalization).
MAX_RECEIPT_CER = 0.10

needs_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="tesseract binary not installed"
)


def image_data(*rows: tuple[str, float, int, int, int, int]) -> dict[str, list[Any]]:
    keys = ("text", "conf", "left", "top", "width", "height")
    return {key: [row[i] for row in rows] for i, key in enumerate(keys)}


class FakeTesseract:
    def __init__(self, data: dict[str, list[Any]]) -> None:
        self.data = data
        self.calls: list[dict[str, Any]] = []

    def __call__(self, image: Image.Image, **kwargs: Any) -> dict[str, list[Any]]:
        self.calls.append({"size": image.size, **kwargs})
        return self.data


def test_engine_satisfies_the_protocol() -> None:
    assert isinstance(TesseractEngine(), OCREngine)


def test_words_below_min_confidence_and_layout_rows_are_dropped() -> None:
    engine = TesseractEngine(OCRConfig(min_confidence=40))
    data = image_data(
        ("", -1, 0, 0, 500, 700),  # page / block / line marker
        ("TOTAL", 96.5, 10, 20, 60, 12),
        ("9.50", 40.0, 200, 20, 30, 12),
        ("~", 12.0, 250, 20, 5, 12),
        ("  ", 95.0, 300, 20, 5, 12),
    )

    words = engine.words_from_data(data)

    assert words == [
        OCRWord(text="TOTAL", bbox=BBox(10, 20, 70, 32), conf=96.5),
        OCRWord(text="9.50", bbox=BBox(200, 20, 230, 32), conf=40.0),
    ]


def test_run_passes_psm_and_maps_boxes_back_after_upscaling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeTesseract(image_data(("TOTAL", 90, 100, 40, 80, 20)))
    monkeypatch.setattr(pytesseract, "image_to_data", fake)
    engine = TesseractEngine(OCRConfig(tesseract_psm=4))

    words = engine.run(Image.new("L", (400, 300), 255))  # upscaled x4 to 1600 px

    assert fake.calls[0]["size"] == (1600, 1200)
    assert fake.calls[0]["config"] == "--psm 4"
    assert words[0].bbox == BBox(25, 10, 45, 15)


def test_upscale_factor() -> None:
    assert upscale_factor((2480, 3508)) == 1  # A4 at 300 DPI
    assert upscale_factor((500, 723)) == 3
    assert upscale_factor((100, 50)) == 4  # capped


def test_missing_binary_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*args: object, **kwargs: object) -> None:
        raise pytesseract.TesseractNotFoundError()

    monkeypatch.setattr(pytesseract, "image_to_data", missing)
    engine = TesseractEngine(OCRConfig(tesseract_cmd="C:/nope/tesseract.exe"))

    assert not engine.is_available()
    with pytest.raises(OCRUnavailableError, match="ocr.tesseract_cmd"):
        engine.run(Image.new("L", (10, 10), 255))


def test_tesseract_failures_are_ocr_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def crash(*args: object, **kwargs: object) -> None:
        raise pytesseract.TesseractError(1, "bad image")

    monkeypatch.setattr(pytesseract, "image_to_data", crash)

    with pytest.raises(OCRError, match="tesseract failed"):
        TesseractEngine().run(Image.new("L", (10, 10), 255))


@needs_tesseract
def test_scanned_receipt_ocr_is_below_agreed_cer() -> None:
    (page,) = layout_pages(RECEIPT, "image/png")
    assert isinstance(page, ScannedPage)

    content = page.to_page_content(TesseractEngine().run(page.image))
    text = "\n".join(line.text for line in content.lines if set(line.text) - set("-_— "))

    score = cer(RECEIPT_TEXT, text)
    assert score <= MAX_RECEIPT_CER, f"CER {score:.3f}\n{text}"
    assert content.source == "ocr"
    assert all(0 <= w.bbox.x0 < w.bbox.x1 <= page.image.width for w in content.words)
