from collections.abc import Iterable
from typing import Any

import pytest
from PIL import Image

from ledgerlens.core.config import OCRConfig
from ledgerlens.ingestion.layout import BBox
from ledgerlens.ingestion.ocr import ENGINES, OCREngine, OCRUnavailableError, get_engine
from ledgerlens.ingestion.ocr import paddle as paddle_module
from ledgerlens.ingestion.ocr.base import OCRError, OCRWord
from ledgerlens.ingestion.ocr.paddle import PaddleEngine
from ledgerlens.ingestion.ocr.tesseract import TesseractEngine


class FakePaddle:
    """Stands in for ``paddleocr.PaddleOCR``: returns one canned page result."""

    def __init__(self, result: dict[str, Any] | Exception) -> None:
        self.result = result
        self.inputs: list[Any] = []

    def predict(self, input: Any) -> Iterable[Any]:
        self.inputs.append(input)
        if isinstance(self.result, Exception):
            raise self.result
        return [self.result]


RESULT = {
    "rec_texts": ["KEDAI KOPI SENTOSA", "TOTAL MYR", "9.50", "smudge"],
    "rec_scores": [0.98, 0.91, 0.87, 0.2],
    "rec_boxes": [[20, 10, 180, 24], [20, 200, 90, 214], [160, 200, 190, 214], [5, 5, 9, 9]],
    "rec_polys": [],
}


def test_default_engine_comes_from_ingestion_yaml() -> None:
    engine = get_engine()

    assert isinstance(engine, TesseractEngine) and isinstance(engine, OCREngine)


def test_engine_by_name_or_config() -> None:
    config = OCRConfig(engine="tesseract", min_confidence=70)

    engine = get_engine(config)

    assert isinstance(engine, TesseractEngine) and engine.config.min_confidence == 70
    assert set(ENGINES) == {"tesseract", "paddle"}


def test_unknown_engine_is_rejected() -> None:
    with pytest.raises(
        ValueError, match=r"unknown OCR engine 'easyocr' \(known: paddle, tesseract\)"
    ):
        get_engine(name="easyocr")


def test_paddle_without_the_package_says_how_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paddle_module, "is_available", lambda: False)

    with pytest.raises(OCRUnavailableError, match="pip install paddlepaddle paddleocr"):
        get_engine(OCRConfig(engine="paddle"))


def test_paddle_selected_from_config_when_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paddle_module, "is_available", lambda: True)

    engine = get_engine(OCRConfig(engine="paddle"))

    assert isinstance(engine, PaddleEngine) and engine.name == "paddle"


def test_paddle_results_become_ocr_words() -> None:
    fake = FakePaddle(RESULT)
    engine = PaddleEngine(OCRConfig(min_confidence=40), predictor=fake)

    words = engine.run(Image.new("RGB", (200, 240), "white"))

    assert words == [
        OCRWord(text="KEDAI KOPI SENTOSA", bbox=BBox(20, 10, 180, 24), conf=98.0),
        OCRWord(text="TOTAL MYR", bbox=BBox(20, 200, 90, 214), conf=91.0),
        OCRWord(text="9.50", bbox=BBox(160, 200, 190, 214), conf=87.0),
    ]
    assert fake.inputs[0].shape == (240, 200, 3)  # BGR array, height first


def test_paddle_polygons_are_used_when_boxes_are_missing() -> None:
    result = {
        "rec_texts": ["9.50"],
        "rec_scores": [0.9],
        "rec_polys": [[[160, 201], [190, 199], [191, 214], [159, 215]]],
    }
    engine = PaddleEngine(OCRConfig(), predictor=FakePaddle(result))

    assert engine.words_from_result(result)[0].bbox == BBox(159, 199, 191, 215)


def test_paddle_failures_are_ocr_errors() -> None:
    engine = PaddleEngine(OCRConfig(), predictor=FakePaddle(MemoryError("out of memory")))

    with pytest.raises(OCRError, match="paddleocr failed: out of memory"):
        engine.run(Image.new("L", (10, 10)))
