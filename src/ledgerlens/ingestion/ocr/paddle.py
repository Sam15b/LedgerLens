"""OCR with PaddleOCR -- optional, often better than Tesseract on noisy receipts.

Not installed by default (it pulls in PaddlePaddle). To use it::

    pip install paddlepaddle paddleocr     # see paddlepaddle.org.cn for GPU builds

and set ``ocr.engine: paddle`` in ``configs/ingestion.yaml``. Without the package,
creating the engine raises :class:`~ledgerlens.ingestion.ocr.base.OCRUnavailableError`.

Written against the PaddleOCR 3.x ``predict`` API. PaddleOCR detects text *lines* rather
than words, so each :class:`OCRWord` here is one detected line segment; layout groups
them into lines and cells the same way.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Iterable, Sequence
from typing import Any, Protocol

import numpy as np
from PIL import Image

from ledgerlens.core.config import OCRConfig, get_config
from ledgerlens.ingestion.layout import BBox
from ledgerlens.ingestion.ocr.base import OCRError, OCRUnavailableError, OCRWord


class PaddlePredictor(Protocol):
    """The part of ``paddleocr.PaddleOCR`` this engine uses."""

    def predict(self, input: Any) -> Iterable[Any]: ...


class PaddleEngine:
    """:class:`~ledgerlens.ingestion.ocr.base.OCREngine` backed by PaddleOCR.

    The model is loaded on first use (it downloads weights the first time). Pass
    ``predictor`` to supply a ready (or fake) ``PaddleOCR`` instance.
    """

    name = "paddle"

    def __init__(
        self,
        config: OCRConfig | None = None,
        *,
        lang: str = "en",
        predictor: PaddlePredictor | None = None,
    ) -> None:
        self.config = config or get_config().ingestion.ocr
        self.lang = lang
        self._predictor = predictor
        if predictor is None and not is_available():
            raise OCRUnavailableError(
                "paddleocr is not installed; `pip install paddlepaddle paddleocr` "
                "or set ocr.engine: tesseract"
            )

    @property
    def predictor(self) -> PaddlePredictor:
        if self._predictor is None:
            module = importlib.import_module("paddleocr")
            # Pages are already upright scans: skip the orientation / unwarping models.
            self._predictor = module.PaddleOCR(
                lang=self.lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        return self._predictor

    def run(self, image: Image.Image) -> list[OCRWord]:
        array = np.asarray(image.convert("RGB"))[:, :, ::-1]  # PaddleOCR expects BGR
        try:
            results = list(self.predictor.predict(array))
        except Exception as exc:  # PaddlePaddle raises many unrelated error types
            raise OCRError(f"paddleocr failed: {exc}") from exc
        words: list[OCRWord] = []
        for result in results:
            words.extend(self.words_from_result(result))
        return words

    def words_from_result(self, result: Any) -> list[OCRWord]:
        """Turn one PaddleOCR page result (``rec_texts`` / ``rec_scores`` / boxes) into words."""
        texts: Sequence[str] = result["rec_texts"]
        scores: Sequence[float] = result["rec_scores"]
        boxes = _boxes(result, len(texts))
        words: list[OCRWord] = []
        for text, score, box in zip(texts, scores, boxes, strict=True):
            conf = float(score) * 100  # PaddleOCR scores are 0-1
            if text.strip() and conf >= self.config.min_confidence:
                words.append(OCRWord(text=text.strip(), bbox=box, conf=min(conf, 100.0)))
        return words


def _boxes(result: Any, count: int) -> list[BBox]:
    """Axis-aligned boxes: ``rec_boxes`` if present, else the bounds of ``rec_polys``."""
    keys = result.keys() if hasattr(result, "keys") else ()
    if "rec_boxes" in keys and len(result["rec_boxes"]) == count:
        return [BBox(*(float(v) for v in box[:4])) for box in result["rec_boxes"]]
    boxes = []
    for poly in result["rec_polys"]:
        xs = [float(point[0]) for point in poly]
        ys = [float(point[1]) for point in poly]
        boxes.append(BBox(min(xs), min(ys), max(xs), max(ys)))
    return boxes


def is_available() -> bool:
    """True if the ``paddleocr`` package can be imported."""
    return importlib.util.find_spec("paddleocr") is not None
