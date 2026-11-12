"""OCR with Tesseract (via ``pytesseract``).

Needs the ``tesseract`` binary: ``apt install tesseract-ocr`` / ``brew install tesseract`` /
``choco install tesseract`` on Windows. If it is not on ``PATH``, set ``ocr.tesseract_cmd``
in ``configs/ingestion.yaml`` to its full path.

Settings (``ocr`` in ``configs/ingestion.yaml``): page segmentation mode ``tesseract_psm``
(6 = one uniform block of text, which suits receipts and invoices), and
``min_confidence`` -- words Tesseract is less sure of than this (0-100) are dropped.

Tesseract is tuned for text scanned at ~300 DPI. Small images (phone crops, low-DPI
scans) are upscaled by a whole factor until the long side reaches
:data:`MIN_LONG_SIDE_PX`, and the word boxes are scaled back, so callers always get
coordinates in the pixels of the image they passed in.
"""

from __future__ import annotations

import shutil
from typing import Any

import pytesseract
from PIL import Image

from ledgerlens.core.config import OCRConfig, get_config
from ledgerlens.ingestion.layout import BBox
from ledgerlens.ingestion.ocr.base import OCRError, OCRUnavailableError, OCRWord
from ledgerlens.ingestion.ocr.preprocess import preprocess

MIN_LONG_SIDE_PX = 1600
MAX_UPSCALE = 4


def upscale_factor(size: tuple[int, int]) -> int:
    """Whole factor (1-4) that brings the image's long side up to ``MIN_LONG_SIDE_PX``."""
    long_side = max(size)
    if long_side <= 0:
        return 1
    return max(1, min(MAX_UPSCALE, -(-MIN_LONG_SIDE_PX // long_side)))


class TesseractEngine:
    """:class:`~ledgerlens.ingestion.ocr.base.OCREngine` backed by the Tesseract binary."""

    name = "tesseract"

    def __init__(
        self, config: OCRConfig | None = None, *, lang: str = "eng", timeout_s: float = 60
    ) -> None:
        self.config = config or get_config().ingestion.ocr
        self.lang = lang
        self.timeout_s = timeout_s

    @property
    def command(self) -> str:
        return self.config.tesseract_cmd or "tesseract"

    def is_available(self) -> bool:
        """True if the Tesseract binary can be found."""
        return shutil.which(self.command) is not None

    def run(self, image: Image.Image) -> list[OCRWord]:
        """Preprocess ``image`` (per ``ocr.preprocess``), OCR it, drop low-confidence words."""
        factor = upscale_factor(image.size)
        if factor > 1:
            image = image.resize(
                (image.width * factor, image.height * factor), Image.Resampling.LANCZOS
            )
        cleaned = preprocess(image, self.config.preprocess).image
        pytesseract.pytesseract.tesseract_cmd = self.command
        try:
            data: dict[str, list[Any]] = pytesseract.image_to_data(
                cleaned,
                lang=self.lang,
                config=f"--psm {self.config.tesseract_psm}",
                output_type=pytesseract.Output.DICT,
                timeout=self.timeout_s,
            )
        except pytesseract.TesseractNotFoundError as exc:
            raise OCRUnavailableError(
                f"tesseract binary not found ({self.command!r}); install it or set "
                "ocr.tesseract_cmd in configs/ingestion.yaml"
            ) from exc
        except (pytesseract.TesseractError, RuntimeError) as exc:  # RuntimeError: timeout
            raise OCRError(f"tesseract failed: {exc}") from exc
        return self.words_from_data(data, scale=1 / factor)

    def words_from_data(self, data: dict[str, list[Any]], *, scale: float = 1.0) -> list[OCRWord]:
        """Turn ``image_to_data`` output into words, in Tesseract's reading order.

        Rows with ``conf == -1`` are page/block/line markers, not words. Boxes are
        multiplied by ``scale`` (to undo upscaling).
        """
        words: list[OCRWord] = []
        for text, conf, left, top, width, height in zip(
            data["text"],
            data["conf"],
            data["left"],
            data["top"],
            data["width"],
            data["height"],
            strict=True,
        ):
            text = str(text).strip()
            confidence = float(conf)
            if not text or confidence < 0 or confidence < self.config.min_confidence:
                continue
            x0, y0 = float(left), float(top)
            words.append(
                OCRWord(
                    text=text,
                    bbox=BBox(x0, y0, x0 + float(width), y0 + float(height)).scaled(scale),
                    conf=min(confidence, 100.0),
                )
            )
        return words
