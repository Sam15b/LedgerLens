"""OCR engines behind one interface; ``ocr.engine`` in ``configs/ingestion.yaml`` picks one.

engine = get_engine()            # TesseractEngine or PaddleEngine, per config
words = engine.run(page.image)
"""

from __future__ import annotations

from collections.abc import Callable

from ledgerlens.core.config import OCRConfig, get_config
from ledgerlens.ingestion.ocr.base import OCREngine, OCRError, OCRUnavailableError, OCRWord

EngineFactory = Callable[[OCRConfig], OCREngine]


def _tesseract(config: OCRConfig) -> OCREngine:
    from ledgerlens.ingestion.ocr.tesseract import TesseractEngine

    return TesseractEngine(config)


def _paddle(config: OCRConfig) -> OCREngine:
    from ledgerlens.ingestion.ocr.paddle import PaddleEngine

    return PaddleEngine(config)


ENGINES: dict[str, EngineFactory] = {"tesseract": _tesseract, "paddle": _paddle}


def get_engine(config: OCRConfig | None = None, *, name: str | None = None) -> OCREngine:
    """Build the OCR engine named by ``name``, else by ``config.engine`` (``ocr.engine``).

    Raises ``ValueError`` for an unknown name and :class:`OCRUnavailableError` if the
    engine's package is not installed.
    """
    config = config or get_config().ingestion.ocr
    name = name or config.engine
    try:
        factory = ENGINES[name]
    except KeyError:
        known = ", ".join(sorted(ENGINES))
        raise ValueError(f"unknown OCR engine {name!r} (known: {known})") from None
    return factory(config)


__all__ = [
    "ENGINES",
    "OCREngine",
    "OCRError",
    "OCRUnavailableError",
    "OCRWord",
    "get_engine",
]
