"""The interface every OCR engine implements.

An engine takes one page image and returns the words it read, each with a bounding box in
image pixels and a confidence. Layout (:meth:`ScannedPage.to_page_content`) then scales
the boxes and groups the words exactly like a text-layer page.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from PIL import Image
from pydantic import Field

from ledgerlens.ingestion.layout import Word


class OCRWord(Word):
    """A word read by OCR: ``bbox`` is in image pixels, ``conf`` is 0-100 and required."""

    conf: float = Field(ge=0, le=100)


class OCRError(RuntimeError):
    """OCR could not run (engine missing or crashed), as opposed to reading nothing."""


class OCRUnavailableError(OCRError):
    """The engine's binary or Python package is not installed."""


@runtime_checkable
class OCREngine(Protocol):
    """Reads the words on a page image."""

    name: str

    def run(self, image: Image.Image) -> list[OCRWord]:
        """Words on ``image`` in reading order, already filtered by minimum confidence."""
        ...
