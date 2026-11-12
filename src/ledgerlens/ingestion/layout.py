"""Page layout: decide which pages need OCR, and turn words into lines, blocks and tables.

A digital PDF already carries its text, so those pages skip OCR entirely: a page whose
text layer has at least ``layout.text_layer_min_chars`` characters (``configs/ingestion.yaml``)
is read directly with PyMuPDF. Anything below that -- a scan saved as PDF, or a page
with only a logo and a signature -- goes down the OCR path.

Both paths produce a :class:`PageContent`: the page's words with their bounding boxes.
Coordinates are PDF points (1/72 inch, origin top-left) for PDF pages and pixels for
images.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, NamedTuple

import pymupdf
from pydantic import BaseModel, ConfigDict, Field

from ledgerlens.core.config import LayoutConfig, get_config

PageSource = Literal["text_layer", "ocr"]


class BBox(NamedTuple):
    """Axis-aligned box ``(x0, y0, x1, y1)``, top-left origin."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2

    def scaled(self, factor: float) -> BBox:
        """The box with every coordinate multiplied by ``factor`` (e.g. pixels -> points)."""
        return BBox(self.x0 * factor, self.y0 * factor, self.x1 * factor, self.y1 * factor)

    @staticmethod
    def union(boxes: Iterable[BBox]) -> BBox:
        """Smallest box containing all ``boxes`` (which must not be empty)."""
        x0s, y0s, x1s, y1s = zip(*boxes, strict=True)
        return BBox(min(x0s), min(y0s), max(x1s), max(y1s))


class Word(BaseModel):
    """One word and where it is on the page. ``conf`` (0-100) is set by OCR only."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    bbox: BBox
    conf: float | None = Field(default=None, ge=0, le=100)


class PageContent(BaseModel):
    """Everything layout knows about one page, whichever path (text layer or OCR) read it."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    source: PageSource
    words: list[Word]

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words)


# --------------------------------------------------------------------------- text layer


def text_layer_chars(page: pymupdf.Page) -> int:
    """Non-whitespace characters in the page's text layer."""
    text: str = page.get_text("text")
    return sum(not ch.isspace() for ch in text)


def has_text_layer(page: pymupdf.Page, min_chars: int | None = None) -> bool:
    """True if the page has enough extractable text to skip OCR.

    ``min_chars`` defaults to ``layout.text_layer_min_chars``. Whitespace does not count,
    so a page of layout-only spaces is still treated as scanned.
    """
    if min_chars is None:
        min_chars = _layout_config().text_layer_min_chars
    return text_layer_chars(page) >= min_chars


def pages_needing_ocr(doc: pymupdf.Document, min_chars: int | None = None) -> list[int]:
    """1-based numbers of the pages in ``doc`` that have no usable text layer."""
    return [
        number
        for number in range(1, doc.page_count + 1)
        if not has_text_layer(doc[number - 1], min_chars)
    ]


def extract_words(page: pymupdf.Page) -> list[Word]:
    """Words of the page's text layer with their boxes, in the order the PDF wrote them.

    PyMuPDF returns ``(x0, y0, x1, y1, text, block, line, word)`` tuples; sorting by
    ``(block, line, word)`` keeps the PDF's own order.
    """
    raw: list[tuple[float, float, float, float, str, int, int, int]] = page.get_text("words")
    raw.sort(key=lambda w: (w[5], w[6], w[7]))
    return [
        Word(text=text, bbox=BBox(x0, y0, x1, y1))
        for x0, y0, x1, y1, text, *_ in raw
        if text.strip()
    ]


def read_text_page(page: pymupdf.Page) -> PageContent:
    """:class:`PageContent` for a page that has a text layer."""
    return PageContent(
        page_number=page.number + 1,
        width=page.rect.width,
        height=page.rect.height,
        source="text_layer",
        words=extract_words(page),
    )


def _layout_config() -> LayoutConfig:
    return get_config().ingestion.layout
