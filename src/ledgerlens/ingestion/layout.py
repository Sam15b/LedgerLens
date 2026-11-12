"""Page layout: decide which pages need OCR, and turn words into lines, blocks and tables.

A digital PDF already carries its text, so those pages skip OCR entirely: a page whose
text layer has at least ``layout.text_layer_min_chars`` characters (``configs/ingestion.yaml``)
is read directly with PyMuPDF. Anything below that -- a scan saved as PDF, or a page
with only a logo and a signature -- goes down the OCR path.
"""

from __future__ import annotations

import pymupdf

from ledgerlens.core.config import LayoutConfig, get_config


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


def _layout_config() -> LayoutConfig:
    return get_config().ingestion.layout
