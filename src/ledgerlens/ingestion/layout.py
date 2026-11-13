"""Page layout: decide which pages need OCR, and turn words into lines, blocks and tables.

A digital PDF already carries its text, so those pages skip OCR entirely: a page whose
text layer has at least ``layout.text_layer_min_chars`` characters (``configs/ingestion.yaml``)
is read directly with PyMuPDF. Anything below that -- a scan saved as PDF, or a page
with only a logo and a signature -- goes down the OCR path: it is rasterized at
``ocr.dpi`` and handed over as a :class:`ScannedPage`. Images (PNG, JPEG, TIFF frames)
always take the OCR path. :func:`layout_pages` makes that call for every page.

Both paths produce a :class:`PageContent`: the page's words with their bounding boxes,
grouped into lines, blocks and (heuristically) tables.
Coordinates are PDF points (1/72 inch, origin top-left) for PDF pages and pixels for
images.
"""

from __future__ import annotations

import io
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

import pymupdf
from PIL import Image, ImageSequence
from pydantic import BaseModel, ConfigDict, Field

from ledgerlens.core.config import LayoutConfig, get_config

PDF_POINTS_PER_INCH = 72

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


class Cell(BaseModel):
    """Words of one line that sit together, separated from the next cell by a wide gap."""

    model_config = ConfigDict(frozen=True)

    text: str
    bbox: BBox


class Line(BaseModel):
    """Words sharing a baseline, left to right, split into :class:`Cell` s at wide gaps."""

    model_config = ConfigDict(frozen=True)

    words: list[Word] = Field(min_length=1)
    cells: list[Cell] = Field(min_length=1)
    bbox: BBox

    @property
    def text(self) -> str:
        """Cells joined by two spaces, so column gaps stay visible in plain text."""
        return CELL_SEPARATOR.join(cell.text for cell in self.cells)


class Block(BaseModel):
    """Consecutive lines with no paragraph-sized gap between them."""

    model_config = ConfigDict(frozen=True)

    lines: list[Line] = Field(min_length=1)
    bbox: BBox

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


class Table(BaseModel):
    """Consecutive multi-cell lines whose cells line up in columns. Row 0 is the header."""

    model_config = ConfigDict(frozen=True)

    rows: list[list[str]] = Field(min_length=2)
    bbox: BBox

    @property
    def n_columns(self) -> int:
        return len(self.rows[0])

    def to_markdown(self) -> str:
        def row(cells: list[str]) -> str:
            return "| " + " | ".join(cell.replace("|", r"\|") for cell in cells) + " |"

        header, *body = self.rows
        divider = "| " + " | ".join("---" for _ in header) + " |"
        return "\n".join([row(header), divider, *(row(cells) for cells in body)])


class PageContent(BaseModel):
    """Everything layout knows about one page, whichever path (text layer or OCR) read it."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    source: PageSource
    words: list[Word]
    lines: list[Line] = Field(default_factory=list)
    blocks: list[Block] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)

    @classmethod
    def from_words(
        cls,
        words: Sequence[Word],
        *,
        page_number: int,
        width: float,
        height: float,
        source: PageSource,
    ) -> PageContent:
        """Group ``words`` into lines, blocks and tables. Used by both the text and OCR paths."""
        lines = group_lines(words)
        return cls(
            page_number=page_number,
            width=width,
            height=height,
            source=source,
            words=[word for line in lines for word in line.words],
            lines=lines,
            blocks=group_blocks(lines),
            tables=find_tables(lines),
        )

    @property
    def text(self) -> str:
        """The page as plain text: blocks separated by a blank line."""
        return "\n\n".join(block.text for block in self.blocks)


# --------------------------------------------------------------------------- grouping

CELL_SEPARATOR = "  "
# Thresholds are multiples of the text height, so they work for PDF points and OCR pixels.
_SAME_LINE_OFFSET = 0.5  # centre offset (of the shorter height) still on the same line
_CELL_GAP = 1.0  # horizontal gap that starts a new cell
_BLOCK_GAP = 0.5  # vertical gap that starts a new block
_COLUMN_TOLERANCE = 0.75  # how far cell edges may drift and still be one column


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def group_lines(words: Sequence[Word]) -> list[Line]:
    """Group words into lines, top to bottom, each left to right.

    A word joins the row whose centre line (the mean of its words' vertical centres) is
    within half a text height of the word's own centre -- for words of equal height, the
    same as "boxes overlap by at least half". Rows are compared by centre, not by the
    union of their boxes: a tall heading beside two lines of small text (a 20 pt
    "PURCHASE ORDER" next to a name and an address) would otherwise grow one row's box
    over both lines and merge them, interleaving their words.

    Uses geometry only (not PDF line numbers), so OCR words group the same way.
    """
    rows: list[list[Word]] = []
    centres: list[float] = []  # mean vertical centre of each row's words
    heights: list[float] = []  # mean height of each row's words
    for word in sorted(words, key=lambda w: (w.bbox.center_y, w.bbox.x0)):
        best, best_offset = None, float("inf")
        for i, (centre, height) in enumerate(zip(centres, heights, strict=True)):
            offset = abs(word.bbox.center_y - centre)
            if offset <= _SAME_LINE_OFFSET * min(word.bbox.height, height) and offset < best_offset:
                best, best_offset = i, offset
        if best is None:
            rows.append([word])
            centres.append(word.bbox.center_y)
            heights.append(word.bbox.height)
            continue
        rows[best].append(word)
        n = len(rows[best])
        centres[best] += (word.bbox.center_y - centres[best]) / n
        heights[best] += (word.bbox.height - heights[best]) / n
    lines = [_make_line(sorted(row, key=lambda w: w.bbox.x0)) for row in rows]
    return sorted(lines, key=lambda line: (line.bbox.y0, line.bbox.x0))


def _make_line(words: list[Word]) -> Line:
    height = _median([w.bbox.height for w in words])
    groups: list[list[Word]] = [[words[0]]]
    for prev, word in zip(words, words[1:], strict=False):
        if word.bbox.x0 - prev.bbox.x1 > _CELL_GAP * height:
            groups.append([word])
        else:
            groups[-1].append(word)
    cells = [
        Cell(text=" ".join(w.text for w in group), bbox=BBox.union(w.bbox for w in group))
        for group in groups
    ]
    return Line(words=words, cells=cells, bbox=BBox.union(w.bbox for w in words))


def _starts_new_block(prev: Line, line: Line) -> bool:
    gap = line.bbox.y0 - prev.bbox.y1
    return gap > _BLOCK_GAP * max(prev.bbox.height, line.bbox.height)


def group_blocks(lines: Sequence[Line]) -> list[Block]:
    """Split top-to-bottom ``lines`` into blocks wherever a paragraph-sized gap appears."""
    runs: list[list[Line]] = []
    for line in lines:
        if runs and not _starts_new_block(runs[-1][-1], line):
            runs[-1].append(line)
        else:
            runs.append([line])
    return [Block(lines=run, bbox=BBox.union(line.bbox for line in run)) for run in runs]


class _Column:
    def __init__(self, box: BBox) -> None:
        self.x0, self.x1 = box.x0, box.x1

    def matches(self, box: BBox, tolerance: float) -> bool:
        # left-aligned text shares x0, right-aligned amounts share x1
        return abs(box.x0 - self.x0) <= tolerance or abs(box.x1 - self.x1) <= tolerance


def find_tables(lines: Sequence[Line]) -> list[Table]:
    """Find tables: runs of 2+ adjacent lines with 2+ cells whose cells line up in columns.

    A heuristic for the simple ruled-less tables on invoices, POs and rate cards; the
    first row of a run is taken as the header.
    """
    tables: list[Table] = []
    run: list[Line] = []
    columns: list[_Column] = []
    for line in [*lines, None]:
        if line is not None and len(line.cells) >= 2:
            tolerance = _COLUMN_TOLERANCE * line.bbox.height
            if run and not _starts_new_block(run[-1], line):
                unmatched = [
                    c
                    for c in line.cells
                    if not any(col.matches(c.bbox, tolerance) for col in columns)
                ]
                if len(line.cells) - len(unmatched) >= 2 and len(unmatched) <= 1:
                    run.append(line)
                    columns.extend(_Column(c.bbox) for c in unmatched)
                    continue
            _close(run, columns, tables)
            run, columns = [line], [_Column(c.bbox) for c in line.cells]
        else:
            _close(run, columns, tables)
            run, columns = [], []
    return tables


def _close(run: list[Line], columns: list[_Column], tables: list[Table]) -> None:
    if len(run) < 2:
        return
    columns = sorted(columns, key=lambda col: col.x0)
    rows: list[list[str]] = []
    for line in run:
        tolerance = _COLUMN_TOLERANCE * line.bbox.height
        cells = [""] * len(columns)
        for cell in line.cells:
            index = next(
                (i for i, col in enumerate(columns) if col.matches(cell.bbox, tolerance)),
                min(range(len(columns)), key=lambda i: abs(columns[i].x0 - cell.bbox.x0)),
            )
            cells[index] = f"{cells[index]} {cell.text}".strip()
        rows.append(cells)
    tables.append(Table(rows=rows, bbox=BBox.union(line.bbox for line in run)))


# --------------------------------------------------------------------------- text layer

# PyMuPDF keeps typographic ligatures by default, so "files" comes out as "ﬁles" when
# the font set it with an fi ligature. Label patterns, amounts and vendor matching all
# compare plain letters, so ligatures are expanded on the way out.
_TEXT_FLAGS = pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_PRESERVE_LIGATURES
_WORD_FLAGS = pymupdf.TEXTFLAGS_WORDS & ~pymupdf.TEXT_PRESERVE_LIGATURES


def text_layer_chars(page: pymupdf.Page) -> int:
    """Non-whitespace characters in the page's text layer."""
    text: str = page.get_text("text", flags=_TEXT_FLAGS)
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
    ``(block, line, word)`` keeps the PDF's own order. Ligatures are expanded.
    """
    raw: list[tuple[float, float, float, float, str, int, int, int]] = page.get_text(
        "words", flags=_WORD_FLAGS
    )
    raw.sort(key=lambda w: (w[5], w[6], w[7]))
    return [
        Word(text=text, bbox=BBox(x0, y0, x1, y1))
        for x0, y0, x1, y1, text, *_ in raw
        if text.strip()
    ]


def read_text_page(page: pymupdf.Page) -> PageContent:
    """:class:`PageContent` for a page that has a text layer."""
    return PageContent.from_words(
        extract_words(page),
        page_number=page.number + 1,
        width=page.rect.width,
        height=page.rect.height,
        source="text_layer",
    )


# --------------------------------------------------------------------------- OCR path


@dataclass(frozen=True)
class ScannedPage:
    """A page with no usable text layer, rasterized for OCR.

    ``scale`` converts OCR pixel coordinates to the page's units: ``72 / dpi`` for PDF
    pages (so OCR boxes come back in PDF points, like text-layer pages), ``1.0`` for
    images (which stay in pixels).
    """

    page_number: int
    image: Image.Image
    dpi: int
    scale: float

    def to_page_content(self, words: Sequence[Word]) -> PageContent:
        """Wrap OCR ``words`` (in image pixels) as this page's :class:`PageContent`."""
        scaled = [w.model_copy(update={"bbox": w.bbox.scaled(self.scale)}) for w in words]
        return PageContent.from_words(
            scaled,
            page_number=self.page_number,
            width=self.image.width * self.scale,
            height=self.image.height * self.scale,
            source="ocr",
        )


def rasterize(page: pymupdf.Page, dpi: int | None = None) -> Image.Image:
    """Render ``page`` to an 8-bit grayscale image at ``dpi`` (default ``ocr.dpi``, 300)."""
    if dpi is None:
        dpi = get_config().ingestion.ocr.dpi
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
    return Image.frombytes("L", (pix.width, pix.height), pix.samples)


def layout_pages(
    source: Path | bytes,
    mime_type: str,
    *,
    min_chars: int | None = None,
    dpi: int | None = None,
) -> list[PageContent | ScannedPage]:
    """Read every page of a loaded file, in order.

    PDF pages with a text layer come back as :class:`PageContent`; scanned PDF pages and
    image frames come back as :class:`ScannedPage` for OCR.
    """
    if mime_type == "application/pdf":
        return _pdf_pages(source, min_chars, dpi)
    return _image_pages(source)


def _pdf_pages(
    source: Path | bytes, min_chars: int | None, dpi: int | None
) -> list[PageContent | ScannedPage]:
    if dpi is None:
        dpi = get_config().ingestion.ocr.dpi
    doc = (
        pymupdf.open(source)
        if isinstance(source, Path)
        else pymupdf.open(stream=source, filetype="pdf")
    )
    pages: list[PageContent | ScannedPage] = []
    with doc:
        for index in range(doc.page_count):
            page = doc[index]
            if has_text_layer(page, min_chars):
                pages.append(read_text_page(page))
            else:
                image = rasterize(page, dpi)
                pages.append(ScannedPage(index + 1, image, dpi, PDF_POINTS_PER_INCH / dpi))
    return pages


def _image_pages(source: Path | bytes) -> list[PageContent | ScannedPage]:
    pages: list[PageContent | ScannedPage] = []
    with Image.open(source if isinstance(source, Path) else io.BytesIO(source)) as image:
        dpi = _image_dpi(image)
        for number, frame in enumerate(ImageSequence.Iterator(image), start=1):
            pages.append(ScannedPage(number, frame.convert("L"), dpi, 1.0))
    return pages


def _image_dpi(image: Image.Image) -> int:
    """The DPI an image was saved with (scanners record it), else the OCR default."""
    info = image.info.get("dpi")
    if isinstance(info, tuple) and info and float(info[0]) > 1:
        return round(float(info[0]))
    return get_config().ingestion.ocr.dpi


def _layout_config() -> LayoutConfig:
    return get_config().ingestion.layout
