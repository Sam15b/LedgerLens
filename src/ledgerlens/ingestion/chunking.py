"""Split a document into retrieval chunks, according to its type.

========================  ==========================================================
doc_type                  chunks
========================  ==========================================================
invoice, receipt, PO      one **header** chunk (parties, numbers, dates, totals) and one
                          **line_items** chunk (the item table as markdown)
payment, unknown          one **body** chunk
========================  ==========================================================

Short, number-heavy documents retrieve better as a few coherent chunks than as fixed-size
windows: a question about "the invoice for PO-88231" should hit the header chunk, one about
"shipping labels" the item table.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ledgerlens.core.ids import chunk_vector_id
from ledgerlens.core.schemas import DocType
from ledgerlens.ingestion.layout import BBox, Line, PageContent, Table

ChunkKind = Literal["header", "line_items", "clause", "body"]

_ITEM_HEADER_WORDS = ("description", "item", "qty", "quantity", "unit price", "price", "amount")
_TOTAL_WORDS = ("subtotal", "sub-total", "total", "tax", "vat", "gst", "sst", "discount")


class Chunk(BaseModel):
    """A piece of a document that is embedded and retrieved on its own."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    doc_type: DocType
    kind: ChunkKind
    text: str = Field(min_length=1)
    pages: list[int] = Field(min_length=1)
    title: str | None = None  # clause heading, for contract chunks

    def vector_id(self, document_id: str) -> str:
        """Pinecone id: ``{document_id}#c{index}``."""
        return chunk_vector_id(document_id, self.index)


class _Draft(BaseModel):
    kind: ChunkKind
    text: str
    pages: list[int]
    title: str | None = None


def chunk_document(pages: Sequence[PageContent], doc_type: DocType | str) -> list[Chunk]:
    """Chunks for a document whose pages have been read (text layer or OCR)."""
    kind = DocType(doc_type)
    if kind in (DocType.INVOICE, DocType.RECEIPT, DocType.PURCHASE_ORDER):
        drafts = _itemized(pages)
    else:
        drafts = _single(pages)
    return _number(drafts, kind)


def _number(drafts: Sequence[_Draft], doc_type: DocType) -> list[Chunk]:
    return [
        Chunk(index=i, doc_type=doc_type, kind=d.kind, text=d.text, pages=d.pages, title=d.title)
        for i, d in enumerate(d for d in drafts if d.text.strip())
    ]


def _single(pages: Sequence[PageContent]) -> list[_Draft]:
    text = "\n\n".join(page.text for page in pages if page.text.strip())
    return [_Draft(kind="body", text=text, pages=[p.page_number for p in pages if p.words])]


# --------------------------------------------------------------------------- itemized


def _has_item_header(table: Table) -> bool:
    header = " ".join(table.rows[0]).lower()
    return sum(word in header for word in _ITEM_HEADER_WORDS) >= 2


def _is_total_row(row: list[str]) -> bool:
    first = row[0].lower()
    return any(word in first for word in _TOTAL_WORDS)


def item_rows(table: Table) -> list[list[str]]:
    """The item rows of ``table`` (with its header row, if it has one).

    Rows from the first totals row ("Subtotal", "TOTAL MYR", "VAT 20%") on are not items:
    receipts often print totals straight under the items, in the same columns. A table that
    starts with totals has no items.
    """
    start = 1 if _has_item_header(table) else 0
    end = next(
        (i for i in range(start, len(table.rows)) if _is_total_row(table.rows[i])),
        len(table.rows),
    )
    return table.rows[:end] if end > start else []


def is_line_item_table(table: Table) -> bool:
    return bool(item_rows(table))


def _itemized(pages: Sequence[PageContent]) -> list[_Draft]:
    header_parts: list[str] = []
    header_pages: list[int] = []
    item_tables: list[str] = []
    item_pages: list[int] = []
    for page in pages:
        item_lines: set[int] = set()  # indices into page.lines
        for table in page.tables:
            rows = item_rows(table)
            if not rows:
                continue
            # A table's rows are consecutive page lines, in order, inside its box.
            in_table = [i for i, line in enumerate(page.lines) if _inside(line, table.bbox)]
            item_lines.update(in_table[: len(rows)])
            item_tables.append(
                Table(rows=rows, bbox=table.bbox).to_markdown()
                if len(rows) > 1
                else _single_row_markdown(rows[0])
            )
        kept = [line for i, line in enumerate(page.lines) if i not in item_lines]
        if kept:
            header_parts.append(_lines_text(kept))
            header_pages.append(page.page_number)
        if item_lines:
            item_pages.append(page.page_number)
    drafts = []
    if header_parts:
        drafts.append(_Draft(kind="header", text="\n\n".join(header_parts), pages=header_pages))
    if item_tables:
        drafts.append(_Draft(kind="line_items", text="\n\n".join(item_tables), pages=item_pages))
    return drafts


def _single_row_markdown(row: list[str]) -> str:
    return "| " + " | ".join(cell.replace("|", r"\|") for cell in row) + " |"


def _inside(line: Line, box: BBox) -> bool:
    return box.y0 <= line.bbox.center_y <= box.y1


def _lines_text(lines: Sequence[Line]) -> str:
    """Lines joined by newlines, with a blank line where the page had a paragraph gap."""
    out: list[str] = []
    for prev, line in zip([None, *lines], lines, strict=False):
        if prev is not None and line.bbox.y0 - prev.bbox.y1 > 0.5 * max(
            prev.bbox.height, line.bbox.height
        ):
            out.append("")
        out.append(line.text)
    return "\n".join(out)
