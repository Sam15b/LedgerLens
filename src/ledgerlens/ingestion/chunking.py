"""Split a document into retrieval chunks, according to its type.

========================  ==========================================================
doc_type                  chunks
========================  ==========================================================
invoice, receipt, PO      one **header** chunk (parties, numbers, dates, totals) and one
                          **line_items** chunk (the item table as markdown)
contract                  one **clause** chunk per clause heading (``1.``, ``Section 2``,
                          ``ARTICLE IV``, ``Schedule 1`` ...); a clause longer than
                          ``chunking.contract.max_tokens`` (512) is split recursively --
                          paragraphs, then lines, sentences, words -- with up to
                          ``overlap`` (50) tokens repeated between neighbours
payment, unknown          one **body** chunk
========================  ==========================================================

Tokens are counted with the embedding model's own tokenizer (``embedding.dense_model``),
since its input limit is what the 512 protects; if that tokenizer cannot be loaded
(offline, package missing) a word-and-punctuation count is used instead.

Short, number-heavy documents retrieve better as a few coherent chunks than as fixed-size
windows: a question about "the invoice for PO-88231" should hit the header chunk, one about
"shipping labels" the item table.
"""

from __future__ import annotations

import functools
import logging
import re
from collections.abc import Callable, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ledgerlens.core.config import ChunkingConfig, get_config
from ledgerlens.core.ids import chunk_vector_id
from ledgerlens.core.schemas import DocType
from ledgerlens.ingestion.layout import BBox, Line, PageContent, Table

logger = logging.getLogger(__name__)

ChunkKind = Literal["header", "line_items", "clause", "body"]
TokenCounter = Callable[[str], int]

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


def chunk_document(
    pages: Sequence[PageContent],
    doc_type: DocType | str,
    *,
    config: ChunkingConfig | None = None,
    count_tokens: TokenCounter | None = None,
) -> list[Chunk]:
    """Chunks for a document whose pages have been read (text layer or OCR).

    ``count_tokens`` defaults to the embedding model's tokenizer; only contracts need it.
    """
    kind = DocType(doc_type)
    config = config or get_config().ingestion.chunking
    if kind in (DocType.INVOICE, DocType.RECEIPT, DocType.PURCHASE_ORDER):
        drafts = _itemized(pages)
    elif kind is DocType.CONTRACT:
        counter = count_tokens or default_token_counter()
        drafts = _clauses(pages, config.contract.max_tokens, config.contract.overlap, counter)
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


# --------------------------------------------------------------------------- contracts

# "1. Term", "4.2 Fees", "Section 3", "ARTICLE IV", "Clause 7", "Schedule 1 - Rate card"
CLAUSE_HEADING = re.compile(
    r"^\s*(?:"
    r"\d+(?:\.\d+)*\.?\s+[A-Z]"
    r"|(?i:section|article|clause)\s+(?:\d+(?:\.\d+)*|[IVXLC]+)\b"
    r"|(?i:schedule|exhibit|annex|appendix)\s+[\dA-Z]+\b"
    r")"
)
SEPARATORS = ("\n\n", "\n", ". ", " ")


def is_clause_heading(text: str) -> bool:
    return CLAUSE_HEADING.match(text) is not None


def _clauses(
    pages: Sequence[PageContent], max_tokens: int, overlap: int, count: TokenCounter
) -> list[_Draft]:
    sections: list[tuple[str | None, list[tuple[int, Line]]]] = [(None, [])]
    for page in pages:
        for line in page.lines:
            if is_clause_heading(line.text):
                sections.append((line.text, []))
            sections[-1][1].append((page.page_number, line))
    drafts: list[_Draft] = []
    for title, lines in sections:
        if not lines:
            continue
        pages_used = sorted({number for number, _ in lines})
        text = "\n\n".join(
            _lines_text([line for number, line in lines if number == page_number])
            for page_number in pages_used
        )
        for part in split_text(text, max_tokens, overlap, count):
            drafts.append(_Draft(kind="clause", text=part, pages=pages_used, title=title))
    return drafts


def split_text(text: str, max_tokens: int, overlap: int, count: TokenCounter) -> list[str]:
    """Split ``text`` into parts of at most ``max_tokens``, at the most natural boundary.

    Tries paragraph breaks, then line breaks, sentence ends, and spaces. Consecutive parts
    share up to ``overlap`` tokens: whole trailing pieces (sentences, lines) of the previous
    part, so the overlap never starts mid-word.
    """
    if count(text) <= max_tokens:
        return [text.strip()] if text.strip() else []
    pieces = _pieces(text, max_tokens, count, 0)
    sizes = [count(piece) for piece in pieces]
    parts: list[str] = []
    window: list[int] = []  # indices into pieces
    for i, size in enumerate(sizes):
        if window and sum(sizes[j] for j in window) + size > max_tokens:
            parts.append("".join(pieces[j] for j in window).strip())
            tail: list[int] = []
            for j in reversed(window):
                if sum(sizes[k] for k in tail) + sizes[j] > overlap:
                    break
                tail.insert(0, j)
            window = tail if sum(sizes[k] for k in tail) + size <= max_tokens else []
        window.append(i)
    if window:
        parts.append("".join(pieces[j] for j in window).strip())
    return [part for part in parts if part]


def _pieces(text: str, max_tokens: int, count: TokenCounter, level: int) -> list[str]:
    """Cut ``text`` at ``SEPARATORS[level]`` (kept on the left piece); recurse on big pieces."""
    if level >= len(SEPARATORS):
        return _hard_split(text, max_tokens, count)
    sep = SEPARATORS[level]
    raw = text.split(sep)
    pieces = [part + sep for part in raw[:-1]] + [raw[-1]]
    out: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        if count(piece) <= max_tokens:
            out.append(piece)
        else:
            out.extend(_pieces(piece, max_tokens, count, level + 1))
    return out


def _hard_split(text: str, max_tokens: int, count: TokenCounter) -> list[str]:
    """Last resort for a single 'word' longer than the limit: cut it by characters."""
    tokens = max(count(text), 1)
    step = max(1, len(text) * max_tokens // tokens)
    return [text[i : i + step] for i in range(0, len(text), step)]


# --------------------------------------------------------------------------- tokens

_APPROX_TOKEN = re.compile(r"\w+|[^\w\s]")


def approx_token_count(text: str) -> int:
    """Words plus punctuation marks: close to, and never above, a WordPiece token count."""
    return len(_APPROX_TOKEN.findall(text))


@functools.lru_cache(maxsize=4)
def default_token_counter(model_name: str | None = None) -> TokenCounter:
    """Token counter for the embedding model (``embedding.dense_model``), loaded once.

    Uses the Hugging Face ``tokenizers`` package (installed with sentence-transformers);
    falls back to :func:`approx_token_count` if it is missing or the tokenizer cannot be
    downloaded.
    """
    name = model_name or get_config().ingestion.embedding.dense_model
    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_pretrained(name)
    except Exception as exc:  # ImportError, network / hub errors
        logger.warning("tokenizer for %s unavailable (%s); approximating token counts", name, exc)
        return approx_token_count

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    return count
