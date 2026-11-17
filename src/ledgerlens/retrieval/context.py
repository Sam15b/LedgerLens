"""Context assembly: retrieved chunks and graph facts -> the text an answer is written from.

Every piece of evidence is labelled with citation tags the answer must quote::

    [doc:INV-889] (invoice, p. 1) INVOICE INV-889 Acme Supplies Ltd ... Total 4,200.00
    [doc:INV-889] [doc:PO-1042] Invoice INV-889 (Acme Supplies Ltd, ...) -> PO-1042 -> no payment

and :attr:`AssembledContext.citations` maps each tag back to a ``document_id``, so a
citation in the answer resolves to a real document (:meth:`AssembledContext.resolve`).

A tag is ``doc:<document number>`` when the document has one (``doc:<id prefix>``
otherwise). Two different documents can print the same number -- duplicates exist -- so a
second document claiming a tag gets ``doc:<number>@<id prefix>`` instead of sharing it.
A graph fact about records that came from no document has no tag; it is marked
``[graph]`` and cannot be cited.

The evidence is deduplicated and fitted to a :class:`ContextBudget` -- 3,000 tokens by
default, 60 % for chunks and 40 % for graph facts, at most two chunks per document, one
fact per record -- before it goes into a prompt.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from ledgerlens.ingestion.chunking import approx_token_count
from ledgerlens.retrieval import GraphFact, RetrievalResult, ScoredChunk

UNCITABLE = "graph"
"""The label of a fact with no document behind it."""

_SPACE = re.compile(r"\s+")
_CITATION_GROUP = re.compile(r"\[([^\[\]]*doc:[^\[\]]*)\]")
"""A bracketed citation group in an answer: ``[doc:INV-889]`` or ``[doc:A, doc:B]``."""


def _flat(text: str) -> str:
    return _SPACE.sub(" ", text).strip()


@dataclass(frozen=True)
class ContextItem:
    """One chunk or fact, with the tags that cite it."""

    kind: Literal["chunk", "fact"]
    text: str
    tags: tuple[str, ...]
    document_ids: tuple[str, ...]
    source: str
    """Chunk id, or the graph template that produced the fact."""
    key: str = ""
    """What duplicates share: the chunk id, or the record a fact is about."""

    def render(self) -> str:
        labels = " ".join(f"[{tag}]" for tag in self.tags) or f"[{UNCITABLE}]"
        return f"{labels} {self.text}"


class CitationRegistry:
    """Hands out citation tags: one tag per document, unique across documents."""

    def __init__(self) -> None:
        self.citations: dict[str, str] = {}  # tag -> document id
        self._by_document: dict[str, str] = {}  # document id -> tag

    def tag(self, wanted: str, document_id: str) -> str:
        """The tag for ``document_id``: ``wanted`` unless another document holds it."""
        if document_id in self._by_document:
            return self._by_document[document_id]
        tag = wanted
        if self.citations.get(tag, document_id) != document_id:
            tag = f"{wanted}@{document_id[:8]}"
        self.citations[tag] = document_id
        self._by_document[document_id] = tag
        return tag


def chunk_item(chunk: ScoredChunk, registry: CitationRegistry) -> ContextItem:
    """A chunk as context: its tag, then where it is from, then its text."""
    tag = registry.tag(chunk.citation, chunk.document_id)
    where = ", ".join(
        part for part in (chunk.doc_type, f"p. {chunk.page}" if chunk.page else None) if part
    )
    text = f"({where}) {_flat(chunk.text)}" if where else _flat(chunk.text)
    return ContextItem("chunk", text, (tag,), (chunk.document_id,), chunk.id, chunk.id)


def fact_item(fact: GraphFact, registry: CitationRegistry) -> ContextItem:
    """A graph fact as context, tagged with every document it names."""
    tags = tuple(
        dict.fromkeys(registry.tag(tag, document_id) for tag, document_id in fact.citations.items())
    )
    documents = tuple(dict.fromkeys([*fact.citations.values(), *fact.document_ids]))
    key = fact.subject or _flat(fact.text)
    return ContextItem("fact", _flat(fact.text), tags, documents, fact.source, key)


@dataclass
class CitationCheck:
    """The citations in an answer: those that resolve, and those that do not."""

    resolved: dict[str, str] = field(default_factory=dict)
    """Tag -> document id, in order of first citation."""
    unknown: list[str] = field(default_factory=list)
    """Tags the answer cites that are not in the context (invented or mistyped)."""

    @property
    def ok(self) -> bool:
        """At least one citation, and every citation resolves."""
        return bool(self.resolved) and not self.unknown

    @property
    def document_ids(self) -> list[str]:
        return list(dict.fromkeys(self.resolved.values()))


def cited_tags(answer: str) -> list[str]:
    """Citation tags in ``answer``, in order, once each (``[doc:A] [doc:B]`` and
    ``[doc:A, doc:B]`` are both understood)."""
    tags: list[str] = []
    for group in _CITATION_GROUP.findall(answer):
        for part in group.split(","):
            part = part.strip()
            if part.startswith("doc:") and " " not in part:
                tags.append(part)
    return list(dict.fromkeys(tags))


@dataclass
class AssembledContext:
    """The evidence for one answer, as tagged text, with its citation map."""

    chunks: list[ContextItem] = field(default_factory=list)
    facts: list[ContextItem] = field(default_factory=list)
    citations: dict[str, str] = field(default_factory=dict)
    """Tag -> document id, for every tag in the context."""
    tokens: dict[str, int] = field(default_factory=dict)
    """Tokens used by ``chunks`` and ``facts``."""
    dropped: dict[str, int] = field(default_factory=dict)
    """Chunks and facts left out: duplicates, or over the budget."""

    @property
    def items(self) -> list[ContextItem]:
        return [*self.chunks, *self.facts]

    @property
    def text(self) -> str:
        """Document excerpts, then graph facts, one per line under a heading each."""
        sections = []
        if self.chunks:
            lines = "\n".join(item.render() for item in self.chunks)
            sections.append(f"Document excerpts:\n{lines}")
        if self.facts:
            lines = "\n".join(item.render() for item in self.facts)
            sections.append(f"Facts from the document graph:\n{lines}")
        return "\n\n".join(sections) or "(no evidence found)"

    def resolve(self, answer: str) -> CitationCheck:
        """Map the answer's citations to document ids."""
        check = CitationCheck()
        for tag in cited_tags(answer):
            if tag in self.citations:
                check.resolved[tag] = self.citations[tag]
            else:
                check.unknown.append(tag)
        return check


# --------------------------------------------------------------------------- budget

TokenCounter = Callable[[str], int]


@functools.lru_cache(maxsize=1)
def default_token_counter() -> TokenCounter:
    """Counts tokens as the ``o200k_base`` encoding does (close enough for any provider's
    budget); approximates if tiktoken or its encoding is unavailable."""
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("o200k_base")
    except Exception:  # ImportError, or the encoding cannot be downloaded
        return approx_token_count

    def count(text: str) -> int:
        return len(encoding.encode(text, disallowed_special=()))

    return count


@dataclass(frozen=True)
class ContextBudget:
    """How much evidence goes into a prompt.

    ``tokens`` in all, of which chunks get ``chunk_share`` and graph facts the rest. A
    share one side does not use goes to the other: with no facts, chunks may fill the
    whole budget. At most ``per_document`` chunks come from any one document, so a long
    document cannot crowd out the others.
    """

    tokens: int = 3000
    chunk_share: float = 0.6
    per_document: int = 2

    def __post_init__(self) -> None:
        if self.tokens < 1 or not 0.0 <= self.chunk_share <= 1.0 or self.per_document < 1:
            raise ValueError(f"invalid context budget {self}")


def dedupe_chunks(items: Sequence[ContextItem], per_document: int) -> list[ContextItem]:
    """Chunks in order, without repeats (same chunk id or same text) and at most
    ``per_document`` per document."""
    kept: list[ContextItem] = []
    seen: set[str] = set()
    per: dict[str, int] = {}
    for item in items:
        document = item.document_ids[0]
        if item.source in seen or item.text in seen or per.get(document, 0) >= per_document:
            continue
        seen.update((item.source, item.text))
        per[document] = per.get(document, 0) + 1
        kept.append(item)
    return kept


def dedupe_facts(items: Sequence[ContextItem]) -> list[ContextItem]:
    """Facts in order, one per record: two templates can reach the same invoice, and the
    first (most relevant) rendering is kept. Facts are not deduped by the documents they
    *cite* -- every invoice under one contract cites that contract."""
    kept: list[ContextItem] = []
    seen: set[str] = set()
    for item in items:
        if item.key in seen or item.text in seen:
            continue
        seen.update((item.key, item.text))
        kept.append(item)
    return kept


def _fill(
    items: Sequence[ContextItem], budget: int, count: TokenCounter
) -> tuple[list[ContextItem], int]:
    """The items that fit in ``budget`` tokens, best first, and the tokens they use. An
    item that does not fit is skipped; a shorter one after it may still fit."""
    taken: list[ContextItem] = []
    used = 0
    for item in items:
        size = count(item.render()) + 1  # + the newline
        if used + size <= budget:
            taken.append(item)
            used += size
    return taken, used


def assemble(
    chunks: Iterable[ScoredChunk],
    facts: Iterable[GraphFact],
    *,
    budget: ContextBudget | None = None,
    count: TokenCounter | None = None,
) -> AssembledContext:
    """Tag, dedupe and budget the evidence.

    Chunks are tagged first, so excerpts claim the plain tags. Chunks fill their share,
    facts fill what is left, then chunks may use whatever the facts did not need. Only the
    tags of what made it in are in ``citations``.
    """
    budget = budget or ContextBudget()
    count = count or default_token_counter()
    registry = CitationRegistry()
    all_chunks = [chunk_item(chunk, registry) for chunk in chunks]
    all_facts = [fact_item(fact, registry) for fact in facts]
    chunk_pool = dedupe_chunks(all_chunks, budget.per_document)
    fact_pool = dedupe_facts(all_facts)

    share = budget.tokens if not fact_pool else int(budget.tokens * budget.chunk_share)
    first, chunk_tokens = _fill(chunk_pool, share, count)
    taken_facts, fact_tokens = _fill(fact_pool, budget.tokens - chunk_tokens, count)
    rest = [item for item in chunk_pool if item not in first]
    more, more_tokens = _fill(rest, budget.tokens - chunk_tokens - fact_tokens, count)
    taken_chunks = [item for item in chunk_pool if item in first or item in more]

    used = {tag for item in (*taken_chunks, *taken_facts) for tag in item.tags}
    return AssembledContext(
        chunks=taken_chunks,
        facts=taken_facts,
        citations={tag: doc for tag, doc in registry.citations.items() if tag in used},
        tokens={"chunks": chunk_tokens + more_tokens, "facts": fact_tokens},
        dropped={
            "chunks": len(all_chunks) - len(taken_chunks),
            "facts": len(all_facts) - len(taken_facts),
        },
    )


def assemble_context(
    result: RetrievalResult,
    *,
    budget: ContextBudget | None = None,
    count: TokenCounter | None = None,
) -> AssembledContext:
    """The context for a retrieval result."""
    return assemble(result.chunks, result.graph_facts, budget=budget, count=count)
