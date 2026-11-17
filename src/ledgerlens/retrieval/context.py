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
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

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
    return ContextItem("chunk", text, (tag,), (chunk.document_id,), chunk.id)


def fact_item(fact: GraphFact, registry: CitationRegistry) -> ContextItem:
    """A graph fact as context, tagged with every document it names."""
    tags = tuple(
        dict.fromkeys(registry.tag(tag, document_id) for tag, document_id in fact.citations.items())
    )
    documents = tuple(dict.fromkeys([*fact.citations.values(), *fact.document_ids]))
    return ContextItem("fact", _flat(fact.text), tags, documents, fact.source)


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


def assemble(chunks: Iterable[ScoredChunk], facts: Iterable[GraphFact]) -> AssembledContext:
    """Tag every chunk and fact (chunks first, so excerpts claim the plain tags)."""
    registry = CitationRegistry()
    chunk_items = [chunk_item(chunk, registry) for chunk in chunks]
    fact_items = [fact_item(fact, registry) for fact in facts]
    return AssembledContext(chunk_items, fact_items, registry.citations)


def assemble_context(result: RetrievalResult) -> AssembledContext:
    """The context for a retrieval result."""
    return assemble(result.chunks, result.graph_facts)
