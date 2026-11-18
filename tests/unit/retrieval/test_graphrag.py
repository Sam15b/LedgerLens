"""R4 GraphRAG and context assembly without stores: fusion, budget, dedupe, citations."""

from collections.abc import Mapping
from datetime import date
from typing import Any

import pytest

from ledgerlens.retrieval import GraphFact, QueryFilters, RetrievalResult, ScoredChunk
from ledgerlens.retrieval.analyze import QueryAnalyzer
from ledgerlens.retrieval.context import (
    ContextBudget,
    answer_messages,
    assemble,
    assemble_context,
    cited_tags,
)
from ledgerlens.retrieval.graph import GraphExpander
from ledgerlens.retrieval.graphrag import GraphRAGRetriever
from ledgerlens.storage.neo4j.linker import VendorCandidate
from ledgerlens.storage.neo4j.queries import CypherTemplate

TODAY = date(2026, 11, 17)


def words(text: str) -> int:
    """A predictable token counter: one token per word."""
    return len(text.split())


def chunk(document_id: str, n: int = 0, number: str | None = None, text: str = "") -> ScoredChunk:
    return ScoredChunk(
        id=f"{document_id}#c{n}",
        document_id=document_id,
        text=text or f"text of {document_id} chunk {n}",
        score=1.0 - n / 10,
        doc_type="invoice",
        doc_number=number,
        page=1,
    )


def fact(text: str, subject: str | None = None, **citations: str) -> GraphFact:
    return GraphFact(
        text=text,
        citations={f"doc:{tag}": doc for tag, doc in citations.items()},
        document_ids=tuple(citations.values()),
        subject=subject,
    )


# --------------------------------------------------------------------------- citation tags


def test_every_line_carries_its_tags_and_the_map_resolves_them() -> None:
    context = assemble(
        [chunk("d-inv", number="INV-889", text="INVOICE INV-889\n  Total   4,200.00")],
        [fact("Invoice INV-889 -> PO-1042 -> no payment", "invoice:1", **{"INV-889": "d-inv"})],
        count=words,
    )

    assert context.text == (
        "Document excerpts:\n"
        "[doc:INV-889] (invoice, p. 1) INVOICE INV-889 Total 4,200.00\n\n"
        "Facts from the document graph:\n"
        "[doc:INV-889] Invoice INV-889 -> PO-1042 -> no payment"
    )
    assert context.citations == {"doc:INV-889": "d-inv"}


def test_two_documents_with_one_number_get_distinct_tags() -> None:
    context = assemble(
        [chunk("aaaaaaaa-1", number="INV-1"), chunk("bbbbbbbb-2", number="INV-1")],
        [fact("Invoice INV-1 (the second)", "invoice:2", **{"INV-1": "bbbbbbbb-2"})],
        count=words,
    )

    assert context.citations == {"doc:INV-1": "aaaaaaaa-1", "doc:INV-1@bbbbbbbb": "bbbbbbbb-2"}
    assert context.facts[0].tags == ("doc:INV-1@bbbbbbbb",)


def test_a_fact_without_documents_is_marked_uncitable() -> None:
    context = assemble([], [fact("Vendor Acme: outstanding 10.00 GBP", "vendor:V001")])

    assert context.facts[0].render() == "[graph] Vendor Acme: outstanding 10.00 GBP"
    assert context.citations == {}


def test_documents_without_a_number_are_cited_by_id_prefix() -> None:
    context = assemble([chunk("0123456789abcdef")], [], count=words)

    assert context.citations == {"doc:01234567": "0123456789abcdef"}


@pytest.mark.parametrize(
    ("answer", "tags"),
    [
        ("Total 4,200.00 GBP [doc:INV-889].", ["doc:INV-889"]),
        ("Linked [doc:INV-889] [doc:PO-1042] and [doc:INV-889].", ["doc:INV-889", "doc:PO-1042"]),
        ("Both [doc:INV-1, doc:INV-1@bbbbbbbb].", ["doc:INV-1", "doc:INV-1@bbbbbbbb"]),
        ("No tags here [graph] [1].", []),
    ],
)
def test_cited_tags(answer: str, tags: list[str]) -> None:
    assert cited_tags(answer) == tags


def test_resolve_separates_real_from_invented_citations() -> None:
    context = assemble([chunk("d-inv", number="INV-889")], [], count=words)

    good = context.resolve("Paid [doc:INV-889].")
    bad = context.resolve("Paid [doc:INV-889] by PAY-1 [doc:PAY-1].")
    none = context.resolve("Paid.")

    assert good.ok and good.document_ids == ["d-inv"]
    assert not bad.ok and bad.unknown == ["doc:PAY-1"]
    assert not none.ok


# --------------------------------------------------------------------------- budget and dedupe


def sized_chunks(n: int) -> list[ScoredChunk]:
    """Chunks that render to 19 words: tag, "(invoice, p. 1)", 15 words (+1 newline = 20)."""
    return [chunk(f"d{i}", text=" ".join([str(i)] + ["w"] * 14)) for i in range(n)]


def sized_facts(n: int) -> list[GraphFact]:
    """Facts that render to 19 words: "[graph]" and 18 words (+1 newline = 20)."""
    return [fact(" ".join([str(i)] + ["f"] * 17), f"invoice:{i}") for i in range(n)]


def test_chunks_get_sixty_percent_when_there_are_facts() -> None:
    context = assemble(
        sized_chunks(10), sized_facts(10), budget=ContextBudget(tokens=100), count=words
    )

    assert len(context.chunks) == 3 and len(context.facts) == 2
    assert context.tokens == {"chunks": 60, "facts": 40}
    assert context.dropped == {"chunks": 7, "facts": 8}


def test_unused_budget_flows_to_the_other_side() -> None:
    budget = ContextBudget(tokens=100)

    without_facts = assemble(sized_chunks(10), [], budget=budget, count=words)
    one_fact = assemble(sized_chunks(10), [fact("short fact", "x")], budget=budget, count=words)
    without_chunks = assemble([], sized_facts(10), budget=budget, count=words)

    assert len(without_facts.chunks) == 5  # the whole budget
    # 60 for chunks, 4 for the fact, then one more chunk in the 36 left
    assert (len(one_fact.chunks), len(one_fact.facts)) == (4, 1)
    assert len(without_chunks.facts) == 5


def test_chunks_dedupe_by_id_text_and_per_document_cap() -> None:
    chunks = [
        chunk("d1", 0),
        chunk("d1", 0),  # same chunk again
        chunk("d1", 1),
        chunk("d1", 2),  # third chunk of d1
        chunk("d2", 0, text="text of d1 chunk 0"),  # same text, other document
        chunk("d2", 1),
    ]

    context = assemble(chunks, [], budget=ContextBudget(per_document=2), count=words)

    assert [item.source for item in context.chunks] == ["d1#c0", "d1#c1", "d2#c1"]
    assert context.dropped["chunks"] == 3


def test_facts_dedupe_by_record_not_by_the_documents_they_cite() -> None:
    contract = {"CT-V001": "d-ct"}
    facts = [
        fact("Invoice INV-1 -> contract CT-V001 -> no payment", "invoice:1", **contract),
        fact("Invoice INV-2 -> contract CT-V001 -> no payment", "invoice:2", **contract),
        fact("Invoice INV-1 (found again by another template)", "invoice:1", **contract),
    ]

    context = assemble([], facts, count=words)

    assert [item.key for item in context.facts] == ["invoice:1", "invoice:2"]


def test_citations_only_cover_what_made_it_into_the_budget() -> None:
    chunks = [chunk("d1", number="INV-1"), chunk("d2", number="INV-2", text=" ".join(["w"] * 50))]

    context = assemble(chunks, [], budget=ContextBudget(tokens=20), count=words)

    assert context.citations == {"doc:INV-1": "d1"}


def test_budget_is_validated() -> None:
    with pytest.raises(ValueError):
        ContextBudget(tokens=0)
    with pytest.raises(ValueError):
        ContextBudget(chunk_share=1.5)


def test_answer_prompt_carries_question_context_and_date() -> None:
    context = assemble([chunk("d-inv", number="INV-889")], [], count=words)

    (message,) = answer_messages("Is INV-889 paid?", context, today=TODAY)

    assert "Today is 2026-11-17." in message.content
    assert "QUESTION: Is INV-889 paid?" in message.content
    assert message.content.rstrip().endswith("[doc:INV-889] (invoice, p. 1) text of d-inv chunk 0")


# --------------------------------------------------------------------------- R4


class StubRetriever:
    name = "r3_stub"

    def __init__(self, chunks: list[ScoredChunk]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, QueryFilters | None, int]] = []

    def retrieve(
        self, query: str, *, filters: QueryFilters | None = None, k: int = 5
    ) -> RetrievalResult:
        self.calls.append((query, filters, k))
        timings = {"embed": 1.0, "rerank": 2.0, "total": 3.0}
        return RetrievalResult.build(query, self.name, self.chunks[:k], timings_ms=timings)


class StubRunner:
    def __init__(self, rows: list[dict[str, Any]], fail: bool = False) -> None:
        self.rows = rows
        self.fail = fail
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def run(self, template: CypherTemplate, params: Mapping[str, Any]) -> list[dict[str, Any]]:
        if self.fail:
            raise ConnectionError("neo4j is down")
        self.calls.append((template.name, dict(params)))
        return self.rows


INVOICE_ROW: dict[str, Any] = {
    "invoice_id": "SINV-1",
    "invoice_number": "INV-889",
    "document_id": "d-inv",
    "total": 4200.0,
    "currency": "GBP",
    "vendor": "Acme Supplies Ltd",
    "po": {"po_number": "PO-1042", "document_id": "d-po"},
    "contract": None,
    "payments": [],
    "paid": 0.0,
}
ANALYZER = QueryAnalyzer([VendorCandidate("V001", "Acme Supplies Ltd")], today=TODAY)


def r4(chunks: list[ScoredChunk], runner: StubRunner) -> tuple[GraphRAGRetriever, StubRetriever]:
    base = StubRetriever(chunks)
    expander = GraphExpander(runner=runner, today=TODAY)
    return GraphRAGRetriever(base, expander, analyzer=ANALYZER), base


def test_r4_fuses_vector_seeds_with_graph_facts() -> None:
    runner = StubRunner([INVOICE_ROW])
    retriever, base = r4([chunk("d-inv", number="INV-889"), chunk("d-x", number="INV-2")], runner)

    result = retriever.retrieve("Which payment settled INV-889?", k=2)

    assert result.retriever == "r4_graphrag"
    assert [c.document_id for c in result.chunks] == ["d-inv", "d-x"]
    assert runner.calls == [
        (
            "invoice_neighborhood",
            {"seeds": ["INV-889", "d-inv", "d-x"], "per_seed": 10, "limit": 30, "max_hops": 2},
        )
    ]
    assert result.graph_facts[0].text.endswith("-> PO-1042 -> no payment")
    assert result.citations == {"doc:INV-889": "d-inv", "doc:INV-2": "d-x", "doc:PO-1042": "d-po"}
    assert {"analyze", "embed", "rerank", "graph", "total"} <= set(result.timings_ms)
    assert result.timings_ms["total"] >= result.timings_ms["graph"]


def test_r4_filters_come_from_the_analysis_unless_given() -> None:
    retriever, base = r4([], StubRunner([]))

    retriever.retrieve("Acme invoices in March 2026")
    given = QueryFilters(vendor_ids=("V009",))
    retriever.retrieve("Acme invoices in March 2026", filters=given)

    analysed = base.calls[0][1]
    assert analysed is not None and analysed.vendor_ids == ("V001",)
    assert analysed.date_from == date(2026, 3, 1)
    assert base.calls[1][1] == given


def test_r4_falls_back_to_the_vector_result_when_the_graph_fails() -> None:
    retriever, _ = r4([chunk("d-inv", number="INV-889")], StubRunner([], fail=True))

    result = retriever.retrieve("Which payment settled INV-889?")

    assert [c.document_id for c in result.chunks] == ["d-inv"]
    assert result.graph_facts == []
    assert "graph_error" in result.timings_ms


def test_answer_citations_resolve_to_real_document_ids() -> None:
    retriever, _ = r4([chunk("d-inv", number="INV-889")], StubRunner([INVOICE_ROW]))
    context = assemble_context(retriever.retrieve("Which payment settled INV-889?"), count=words)
    answer = (
        "No payment settles INV-889 [doc:INV-889]; it references PO-1042 [doc:PO-1042] "
        "and is 4,200.00 GBP [doc:INV-889]."
    )

    check = context.resolve(answer)

    assert check.ok
    assert check.resolved == {"doc:INV-889": "d-inv", "doc:PO-1042": "d-po"}
