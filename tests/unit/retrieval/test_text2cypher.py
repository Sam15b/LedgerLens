"""Text2Cypher without Neo4j or an LLM: guardrail rejections, generation, R5 retry/fallback."""

from datetime import date
from typing import Any

import pytest
from pydantic import BaseModel

from ledgerlens.llm.types import Message
from ledgerlens.retrieval import QueryFilters, RetrievalResult, ScoredChunk
from ledgerlens.retrieval.analyze import QueryAnalyzer
from ledgerlens.retrieval.graph import GraphExpander
from ledgerlens.retrieval.graphrag import GraphRAGRetriever, Text2CypherRetriever, cypher_facts
from ledgerlens.retrieval.text2cypher import (
    CypherGenerator,
    CypherRejectedError,
    CypherRun,
    check_cypher,
    code_only,
    hints,
    strip_fences,
)
from ledgerlens.storage.neo4j.linker import VendorCandidate

TODAY = date(2026, 11, 17)
ANALYZER = QueryAnalyzer([VendorCandidate("V001", "Acme Supplies Ltd")], today=TODAY)

# --------------------------------------------------------------------------- rejections


@pytest.mark.parametrize(
    ("cypher", "reason"),
    [
        ("CREATE (n:Invoice {id: 'x'}) RETURN n", "CREATE"),
        ("MERGE (v:Vendor {id: 'V001'}) RETURN v", "MERGE"),
        ("MATCH (i:Invoice) SET i.total = 0 RETURN i", "SET"),
        ("MATCH (i:Invoice) DELETE i", "DELETE"),
        ("MATCH (n) DETACH DELETE n", "DETACH"),
        ("MATCH (v:Vendor) REMOVE v.tax_id RETURN v", "REMOVE"),
        ("DROP CONSTRAINT vendor_id", "DROP"),
        ("MATCH (i:Invoice) FOREACH (x IN [1] | SET i.y = x) RETURN i", "FOREACH"),
        ("LOAD CSV FROM 'file:///x.csv' AS row RETURN row", "LOAD CSV"),
        ("CALL { MATCH (n) RETURN n } IN TRANSACTIONS RETURN 1", "IN TRANSACTIONS"),
        ("match (n) detach delete n", "DETACH"),  # case does not matter
        ("MATCH (n)\nWITH n\nCREATE\n(m)", "CREATE"),
        ("CALL dbms.components()", "procedure calls"),
        ("CALL db.labels() YIELD label RETURN label", "procedure calls"),
        ("CALL apoc.periodic.iterate('MATCH (n) RETURN n', 'DELETE n', {})", "procedure calls"),
        ("RETURN apoc.cypher.runFirstColumnSingle('MATCH (n) DELETE n', {})", "apoc"),
        ("MATCH (n) RETURN n; MATCH (m) DETACH DELETE m", "one statement"),
        ("USE system SHOW USERS", "USE"),
        ("MATCH (n:Invoice) WITH n", "must RETURN"),
        ("   ", "empty"),
    ],
)
def test_write_and_admin_queries_are_rejected(cypher: str, reason: str) -> None:
    with pytest.raises(CypherRejectedError, match=reason):
        check_cypher(cypher)


@pytest.mark.parametrize(
    "cypher",
    [
        "MATCH (v:Vendor {name: 'Set Solutions'}) RETURN v.id AS id LIMIT 5",
        'MATCH (v:Vendor) WHERE v.name = "Create & Merge Ltd" RETURN v LIMIT 5',
        "MATCH (i:Invoice) // delete these later\nRETURN i.id LIMIT 5",
        "MATCH (i:Invoice) /* SET */ RETURN i.id LIMIT 5",
        "MATCH (i:Invoice) RETURN i.created AS `delete` LIMIT 5",
        "MATCH (i) CALL (i) { MATCH (i)-[:HAS_LINE]->(l) RETURN count(l) AS n } RETURN n LIMIT 5",
        "MATCH (i:Invoice) WHERE i.offset > 0 RETURN i.reset_at, i.created_by LIMIT 5",
    ],
)
def test_keywords_inside_strings_comments_names_and_words_are_fine(cypher: str) -> None:
    assert check_cypher(cypher) == cypher


def test_a_write_hidden_after_a_string_is_still_found() -> None:
    with pytest.raises(CypherRejectedError, match="DELETE"):
        check_cypher("MATCH (n {url: 'http://x'}) DELETE n")
    blanked = code_only("MATCH (n {url: 'http://x'}) // c\nRETURN n")
    assert blanked == "MATCH (n {url: ''})  \nRETURN n"


@pytest.mark.parametrize(
    ("cypher", "expected"),
    [
        ("MATCH (n) RETURN n", "MATCH (n) RETURN n\nLIMIT 50"),
        ("MATCH (n) RETURN n LIMIT 10", "MATCH (n) RETURN n LIMIT 10"),
        ("MATCH (n) RETURN n LIMIT 5000", "MATCH (n) RETURN n LIMIT 50"),
        ("MATCH (n) RETURN n limit $big", "MATCH (n) RETURN n LIMIT 50"),
        ("```cypher\nMATCH (n) RETURN n LIMIT 3;\n```", "MATCH (n) RETURN n LIMIT 3"),
    ],
)
def test_limit_is_enforced(cypher: str, expected: str) -> None:
    assert check_cypher(cypher) == expected


def test_strip_fences() -> None:
    assert strip_fences("```\nRETURN 1;\n```") == "RETURN 1"
    assert strip_fences("RETURN 1 ;  ") == "RETURN 1"


# --------------------------------------------------------------------------- generation


class StubLLM:
    """``structured`` replies with queued Cypher and records the messages."""

    def __init__(self, *replies: str | Exception) -> None:
        self.replies = list(replies)
        self.calls: list[list[Message]] = []

    def structured(self, messages: list[Message], schema: type[BaseModel], **_: Any) -> Any:
        self.calls.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return schema.model_validate({"cypher": reply})


def generator(llm: StubLLM) -> CypherGenerator:
    return CypherGenerator(llm, today=TODAY)  # type: ignore[arg-type]


def test_hints_carry_what_the_analyzer_resolved() -> None:
    analysis = ANALYZER.analyze("Total unpaid for Acme on INV-889 in Q1 2026")

    assert hints(analysis) == (
        "Acme Supplies Ltd = vendor id V001; document numbers: INV-889; "
        "period: 2026-01-01 to 2026-03-31"
    )
    assert hints(None) == "none"
    assert hints(ANALYZER.analyze("hello")) == "none"


def test_prompt_has_schema_hints_and_question() -> None:
    llm = StubLLM("```cypher\nMATCH (n) RETURN n\n```")

    cypher = generator(llm).generate("How much do we owe Acme?", ANALYZER.analyze("Acme owes"))

    assert cypher == "MATCH (n) RETURN n"
    (prompt,) = llm.calls[0]
    assert "(Payment)-[:SETTLES {invoice_number: string, amount: float}]->(Invoice)" in (
        prompt.content
    )
    assert "Today is 2026-11-17." in prompt.content
    assert "HINTS: Acme Supplies Ltd = vendor id V001" in prompt.content
    assert prompt.content.rstrip().endswith("Question: How much do we owe Acme?")


def test_retry_shows_the_failed_query_and_its_error() -> None:
    llm = StubLLM("RETURN 2")

    generator(llm).generate("q", previous="MATCH (n) RETRUN n", error="SyntaxError: RETRUN")

    messages = llm.calls[0]
    assert [m.role for m in messages] == ["user", "assistant", "user"]
    assert "MATCH (n) RETRUN n" in messages[1].content
    assert "That query failed: SyntaxError: RETRUN" in messages[2].content


# --------------------------------------------------------------------------- R5


class StubGuard:
    """Fails the queued number of times, then returns rows."""

    def __init__(self, rows: list[dict[str, Any]], failures: int = 0) -> None:
        self.rows = rows
        self.failures = failures
        self.ran: list[str] = []

    def run(self, cypher: str) -> CypherRun:
        self.ran.append(cypher)
        if self.failures:
            self.failures -= 1
            raise CypherRejectedError("SET is not allowed: the graph is read-only")
        return CypherRun(cypher, self.rows)


class StubBase:
    name = "r3_stub"

    def retrieve(
        self, query: str, *, filters: QueryFilters | None = None, k: int = 5
    ) -> RetrievalResult:
        chunk = ScoredChunk(id="d1#c0", document_id="d1", text="t", score=1.0, doc_number="INV-1")
        return RetrievalResult.build(query, self.name, [chunk], timings_ms={"embed": 1.0})


class StubRunner:
    def run(self, template: Any, params: Any) -> list[dict[str, Any]]:
        return []


ROWS = [
    {"vendor": "Acme Supplies Ltd", "currency": "GBP", "total_unpaid": 1500.12},
    {"invoice_number": "INV-7", "document_id": "d7", "due_date": date(2026, 1, 2)},
]


def r5(llm: StubLLM, guard: StubGuard) -> Text2CypherRetriever:
    graphrag = GraphRAGRetriever(
        StubBase(), GraphExpander(runner=StubRunner(), today=TODAY), analyzer=ANALYZER
    )
    return Text2CypherRetriever(graphrag, generator(llm), guard)  # type: ignore[arg-type]


def test_r5_puts_cypher_rows_first_for_aggregations() -> None:
    guard = StubGuard(ROWS)
    retriever = r5(StubLLM("MATCH (n) RETURN n LIMIT 5"), guard)

    result = retriever.retrieve("Total unpaid by vendor for Q1?")

    assert result.retriever == "r5_text2cypher"
    texts = [fact.text for fact in result.graph_facts]
    assert texts[0] == "Graph query (2 rows): MATCH (n) RETURN n LIMIT 5"
    assert texts[1] == "Row 1: vendor: Acme Supplies Ltd, currency: GBP, total_unpaid: 1,500.12"
    assert texts[2] == "Row 2: invoice_number: INV-7, due_date: 2026-01-02"
    assert result.citations == {"doc:INV-1": "d1", "doc:INV-7": "d7"}
    assert "text2cypher" in result.timings_ms and "text2cypher_error" not in result.timings_ms


def test_r5_retries_once_with_the_error_then_succeeds() -> None:
    llm = StubLLM("MATCH (i) SET i.x = 1 RETURN i", "MATCH (i) RETURN i LIMIT 5")
    guard = StubGuard(ROWS, failures=1)

    result = r5(llm, guard).retrieve("How many invoices did Acme send?")

    assert guard.ran == ["MATCH (i) SET i.x = 1 RETURN i", "MATCH (i) RETURN i LIMIT 5"]
    assert "SET is not allowed" in llm.calls[1][-1].content
    assert result.graph_facts[0].source == "text2cypher"


def test_r5_falls_back_after_the_retry_fails() -> None:
    llm = StubLLM("MATCH (i) SET i.x = 1 RETURN i", "MATCH (i) SET i.x = 2 RETURN i")
    guard = StubGuard(ROWS, failures=2)

    result = r5(llm, guard).retrieve("How many invoices did Acme send?")

    assert len(guard.ran) == 2
    assert [c.document_id for c in result.chunks] == ["d1"]  # R3's chunks are still there
    assert all(fact.source != "text2cypher" for fact in result.graph_facts)
    assert result.timings_ms["text2cypher_error"] == 1.0


def test_r5_falls_back_when_the_llm_is_down() -> None:
    result = r5(StubLLM(ConnectionError("no provider")), StubGuard(ROWS)).retrieve(
        "Total spend per vendor"
    )

    assert "text2cypher_error" in result.timings_ms
    assert [c.document_id for c in result.chunks] == ["d1"]


def test_r5_skips_the_llm_for_plain_lookups() -> None:
    llm = StubLLM()  # any call would fail: no reply queued

    result = r5(llm, StubGuard(ROWS)).retrieve("What is the total on INV-889?")

    assert llm.calls == []
    assert "text2cypher" not in result.timings_ms


def test_empty_result_still_says_which_query_ran() -> None:
    facts = cypher_facts(CypherRun("MATCH (i) RETURN i LIMIT 5", []))

    assert [fact.text for fact in facts] == ["Graph query (0 rows): MATCH (i) RETURN i LIMIT 5"]
