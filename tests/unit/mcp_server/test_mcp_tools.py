"""The MCP server through an in-memory MCP client: the tools it lists, search_documents
(filters, vendor names, citations, errors) and graph_query (templates with defaults,
Text2Cypher with one retry), with stand-ins for the stores."""

import threading
from collections.abc import AsyncIterator, Mapping
from datetime import date
from typing import Any

import pytest
from mcp import ClientSession
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

from ledgerlens.core.schemas import DocType
from ledgerlens.llm.types import LLMProviderError
from ledgerlens.mcp_server import server as server_module
from ledgerlens.mcp_server.server import INSTRUCTIONS, build_parser, create_server
from ledgerlens.mcp_server.services import Services
from ledgerlens.mcp_server.tools.graph import row_citations
from ledgerlens.retrieval import GraphFact, QueryFilters, RetrievalResult, ScoredChunk
from ledgerlens.retrieval.analyze import VendorFinder
from ledgerlens.retrieval.text2cypher import CypherRejectedError, CypherRun
from ledgerlens.storage.neo4j.linker import VendorCandidate
from ledgerlens.storage.neo4j.queries import CypherTemplate

TODAY = date(2026, 11, 20)


# --------------------------------------------------------------------------- stand-ins


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeRetriever:
    name = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[str, QueryFilters | None, int]] = []

    def retrieve(
        self, query: str, *, filters: QueryFilters | None = None, k: int = 5
    ) -> RetrievalResult:
        self.calls.append((query, filters, k))
        chunk = ScoredChunk(
            id="d1#c0",
            document_id="d1",
            text="INVOICE INV-889 Acme Supplies Ltd total 4,200.00",
            score=0.91234,
            doc_type="invoice",
            doc_number="INV-889",
            vendor_id="V001",
            page=1,
        )
        fact = GraphFact(
            text="Invoice INV-889 (Acme Supplies Ltd, 4,200.00 GBP) -> PO-1042 -> no payment",
            citations={"doc:PO-1042": "d2"},
        )
        return RetrievalResult.build(query, self.name, [chunk], [fact])


class FakeGraph:
    """Answers any template with one invoice row."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def run(self, template: CypherTemplate, params: Mapping[str, Any]) -> list[dict[str, Any]]:
        self.calls.append((template.name, dict(params)))
        return [
            {
                "vendor_id": "V001",
                "vendor": "Acme Supplies Ltd",
                "currency": "GBP",
                "invoices": 2,
                "invoiced": 300.0,
                "paid": 100.0,
                "outstanding": 200.0,
                "open_invoices": 1,
                "overdue": 200.0,
                "overdue_invoices": 1,
                "first_invoice": date(2026, 1, 5),
                "last_invoice": date(2026, 2, 5),
                "contract": {"contract_number": "CT-V001", "document_id": "d9"},
            }
        ]


class FakeGenerator:
    def __init__(self, *queries: str) -> None:
        self.queries = list(queries)
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        question: str,
        analysis: object = None,
        *,
        previous: str | None = None,
        error: str | None = None,
    ) -> str:
        self.calls.append({"question": question, "previous": previous, "error": error})
        return self.queries.pop(0)


class FakeGuard:
    """Rejects queries that write; returns one row for the rest."""

    def run(self, cypher: str) -> CypherRun:
        if "DELETE" in cypher:
            raise CypherRejectedError("write clause DELETE")
        return CypherRun(
            cypher,
            [{"invoice_number": "INV-889", "document_id": "d1", "due_date": date(2026, 4, 13)}],
        )


def vendor_finder() -> VendorFinder:
    return VendorFinder(
        [
            VendorCandidate("V001", "Acme Supplies Ltd"),
            VendorCandidate("V002", "Brightline Packaging Ltd", aliases=("Brightline",)),
        ]
    )


@pytest.fixture
def services() -> Services:
    return Services(
        retriever=FakeRetriever(),
        graph=FakeGraph(),
        cypher_generator=FakeGenerator("MATCH (i:Invoice) RETURN i.invoice_number LIMIT 5"),
        cypher_guard=FakeGuard(),
        vendors=vendor_finder(),
        today=TODAY,
    )


@pytest.fixture
async def client(services: Services) -> AsyncIterator[ClientSession]:
    async with create_connected_server_and_client_session(create_server(services)) as session:
        yield session


# --------------------------------------------------------------------------- server


@pytest.mark.anyio
async def test_lists_search_and_graph_tools(client: ClientSession) -> None:
    tools = {tool.name: tool for tool in (await client.list_tools()).tools}
    assert set(tools) >= {"search_documents", "graph_query"}
    search = tools["search_documents"].inputSchema
    assert search["required"] == ["query"]
    assert set(search["properties"]) == {"query", "doc_type", "vendor", "date_from", "date_to", "k"}
    assert search["properties"]["k"]["maximum"] == 20
    graph = tools["graph_query"]
    assert graph.description is not None and "unpaid_with_po(" in graph.description
    for tool in (tools["search_documents"], tools["graph_query"]):
        assert tool.annotations is not None and tool.annotations.readOnlyHint
        assert tool.outputSchema is not None  # structured output


@pytest.mark.anyio
async def test_instructions_reach_the_client(services: Services) -> None:
    from mcp.shared.memory import create_connected_server_and_client_session

    async with create_connected_server_and_client_session(create_server(services)) as session:
        result = await session.initialize()
    assert result.serverInfo.name == "ledgerlens"
    assert result.instructions == INSTRUCTIONS


def test_parser_defaults_to_stdio_with_warm_up() -> None:
    args = build_parser().parse_args([])
    assert (args.transport, args.warm_up) == ("stdio", True)
    args = build_parser().parse_args(["--transport", "http", "--port", "9000", "--no-warm-up"])
    assert (args.transport, args.port, args.warm_up) == ("http", 9000, False)


def test_run_picks_the_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[str] = []
    monkeypatch.setattr(FastMCP, "run", lambda self, transport: ran.append(transport))
    server_module.run("http", warm_up=False)
    server_module.run("stdio", warm_up=False)
    assert ran == ["streamable-http", "stdio"]


def test_first_search_waits_for_the_warm_up() -> None:
    release = threading.Event()
    order: list[str] = []

    class SlowRetriever:
        name = "slow"

        def retrieve(
            self, query: str, *, filters: QueryFilters | None = None, k: int = 5
        ) -> RetrievalResult:
            if query == "warm-up":
                release.wait(5)
            order.append(query)
            return RetrievalResult.build(query, self.name, [])

    services = Services(retriever=SlowRetriever())
    warm = services.warm_up()
    searcher = threading.Thread(target=lambda: services.ready_retriever().retrieve("real"))
    searcher.start()
    release.set()
    warm.join(5)
    searcher.join(5)
    assert order == ["warm-up", "real"]


def test_failed_warm_up_does_not_block_searches() -> None:
    class Broken:
        name = "broken"

        def retrieve(
            self, query: str, *, filters: QueryFilters | None = None, k: int = 5
        ) -> RetrievalResult:
            raise ConnectionError("pinecone is down")

    services = Services(retriever=Broken())
    services.warm_up().join(5)
    assert services.ready_retriever() is services.retriever


# --------------------------------------------------------------------------- search_documents


def last_call(services: Services) -> tuple[str, QueryFilters, int]:
    query, filters, k = services.retriever.calls[-1]  # type: ignore[attr-defined]
    assert filters is not None
    return query, filters, k


async def search(client: ClientSession, **arguments: Any) -> dict[str, Any]:
    result = await client.call_tool("search_documents", arguments)
    assert not result.isError, result.content
    assert result.structuredContent is not None
    return result.structuredContent


@pytest.mark.anyio
async def test_returns_hits_facts_and_citations(client: ClientSession) -> None:
    found = await search(client, query="total on INV-889")
    (hit,) = found["hits"]
    assert hit["citation"] == "doc:INV-889"
    assert hit["document_id"] == "d1" and hit["score"] == 0.9123
    assert found["facts"] == [
        "Invoice INV-889 (Acme Supplies Ltd, 4,200.00 GBP) -> PO-1042 -> no payment"
    ]
    assert found["citations"] == {"doc:INV-889": "d1", "doc:PO-1042": "d2"}
    assert found["document_ids"] == ["d1"]


@pytest.mark.anyio
async def test_filters_reach_the_retriever(client: ClientSession, services: Services) -> None:
    found = await search(
        client,
        query="unpaid invoices",
        doc_type="invoice",
        vendor="Acme",
        date_from="2026-01-01",
        date_to="2026-03-31",
        k=3,
    )
    query, filters, k = last_call(services)
    assert (query, k) == ("unpaid invoices", 3)
    assert filters.doc_types == (DocType.INVOICE,)
    assert filters.vendor_ids == ("V001",)  # "Acme" -> the master id
    assert (filters.date_from, filters.date_to) == (date(2026, 1, 1), date(2026, 3, 31))
    assert found["filters"]["vendor_ids"] == ["V001"]


@pytest.mark.parametrize("vendor", ["V002", "Brightline", "Brightline Packaging Ltd"])
@pytest.mark.anyio
async def test_vendor_by_id_alias_or_name(
    client: ClientSession, services: Services, vendor: str
) -> None:
    await search(client, query="anything", vendor=vendor)
    assert last_call(services)[1].vendor_ids == ("V002",)


@pytest.mark.anyio
async def test_no_filters_is_an_empty_filter(client: ClientSession, services: Services) -> None:
    found = await search(client, query="anything")
    assert last_call(services)[1].is_empty()
    assert found["filters"] == {}


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"query": "x", "vendor": "Nobody Trading"}, "no vendor in the vendor master"),
        ({"query": "  "}, "query is empty"),
        ({"query": "x", "date_from": "2026-05-01", "date_to": "2026-01-01"}, "is after"),
        ({"query": "x", "k": 50}, "less than or equal to 20"),
        ({"query": "x", "doc_type": "memo"}, "invoice"),
    ],
)
@pytest.mark.anyio
async def test_bad_arguments_are_tool_errors(
    client: ClientSession, arguments: dict[str, Any], message: str
) -> None:
    result = await client.call_tool("search_documents", arguments)
    assert result.isError
    assert message in result.content[0].text  # type: ignore[union-attr]


# --------------------------------------------------------------------------- graph_query


def queue(services: Services, *queries: str) -> Any:
    """The fake generator, answering ``queries`` in turn."""
    generator: Any = services.cypher_generator
    generator.queries, generator.calls = list(queries), []
    return generator


async def graph(client: ClientSession, **arguments: Any) -> dict[str, Any]:
    result = await client.call_tool("graph_query", arguments)
    assert not result.isError, result.content
    assert result.structuredContent is not None
    return result.structuredContent


async def error(client: ClientSession, **arguments: Any) -> str:
    result = await client.call_tool("graph_query", arguments)
    assert result.isError
    return str(result.content[0].text)  # type: ignore[union-attr]


@pytest.mark.anyio
async def test_template_with_defaults_and_vendor_names(
    client: ClientSession, services: Services
) -> None:
    found = await graph(client, template="vendor_ledger", params={"vendor_ids": ["Acme"]})
    name, params = services.graph.calls[-1]  # type: ignore[attr-defined]
    assert name == "vendor_ledger"
    assert params["vendor_ids"] == ["V001"]
    assert params["as_of"] == "2026-11-20"  # today, unless given
    assert params["limit"] == 20  # the template's default
    assert found["source"] == "template:vendor_ledger" and found["cypher"] is None
    assert found["rows"][0]["first_invoice"] == "2026-01-05"  # dates as ISO strings
    (fact,) = found["facts"]
    assert fact.startswith("Vendor Acme Supplies Ltd (V001), GBP: 2 invoices")
    assert found["citations"] == {"doc:CT-V001": "d9"}


@pytest.mark.anyio
async def test_template_errors(client: ClientSession) -> None:
    assert "choose from invoice_neighborhood" in await error(client, template="everything")
    assert "needs seeds" in await error(client, template="contract_for_po")
    assert "takes no parameter(s) colour" in await error(
        client, template="unpaid_with_po", params={"colour": "red"}
    )


@pytest.mark.anyio
async def test_needs_a_template_or_a_question(client: ClientSession) -> None:
    assert "either a template" in await error(client)
    assert "either a template" in await error(client, template="vendor_ledger", question="why?")
    assert "params go with a template" in await error(client, question="why?", params={"limit": 1})


@pytest.mark.anyio
async def test_question_runs_through_text2cypher(client: ClientSession) -> None:
    found = await graph(client, question="which invoices are due?")
    assert found["source"] == "text2cypher"
    assert found["cypher"] == "MATCH (i:Invoice) RETURN i.invoice_number LIMIT 5"
    assert found["rows"] == [
        {"invoice_number": "INV-889", "document_id": "d1", "due_date": "2026-04-13"}
    ]
    assert found["citations"] == {"doc:INV-889": "d1"}


@pytest.mark.anyio
async def test_rejected_query_gets_one_corrected_retry(
    client: ClientSession, services: Services
) -> None:
    generator = queue(services, "MATCH (i) DELETE i", "MATCH (i:Invoice) RETURN i LIMIT 5")
    found = await graph(client, question="clean up the invoices")
    assert found["cypher"] == "MATCH (i:Invoice) RETURN i LIMIT 5"
    assert generator.calls[1]["previous"] == "MATCH (i) DELETE i"
    assert "write clause DELETE" in generator.calls[1]["error"]

    queue(services, "MATCH (i) DELETE i", "MATCH (i) DETACH DELETE i")  # wrong twice
    assert "was rejected" in await error(client, question="clean up the invoices")


@pytest.mark.anyio
async def test_llm_down_is_a_tool_error(client: ClientSession, services: Services) -> None:
    class Down:
        def generate(self, question: str, *args: Any, **kwargs: Any) -> str:
            raise LLMProviderError("HTTP 401", provider="openai")

    services.__dict__["cypher_generator"] = Down()
    assert "Text2Cypher is unavailable" in await error(client, question="how many invoices?")


def test_row_citations_walk_nested_maps() -> None:
    rows = [
        {
            "invoice_number": "INV-1",
            "document_id": "d1",
            "po": {"po_number": "PO-9", "document_id": "d2"},
            "payments": [{"payment_reference": None, "document_id": "abcdef123456"}],
        },
        {"document_id": None, "invoice_number": "INV-2"},
    ]
    assert row_citations(rows) == {
        "doc:INV-1": "d1",
        "doc:PO-9": "d2",
        "doc:abcdef12": "abcdef123456",
    }
