"""QA trajectories: router -> QA -> MCP tools -> a cited answer, through the real MCP server
(in memory, over stand-in stores) and langchain-mcp-adapters, with scripted models."""

from collections.abc import AsyncIterator, Mapping
from datetime import date
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.shared.memory import create_connected_server_and_client_session

from ledgerlens.agents.graph import AgentNodes, build_graph
from ledgerlens.agents.nodes.qa import PlannedCall, QANode, QAPlan, enforce_citations, min_amount
from ledgerlens.agents.nodes.router import Router
from ledgerlens.agents.tools import ToolBox
from ledgerlens.llm.types import LLMProviderError, LLMResponse, Message
from ledgerlens.mcp_server.server import create_server
from ledgerlens.mcp_server.services import Services
from ledgerlens.retrieval import QueryFilters, RetrievalResult, ScoredChunk
from ledgerlens.retrieval.analyze import VendorFinder
from ledgerlens.storage.neo4j.linker import VendorCandidate
from ledgerlens.storage.neo4j.queries import CypherTemplate

pytestmark = pytest.mark.anyio
QUESTION = "Show unpaid invoices from Acme over $5k"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def unpaid_row(number: str, document_id: str | None, total: float, paid: float) -> dict[str, Any]:
    return {
        "invoice_id": f"I-{number}",
        "invoice_number": number,
        "document_id": document_id,
        "doc_type": "invoice",
        "invoice_date": date(2026, 2, 1),
        "due_date": date(2026, 3, 3),
        "total": total,
        "currency": "GBP",
        "vendor_id": "V001",
        "vendor": "Acme Supplies Ltd",
        "po": {"id": "PO-1042", "po_number": "PO-1042", "document_id": None, "total": total},
        "contract": None,
        "payments": [],
        "paid": paid,
        "outstanding": total - paid,
        "hops": 0,
    }


class Retriever:
    name = "fake"

    def __init__(self, empty: bool = False) -> None:
        self.empty = empty
        self.calls: list[QueryFilters | None] = []

    def retrieve(
        self, query: str, *, filters: QueryFilters | None = None, k: int = 5
    ) -> RetrievalResult:
        self.calls.append(filters)
        chunks = (
            []
            if self.empty
            else [
                ScoredChunk(
                    id="d1#c0",
                    document_id="d1",
                    text="[INVOICE INV-889 | Vendor: Acme]\nINVOICE INV-889 Total Due GBP 7,200.00",
                    score=0.9,
                    doc_type="invoice",
                    doc_number="INV-889",
                )
            ]
        )
        return RetrievalResult.build(query, self.name, chunks)


class Graph:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def run(self, template: CypherTemplate, params: Mapping[str, Any]) -> list[dict[str, Any]]:
        self.calls.append((template.name, dict(params)))
        return self.rows if template.name == "unpaid_with_po" else []


def services(*, empty: bool = False) -> Services:
    rows = (
        []
        if empty
        else [unpaid_row("INV-889", "d1", 7200.0, 0.0), unpaid_row("INV-890", None, 300.0, 0.0)]
    )
    return Services(
        retriever=Retriever(empty),
        graph=Graph(rows),
        vendors=VendorFinder([VendorCandidate("V001", "Acme Supplies Ltd")]),
        today=date(2026, 3, 20),
    )


class ScriptedLLM:
    """Plans with ``plan`` and answers with ``answer`` (or fails when they are None)."""

    def __init__(self, plan: list[PlannedCall] | None, answer: str | None) -> None:
        self.plan, self.answer = plan, answer
        self.prompts: list[str] = []

    def structured(self, messages: list[Message], schema: type, **kwargs: Any) -> Any:
        if self.plan is None:
            raise LLMProviderError("HTTP 401", provider="openai")
        return QAPlan(calls=self.plan)

    def complete(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        self.prompts.append(messages[-1].content)
        if self.answer is None:
            raise LLMProviderError("HTTP 401", provider="openai")
        return LLMResponse(text=self.answer, provider="openai", model="gpt-4o")


@pytest.fixture
async def toolbox_for() -> AsyncIterator[Any]:
    """``await toolbox_for(services)``: the MCP server's tools over an in-memory session."""
    stack: list[Any] = []

    async def make(svc: Services) -> ToolBox:
        session_cm = create_connected_server_and_client_session(create_server(svc))
        session = await session_cm.__aenter__()
        stack.append(session_cm)
        return ToolBox(await load_mcp_tools(session))

    yield make
    for session_cm in reversed(stack):
        await session_cm.__aexit__(None, None, None)


async def ask(qa: QANode, svc: Services, text: str = QUESTION, **graph_args: Any) -> dict[str, Any]:
    graph = build_graph(AgentNodes(router=Router(None, vendors=svc.vendors), qa=qa), **graph_args)
    result: dict[str, Any] = await graph.ainvoke({"messages": [HumanMessage(text)]})
    return result


def tool_calls(result: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (call["name"], call["args"])
        for message in result["messages"]
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]


async def test_unpaid_over_5k_without_a_model(toolbox_for: Any) -> None:
    svc = services()
    result = await ask(QANode(await toolbox_for(svc)), svc)
    assert result["intent"] == "question"
    assert tool_calls(result) == [
        ("search_documents", {"query": QUESTION, "k": 5, "vendor": "V001"}),
        (
            "graph_query",
            {"template": "unpaid_with_po", "params": {"vendor_ids": ["V001"], "limit": 50}},
        ),
    ]
    answer = result["messages"][-1]
    assert "INV-889" in answer.content and "[doc:INV-889]" in answer.content
    assert "INV-890" not in answer.content  # 300.00 outstanding: under $5k
    assert answer.additional_kwargs["citation_check"] == "cited"
    assert result["citations"] == {"doc:INV-889": "d1"}
    assert result["step_count"] == 5  # router, qa + 2 tool calls, respond
    results = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert [m.status for m in results] == ["success", "success"]


async def test_model_plans_and_answers_with_citations(toolbox_for: Any) -> None:
    svc = services()
    plan = [
        PlannedCall(
            tool="graph_query",
            args={"template": "unpaid_with_po", "params": {"vendor_ids": ["Acme"]}},
        )
    ]
    llm = ScriptedLLM(plan, "INV-889 has 7,200.00 GBP outstanding [doc:INV-889] [doc:INV-999].")
    result = await ask(QANode(await toolbox_for(svc), llm), svc)  # type: ignore[arg-type]
    assert [name for name, _ in tool_calls(result)] == ["graph_query"]
    answer = result["messages"][-1]
    assert (
        answer.content == "INV-889 has 7,200.00 GBP outstanding [doc:INV-889]."
    )  # invented tag gone
    assert "[doc:INV-889] Invoice INV-889 from Acme Supplies Ltd" in llm.prompts[0]


async def test_uncited_model_answer_is_replaced_by_the_evidence(toolbox_for: Any) -> None:
    svc = services()
    llm = ScriptedLLM(None, "Acme has one big unpaid invoice.")  # planner down, answer uncited
    result = await ask(QANode(await toolbox_for(svc), llm), svc)  # type: ignore[arg-type]
    answer = result["messages"][-1]
    assert answer.additional_kwargs["citation_check"] == "replaced"
    assert "[doc:INV-889]" in answer.content


async def test_no_evidence_is_said_plainly(toolbox_for: Any) -> None:
    svc = services(empty=True)
    result = await ask(QANode(await toolbox_for(svc)), svc)
    answer = result["messages"][-1]
    assert answer.content == "I found no evidence in the ledger for that."
    assert answer.additional_kwargs["citation_check"] == "no_evidence"


async def test_tool_errors_come_back_as_messages(toolbox_for: Any) -> None:
    svc = services()
    plan = [
        PlannedCall(tool="graph_query", args={"template": "everything"}),
        PlannedCall(tool="search_documents", args={"query": "INV-889"}),
    ]
    result = await ask(QANode(await toolbox_for(svc), ScriptedLLM(plan, None)), svc)  # type: ignore[arg-type]
    failed = [m for m in result["messages"] if isinstance(m, ToolMessage) and m.status == "error"]
    assert len(failed) == 1 and "unknown template 'everything'" in str(failed[0].content)
    assert any(e.startswith("graph_query: ") for e in result["errors"])
    assert "[doc:INV-889]" in result["messages"][-1].content  # the search still answered


async def test_qa_only_gets_its_three_tools(toolbox_for: Any) -> None:
    qa = QANode(await toolbox_for(services()))
    assert qa.tools.names == ["graph_query", "run_analytics", "search_documents"]


async def test_step_limit_trims_tool_calls(toolbox_for: Any) -> None:
    svc = services()
    qa = QANode(await toolbox_for(svc), max_steps=4)  # router 1 + qa 1 + respond 1: one call left
    result = await ask(qa, svc, max_steps=4)
    assert [name for name, _ in tool_calls(result)] == ["search_documents"]
    assert "step limit: 1 planned tool call(s) skipped" in result["errors"]


async def test_a_failing_node_answers_with_an_error() -> None:
    class Broken:
        async def __call__(self, state: Any) -> dict[str, Any]:
            raise RuntimeError("index unavailable")

    graph = build_graph(AgentNodes(router=Router(None), qa=Broken()))
    result = await graph.ainvoke({"messages": [HumanMessage("Which invoices are unpaid?")]})
    assert "the qa step failed (RuntimeError: index unavailable)" in result["messages"][-1].content
    assert result["errors"] == ["qa: RuntimeError: index unavailable"]


async def test_run_stops_at_the_step_limit() -> None:
    def runaway(state: Any) -> dict[str, Any]:
        return {"intent": "question", "step_count": 8}

    graph = build_graph(AgentNodes(router=runaway), max_steps=8)
    result = await graph.ainvoke({"messages": [HumanMessage("Which invoices are unpaid?")]})
    assert result["messages"][-1].content.startswith("I stopped after 8 steps (the limit is 8)")


@pytest.mark.parametrize(
    ("question", "amount"),
    [
        ("over $5k", 5000.0),
        ("more than 1,200.50 GBP", 1200.5),
        ("above 2m", 2_000_000.0),
        ("all", None),
    ],
)
def test_min_amount(question: str, amount: float | None) -> None:
    assert min_amount(question) == amount


def test_enforce_citations_keeps_true_no_evidence_answers() -> None:
    from ledgerlens.agents.nodes.qa import Evidence

    evidence = Evidence()
    evidence.add("row", ["doc:INV-1"], "Invoice INV-1")
    text, check = enforce_citations("The ledger does not contain a payment for INV-1.", evidence)
    assert check == "no_evidence" and text.startswith("The ledger does not contain")
