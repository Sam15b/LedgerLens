"""Upload -> extract -> auto-reconcile, and reconcile on request, with a scripted toolbox."""

import json
from collections.abc import Callable, Mapping
from datetime import date
from decimal import Decimal
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ledgerlens.agents.graph import AgentNodes, build_graph
from ledgerlens.agents.nodes.extractor import ExtractorNode
from ledgerlens.agents.nodes.reconciler import ReconcilerNode, report_text
from ledgerlens.agents.nodes.router import Router
from ledgerlens.agents.tools import ToolBox, ToolOutcome
from ledgerlens.core.schemas import DocType, Invoice, LineItem
from ledgerlens.extraction.base import ExtractionResult
from ledgerlens.llm.types import LLMProviderError, LLMResponse, Message

pytestmark = pytest.mark.anyio

DOC = "5a0aef53-1111-4222-8333-444455556666"
Responder = Callable[[dict[str, Any]], dict[str, Any] | str]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class ScriptedTools(ToolBox):
    """Answers each tool with a function of its arguments (a string is a tool error)."""

    def __init__(self, responses: Mapping[str, Responder]) -> None:
        super().__init__([])
        self.responses = dict(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def names(self) -> list[str]:
        return sorted(self.responses)

    def only(self, *names: str) -> ToolBox:
        return self

    async def call(self, name: str, args: Mapping[str, Any]) -> tuple[ToolOutcome, ToolMessage]:
        self.calls.append((name, dict(args)))
        call_id = f"call_{len(self.calls)}"
        result = self.responses[name](dict(args)) if name in self.responses else "no such tool"
        if isinstance(result, str):
            return self._failed(name, args, call_id, result)
        outcome = ToolOutcome(tool=name, args=dict(args), call_id=call_id, ok=True, data=result)
        return outcome, ToolMessage(json.dumps(result), tool_call_id=call_id, name=name)


INVOICE = Invoice(
    invoice_number="INV-889",
    vendor_name="Acme Supplies Ltd",
    invoice_date=date(2026, 3, 14),
    due_date=date(2026, 4, 13),
    po_number="PO-1042",
    currency="GBP",
    total=Decimal("90.00"),
    line_items=[
        LineItem(
            description="Pallet wrap 500m",
            quantity=Decimal(2),
            unit_price=Decimal("45.00"),
            amount=Decimal("90.00"),
        )
    ],
)
REPORT: dict[str, Any] = {
    "invoice_id": "I-889",
    "invoice_number": "INV-889",
    "citation": "doc:INV-889",
    "po_number": "PO-1042",
    "contract_number": "CT-V001",
    "status": "mismatch",
    "mismatches": ["unit_price"],
    "warnings": [],
    "payment_status": "unpaid",
    "citations": {"doc:INV-889": DOC},
    "checks": [
        {"check": "po_exists", "status": "pass", "message": "PO-1042 is on file"},
        {"check": "vendor_match", "status": "pass", "message": "Acme on both"},
        {"check": "line_qty", "status": "pass", "message": "quantities within PO-1042"},
        {
            "check": "unit_price",
            "status": "fail",
            "message": "1 line(s) off CT-V001 rates by more than 2.0%: Pallet wrap 500m at "
            "45.00 vs 42.41 (+6.1%)",
        },
        {"check": "total", "status": "pass", "message": "lines, tax and total add up"},
        {"check": "payment", "status": "pass", "message": "unpaid, due 2026-04-13"},
        {"check": "terms", "status": "pass", "message": "Net 30 as agreed"},
    ],
}


def tools(**overrides: Responder) -> ScriptedTools:
    responses: dict[str, Responder] = {
        "get_document_text": lambda args: {
            "document_id": DOC,
            "filename": "inv-889.pdf",
            "doc_type": "invoice",
            "text": "INVOICE INV-889 ...",
        },
        "save_extraction": lambda args: {
            "document_id": args["document_id"],
            "record_id": "I-889",
            "doc_type": args["record"]["doc_type"],
            "citation": "doc:INV-889",
            "anomaly": {"score": 0.6, "flagged": True, "reasons": ["1 line(s) over the rate"]},
        },
        "reconcile": lambda args: REPORT,
    }
    responses.update(overrides)
    return ScriptedTools(responses)


class FixedExtractor:
    name = "llm"

    def __init__(self, result: Invoice | Exception) -> None:
        self.result = result

    def extract(self, text: str, doc_type: DocType | str) -> ExtractionResult:
        if isinstance(self.result, Exception):
            raise self.result
        return ExtractionResult(extractor=self.name, doc_type=DocType.INVOICE, record=self.result)


class Narrator:
    def __init__(self, text: str | None) -> None:
        self.text = text
        self.prompts: list[str] = []

    def complete(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        self.prompts.append(messages[-1].content)
        if self.text is None:
            raise LLMProviderError("HTTP 401", provider="openai")
        return LLMResponse(text=self.text, provider="openai", model="gpt-4o")


def graph(box: ScriptedTools, *, extractor: Any = None, llm: Any = None) -> Any:
    return build_graph(
        AgentNodes(
            router=Router(None),
            extractor=ExtractorNode(box, extractor or FixedExtractor(INVOICE)),
            reconciler=ReconcilerNode(box, llm),
        )
    )


async def test_upload_is_extracted_saved_and_reconciled() -> None:
    box = tools()
    result = await graph(box).ainvoke(
        {"messages": [HumanMessage("Please process this")], "document_ids": [DOC]}
    )
    assert result["intent"] == "extract"
    assert [name for name, _ in box.calls] == ["get_document_text", "save_extraction", "reconcile"]
    saved_args = box.calls[1][1]
    assert saved_args["document_id"] == DOC and saved_args["record"]["invoice_number"] == "INV-889"
    assert box.calls[2][1] == {"invoice": "I-889"}  # the record just saved
    assert result["extracted"]["record_id"] == "I-889"
    assert result["reconciliation"]["mismatches"] == ["unit_price"]
    replies = [m.text for m in result["messages"] if isinstance(m, AIMessage) and m.content]
    assert replies[0].startswith("Read inv-889.pdf: invoice INV-889 from Acme Supplies Ltd")
    assert "Flagged as unusual (score 0.60)" in replies[0]
    assert "Mismatches: unit_price" in replies[1]
    assert result["citations"] == {"doc:INV-889": DOC}
    assert result["step_count"] == 1 + 3 + 2 + 1  # router, extractor+2, reconciler+1, respond


async def test_a_saved_contract_is_not_reconciled() -> None:
    box = tools(
        get_document_text=lambda args: {
            "document_id": DOC,
            "filename": "ct.pdf",
            "doc_type": "invoice",
            "text": "...",
        },
        save_extraction=lambda args: {
            "document_id": DOC,
            "record_id": "CT-1",
            "doc_type": "contract",
            "citation": "doc:CT-1",
        },
    )
    result = await graph(box).ainvoke(
        {"messages": [HumanMessage("Ingest this document")], "document_ids": [DOC]}
    )
    assert [name for name, _ in box.calls] == ["get_document_text", "save_extraction"]
    assert result.get("reconciliation") is None


async def test_extractor_falls_back_to_rules_when_the_model_fails() -> None:
    node = ExtractorNode(tools(), FixedExtractor(LLMProviderError("HTTP 401", provider="openai")))
    node.fallback = FixedExtractor(INVOICE)
    update = await node({"document_ids": [DOC]})
    assert update["extracted"]["record_id"] == "I-889"


async def test_unreadable_document_is_reported_not_saved() -> None:
    box = tools(get_document_text=lambda args: "no stored document 'x'")
    update = await ExtractorNode(box, FixedExtractor(INVOICE))({"document_ids": ["x"]})
    assert [name for name, _ in box.calls] == ["get_document_text"]
    assert update["extracted"] is None
    assert update["errors"] == ["could not read x: no stored document 'x'"]


async def test_does_inv_889_match_its_po_and_contract_rate() -> None:
    box = tools()
    llm = Narrator("INV-889 is billed above the contract rate [doc:INV-889]; hold it.")
    result = await graph(box, llm=llm).ainvoke(
        {"messages": [HumanMessage("Does INV-889 match its PO and contract rate?")]}
    )
    assert result["intent"] == "reconcile"
    assert box.calls == [("reconcile", {"invoice": "INV-889"})]
    assert result["reconciliation"]["mismatches"] == ["unit_price"]
    reply = result["messages"][-1].content
    assert reply.startswith("INV-889 is billed above the contract rate")  # the narration
    assert "unit_price (mismatch): 1 line(s) off CT-V001 rates" in reply  # then the report
    assert '"mismatches": [\n   "unit_price"\n  ]' in llm.prompts[0]


async def test_report_stands_when_no_model_narrates() -> None:
    box = tools()
    result = await graph(box, llm=Narrator(None)).ainvoke(
        {"messages": [HumanMessage("Reconcile INV-889")]}
    )
    assert result["messages"][-1].content == report_text(REPORT)


async def test_reconciler_asks_which_invoice() -> None:
    update = await ReconcilerNode(tools())({"messages": [HumanMessage("Reconcile it")]})
    assert "Which invoice should I match?" in update["messages"][0].content


def test_report_text_for_a_match() -> None:
    matched = {**REPORT, "status": "match", "mismatches": [], "checks": REPORT["checks"][:3]}
    assert report_text(matched) == (
        "INV-889 [doc:INV-889] matches against PO PO-1042 and contract CT-V001: every check passed."
    )
