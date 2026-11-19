"""The router node: intent accuracy on the labelled set, when the model is asked, what it
finds in a request, and routing through the agent graph."""

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from ledgerlens.agents.graph import OUT_OF_SCOPE, AgentNodes, build_graph
from ledgerlens.agents.nodes.router import Router, RouterDecision, rule_intent
from ledgerlens.agents.state import Intent
from ledgerlens.llm.types import LLMProviderError, Message
from ledgerlens.retrieval.analyze import VendorFinder
from ledgerlens.storage.neo4j.linker import VendorCandidate

UTTERANCES = Path(__file__).parents[3] / "eval" / "router_utterances.jsonl"


def labelled() -> list[dict[str, str]]:
    with UTTERANCES.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_labelled_set_covers_every_intent() -> None:
    rows = labelled()
    counts = {i: sum(r["intent"] == i for r in rows) for i in {r["intent"] for r in rows}}
    assert len(rows) >= 100
    assert set(counts) == {"question", "extract", "reconcile", "remind", "other"}
    assert min(counts.values()) >= 15


def test_rules_alone_reach_95_percent() -> None:
    """Without a model, the rules decide or guess every utterance."""
    router = Router(None)
    rows = labelled()
    wrong = [r for r in rows if router.decide(r["text"])[0].intent != r["intent"]]
    accuracy = 1 - len(wrong) / len(rows)
    assert accuracy >= 0.95, [r["text"] for r in wrong]


class ScriptedLLM:
    """Answers every structured call with one decision, and records the prompts."""

    def __init__(self, decision: RouterDecision | None = None) -> None:
        self.decision = decision
        self.prompts: list[str] = []

    def structured(self, messages: list[Message], schema: type, **kwargs: Any) -> Any:
        self.prompts.append(messages[-1].content)
        if self.decision is None:
            raise LLMProviderError("HTTP 401", provider="openai")
        return self.decision


def test_clear_requests_never_call_the_model() -> None:
    llm = ScriptedLLM(RouterDecision(intent="other"))
    router = Router(llm)  # type: ignore[arg-type]
    cases: list[tuple[str, Intent]] = [
        ("Reconcile INV-889", "reconcile"),
        ("Chase Acme for INV-889", "remind"),
        ("Which invoices are unpaid?", "question"),
        ("thanks!", "other"),
    ]
    for text, intent in cases:
        assert router.decide(text) == (RouterDecision(intent=intent, reason="rules"), "rules")
    assert llm.prompts == []


def test_ambiguous_request_goes_to_the_model() -> None:
    text = "Reconcile INV-889 and chase Acme if it matches"  # two actions
    assert rule_intent(text) is None
    llm = ScriptedLLM(RouterDecision(intent="reconcile", invoice_numbers=["INV-889"]))
    decision, method = Router(llm).decide(text)  # type: ignore[arg-type]
    assert (decision.intent, method) == ("reconcile", "llm")
    assert "Request:\n" + text in llm.prompts[0]


def test_model_down_falls_back_to_the_rules_guess() -> None:
    router = Router(ScriptedLLM(None))  # type: ignore[arg-type]
    decision, method = router.decide("Reconcile INV-889 and chase Acme if it matches")
    assert (decision.intent, method) == ("remind", "fallback")  # remind is first by priority
    assert router.decide("Acme INV-889")[0].intent == "question"  # names the ledger
    assert router.decide("purple monkey dishwasher")[0].intent == "other"


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("Was a reminder already sent for INV-889?", "question"),  # status of an action
        ("What's the contract rate for coffee beans?", "question"),
        ("Was INV-889 billed at the contract rate?", "reconcile"),
        ("Does INV-889 match its PO?", "reconcile"),  # an action beats the question mark
        ("What's the weather in London?", "other"),
    ],
)
def test_rule_precedence(text: str, intent: str) -> None:
    assert rule_intent(text) == intent


def test_node_finds_entities_and_attachments() -> None:
    vendors = VendorFinder([VendorCandidate("V001", "Acme Supplies Ltd")])
    router = Router(None, vendors=vendors)
    update = router(
        {
            "messages": [
                HumanMessage("Hi"),
                AIMessage("Hello"),
                HumanMessage("Does INV-000889 from Acme match PO-2026-01042?"),
            ]
        }
    )
    assert update["intent"] == "reconcile" and update["step_count"] == 1
    entities = update["entities"]
    assert entities["invoice_numbers"] == ["INV-000889"]
    assert entities["po_numbers"] == ["PO-2026-01042"]
    assert (entities["vendor_ids"], entities["vendor"]) == (["V001"], "Acme Supplies Ltd")
    assert entities["routed_by"] == "rules"
    uploaded = router({"messages": [HumanMessage("Here you go")], "document_ids": ["d-1"]})
    assert (uploaded["intent"], uploaded["document_ids"]) == ("extract", ["d-1"])


@pytest.mark.parametrize(
    ("text", "last_node"),
    [
        ("Which Acme invoices are unpaid?", "qa"),
        ("Reconcile INV-889", "reconciler"),
        ("Draft a reminder for INV-889", "reminder_writer"),
        ("Extract the fields from the attached invoice", "extractor"),
    ],
)
def test_graph_hands_off_to_the_specialist(text: str, last_node: str) -> None:
    graph = build_graph(AgentNodes(router=Router(None)))
    result = graph.invoke({"messages": [HumanMessage(text)]})
    reply = result["messages"][-1]
    assert reply.name == last_node and "not available yet" in reply.content
    assert result["step_count"] == 3  # router, specialist, respond


def test_graph_answers_out_of_scope_itself() -> None:
    graph = build_graph(AgentNodes(router=Router(None)))
    result = graph.invoke({"messages": [HumanMessage("Tell me a joke")]})
    assert result["intent"] == "other"
    assert result["messages"][-1].content == OUT_OF_SCOPE
    assert result["step_count"] == 2
