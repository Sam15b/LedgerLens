"""Reminder drafts: tone by days overdue, every fact stated, a pause for approval, resume
on approve / reject. Nothing is ever sent."""

import json
from collections.abc import Callable, Mapping
from datetime import date, timedelta
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from ledgerlens.agents.graph import AgentNodes, build_graph, read_decisions
from ledgerlens.agents.nodes.reminder_writer import (
    REMINDER_TOOLS,
    ReminderWriterNode,
    min_days_overdue,
)
from ledgerlens.agents.nodes.router import Router
from ledgerlens.agents.tools import ToolBox, ToolOutcome
from ledgerlens.mcp_server.tools.reminders import missing_facts

pytestmark = pytest.mark.anyio
TODAY = date(2026, 5, 20)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def invoice(number: str, days_overdue: int, *, status: str = "unpaid") -> dict[str, Any]:
    return {
        "id": f"I-{number}",
        "document_id": None,
        "citation": None,
        "vendor_id": "V001",
        "vendor_name": "Acme Supplies Ltd",
        "invoice_number": number,
        "due_date": (TODAY - timedelta(days=days_overdue)).isoformat(),
        "po_number": "PO-1042",
        "currency": "GBP",
        "total": "4200.00",
        "outstanding": "1200.00",
        "status": status,
    }


class Ledger(ToolBox):
    """The four reminder tools over a dict of invoices; drafts are kept, never sent."""

    def __init__(self, invoices: list[dict[str, Any]]) -> None:
        super().__init__([])
        self.invoices = {i["invoice_number"]: i for i in invoices}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.drafts: list[dict[str, Any]] = []
        self.handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any] | str]] = {
            "list_overdue": self.list_overdue,
            "get_invoice": self.get_invoice,
            "get_vendor_contact": lambda args: {
                "vendor_id": "V001",
                "name": "Acme Supplies Ltd",
                "email": None,
            },
            "draft_reminder": self.draft_reminder,
        }

    @property
    def names(self) -> list[str]:
        return sorted(self.handlers)

    def only(self, *names: str) -> ToolBox:
        assert set(names) == set(REMINDER_TOOLS)
        return self

    def list_overdue(self, args: dict[str, Any]) -> dict[str, Any]:
        rows = [
            {"invoice_id": i["id"], "invoice_number": n}
            for n, i in self.invoices.items()
            if i["status"] == "unpaid"
            and (TODAY - date.fromisoformat(i["due_date"])).days >= args["min_days"]
        ]
        return {"invoices": rows[: args["limit"]], "count": len(rows)}

    def get_invoice(self, args: dict[str, Any]) -> dict[str, Any] | str:
        key = args["invoice"].removeprefix("I-")
        return self.invoices.get(key) or f"no invoice {args['invoice']!r}"

    def draft_reminder(self, args: dict[str, Any]) -> dict[str, Any]:
        record = self.invoices[args["invoice"].removeprefix("I-")]
        draft = {
            "reminder_id": f"r{len(self.drafts) + 1}",
            "invoice_id": record["id"],
            "invoice_number": record["invoice_number"],
            "tone": args["tone"],
            "days_overdue": (TODAY - date.fromisoformat(record["due_date"])).days,
            "status": "pending_approval",
            "sent": False,
            "vendor_email": None,
            "body": args["body"],
            "missing_facts": missing_facts(
                args["body"],
                invoice_number=record["invoice_number"],
                amounts=("4200.00", "1200.00"),
                due_date=date.fromisoformat(record["due_date"]),
                po_number=record["po_number"],
            ),
        }
        self.drafts.append(draft)
        return draft

    async def call(self, name: str, args: Mapping[str, Any]) -> tuple[ToolOutcome, ToolMessage]:
        self.calls.append((name, dict(args)))
        call_id = f"call_{len(self.calls)}"
        result = self.handlers[name](dict(args))
        if isinstance(result, str):
            return self._failed(name, args, call_id, result)
        outcome = ToolOutcome(tool=name, args=dict(args), call_id=call_id, ok=True, data=result)
        return outcome, ToolMessage(json.dumps(result), tool_call_id=call_id, name=name)


def graph(ledger: Ledger) -> Any:
    nodes = AgentNodes(router=Router(None), reminder_writer=ReminderWriterNode(ledger, today=TODAY))
    return build_graph(nodes, checkpointer=InMemorySaver())


THREAD = {"configurable": {"thread_id": "t-1"}}


async def test_tone_follows_days_overdue_and_every_fact_is_stated() -> None:
    ledger = Ledger([invoice("INV-1", 5), invoice("INV-2", 20), invoice("INV-3", 60)])
    await graph(ledger).ainvoke(
        {"messages": [HumanMessage("Draft reminders for INV-1, INV-2 and INV-3")]}, THREAD
    )
    assert [(d["invoice_number"], d["tone"]) for d in ledger.drafts] == [
        ("INV-1", "friendly"),
        ("INV-2", "firm"),
        ("INV-3", "final_notice"),
    ]
    for draft in ledger.drafts:
        assert draft["missing_facts"] == [], draft["body"]
        assert draft["status"] == "pending_approval" and draft["sent"] is False
    assert "FINAL NOTICE" in ledger.drafts[2]["body"]


async def test_drafts_wait_for_approval() -> None:
    ledger = Ledger([invoice("INV-1", 20)])
    app = graph(ledger)
    result = await app.ainvoke({"messages": [HumanMessage("Chase INV-1")]}, THREAD)
    (pause,) = result["__interrupt__"]
    assert pause.value["type"] == "reminder_approval"
    assert [r["reminder_id"] for r in pause.value["reminders"]] == ["r1"]
    state = await app.aget_state(THREAD)
    assert state.next == ("human_approval",)
    assert "waiting for your approval; nothing has been sent" in result["messages"][-1].content


async def test_resume_on_approve_sends_nothing() -> None:
    ledger = Ledger([invoice("INV-1", 20)])
    app = graph(ledger)
    await app.ainvoke({"messages": [HumanMessage("Chase INV-1")]}, THREAD)
    calls_before = len(ledger.calls)
    result = await app.ainvoke(Command(resume={"r1": "approved"}), THREAD)
    assert result["reminder_decisions"] == {"r1": "approved"}
    reply = result["messages"][-1].content
    assert reply.startswith("Approved: INV-1. Nothing has been sent")
    assert len(ledger.calls) == calls_before  # resuming calls no tool at all
    assert (await app.aget_state(THREAD)).next == ()


async def test_resume_on_reject_and_partial_decisions() -> None:
    ledger = Ledger([invoice("INV-1", 20), invoice("INV-2", 50)])
    app = graph(ledger)
    await app.ainvoke({"messages": [HumanMessage("Remind Acme about INV-1 and INV-2")]}, THREAD)
    result = await app.ainvoke(Command(resume={"r2": "rejected"}), THREAD)
    reply = result["messages"][-1].content
    assert "Rejected: INV-2." in reply and "Still waiting for a decision: INV-1." in reply


async def test_nothing_overdue_means_no_pause() -> None:
    ledger = Ledger([invoice("INV-1", -3), invoice("INV-2", 30, status="paid")])
    app = graph(ledger)
    result = await app.ainvoke(
        {"messages": [HumanMessage("Send a reminder for INV-1 and INV-2")]}, THREAD
    )
    assert "__interrupt__" not in result and ledger.drafts == []
    reply = result["messages"][-1].content
    assert "INV-1 is not overdue" in reply and "INV-2 is paid" in reply


async def test_overdue_list_when_no_invoice_is_named() -> None:
    ledger = Ledger([invoice("INV-1", 20), invoice("INV-2", -1)])
    await graph(ledger).ainvoke(
        {"messages": [HumanMessage("Prepare reminders for everything overdue")]}, THREAD
    )
    assert ledger.calls[0][0] == "list_overdue"
    assert [d["invoice_number"] for d in ledger.drafts] == ["INV-1"]


async def test_the_requested_overdue_threshold_is_kept() -> None:
    ledger = Ledger([invoice("INV-1", 40), invoice("INV-2", 10), invoice("INV-3", 30)])
    await graph(ledger).ainvoke(
        {"messages": [HumanMessage("Draft reminders for invoices 30+ days overdue")]}, THREAD
    )
    assert ledger.calls[0][0] == "list_overdue" and ledger.calls[0][1]["min_days"] == 30
    assert sorted(d["invoice_number"] for d in ledger.drafts) == ["INV-1", "INV-3"]


@pytest.mark.parametrize(
    ("text", "days"),
    [
        ("Draft reminders for invoices 30+ days overdue", 30),
        ("remind everyone 45 days or more overdue", 45),
        ("chase invoices at least 60 days past due", 60),
        ("invoices over 30 days late", 31),
        ("anything more than 15 days overdue?", 16),
        ("reminders for invoices overdue by 20 days", 20),
        ("bills overdue for more than 7 days", 8),
        ("Prepare reminders for everything overdue", 1),
        ("Chase INV-30 for Acme", 1),
        ("pay within 30 days", 1),
        ("invoices 0 days overdue", 1),
    ],
)
def test_min_days_overdue(text: str, days: int) -> None:
    assert min_days_overdue(text) == days


def test_read_decisions() -> None:
    pending = [{"reminder_id": "r1"}, {"reminder_id": "r2"}]
    assert read_decisions("approved", pending) == {"r1": "approved", "r2": "approved"}
    assert read_decisions({"decision": "rejected"}, pending) == {"r1": "rejected", "r2": "rejected"}
    assert read_decisions({"r2": "approved", "r9": "approved", "r1": "maybe"}, pending) == {
        "r2": "approved"
    }
    assert read_decisions(None, pending) == {}
