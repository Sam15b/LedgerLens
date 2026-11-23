"""Chat -> reminder draft -> approval through the API, against Postgres (throwaway database).

The reminder writer works through the real MCP server (in memory) over the database; the
API's /chat streams the pause, /reminders lists the draft with its run, and approving it
records the decision and resumes that run. Nothing is sent at any point.

    docker compose up -d --wait postgres
    pytest -m integration tests/integration/test_api_reminders.py
"""

import json
from collections.abc import AsyncIterator
from datetime import date
from decimal import Decimal
from typing import Any

import httpx
import pytest
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.memory import InMemorySaver
from mcp.shared.memory import create_connected_server_and_client_session
from sqlalchemy.orm import Session, sessionmaker

from ledgerlens.agents.graph import AgentNodes, build_graph
from ledgerlens.agents.nodes.reminder_writer import ReminderWriterNode
from ledgerlens.agents.nodes.router import Router
from ledgerlens.agents.tools import ToolBox
from ledgerlens.api.agent import AgentProvider, AgentRuntime
from ledgerlens.api.main import create_app
from ledgerlens.api.routers.reminders import get_sessions
from ledgerlens.core.schemas import Invoice
from ledgerlens.mcp_server.server import create_server
from ledgerlens.mcp_server.services import Services
from ledgerlens.retrieval.analyze import VendorFinder
from ledgerlens.storage.neo4j.linker import VendorCandidate
from ledgerlens.storage.postgres.db import session_scope
from ledgerlens.storage.postgres.repositories import Repositories

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

TODAY = date(2026, 5, 20)
VENDOR = "Acme Supplies Ltd"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in text.strip().split("\n\n"):
        name, data = block.split("\n")
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


@pytest.fixture
async def client(sessions: sessionmaker[Session]) -> AsyncIterator[httpx.AsyncClient]:
    with session_scope(sessions) as session:
        repos = Repositories(session)
        repos.vendors.upsert("V001", VENDOR)
        repos.invoices.upsert(
            Invoice(
                invoice_number="INV-1",
                vendor_name=VENDOR,
                invoice_date=date(2026, 3, 1),
                due_date=date(2026, 4, 30),  # 20 days overdue: firm
                po_number="PO-1",
                currency="GBP",
                total=Decimal("1200.00"),
            ),
            id="I-1",
            vendor_id="V001",
        )
    vendors = VendorFinder([VendorCandidate("V001", VENDOR)])
    services = Services(sessions=sessions, vendors=vendors, today=TODAY)
    async with create_connected_server_and_client_session(create_server(services)) as mcp:
        box = ToolBox(await load_mcp_tools(mcp))
        nodes = AgentNodes(
            router=Router(None, vendors=vendors),
            reminder_writer=ReminderWriterNode(box, today=TODAY),
        )
        graph = build_graph(nodes, checkpointer=InMemorySaver())

        async def factory(stack: Any) -> AgentRuntime:
            return AgentRuntime(graph)

        app = create_app(AgentProvider(factory))
        app.dependency_overrides[get_sessions] = lambda: sessions
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://api") as http:
            yield http


async def test_draft_waits_then_approval_resumes_the_run(client: httpx.AsyncClient) -> None:
    chat = await client.post("/chat", json={"message": "Chase INV-1"})
    assert chat.status_code == 200
    events = sse(chat.text)
    names = [name for name, _ in events]
    assert names[-2:] == ["interrupt", "done"] and "tool_call" in names
    pause, done = dict(events)["interrupt"], events[-1][1]
    (draft,) = pause["reminders"]
    assert draft["tone"] == "firm" and "INV-1" in draft["body"]
    assert done["status"] == "interrupted"

    pending = (await client.get("/reminders")).json()
    assert [r["id"] for r in pending] == [draft["reminder_id"]]
    assert pending[0]["thread_id"] == done["thread_id"]  # the run waiting for it
    assert pending[0]["status"] == "pending_approval"

    approved = await client.post(f"/reminders/{draft['reminder_id']}/approve")
    assert approved.status_code == 200
    body = approved.json()
    assert body["resumed"] is True and body["thread_status"] == "completed"
    assert body["reply"].startswith("Approved: INV-1. Nothing has been sent")
    assert body["reminder"]["status"] == "approved" and body["reminder"]["decided_at"]

    again = await client.post(f"/reminders/{draft['reminder_id']}/reject")
    assert again.status_code == 409 and "already approved" in again.json()["error"]["message"]
    every = (await client.get("/reminders", params={"status": "all"})).json()
    assert {r["status"] for r in every} == {"approved"}  # never "sent"
    follow_up = await client.post(
        "/chat", json={"message": "Which invoices are unpaid?", "thread_id": done["thread_id"]}
    )
    assert follow_up.status_code == 200  # the thread is free again


async def test_unknown_reminder_is_404(client: httpx.AsyncClient) -> None:
    response = await client.post("/reminders/00000000-0000-4000-8000-000000000000/approve")
    assert response.status_code == 404 and response.json()["error"]["code"] == "not_found"


async def test_a_pending_draft_can_be_rewritten_until_it_is_decided(
    client: httpx.AsyncClient,
) -> None:
    chat = await client.post("/chat", json={"message": "Chase INV-1"})
    (draft,) = dict(sse(chat.text))["interrupt"]["reminders"]
    reminder_id = draft["reminder_id"]

    listed = (await client.get("/reminders")).json()[0]
    assert listed["invoice_number"] == "INV-1" and listed["vendor_name"] == VENDOR
    assert listed["currency"] == "GBP" and Decimal(listed["total"]) == Decimal("1200.00")
    assert listed["due_date"] == "2026-04-30"

    edited = await client.patch(
        f"/reminders/{reminder_id}", json={"draft": "Dear Acme,\nPlease pay."}
    )
    assert edited.status_code == 200
    assert edited.json()["draft"] == "Dear Acme,\nPlease pay."
    assert (
        edited.json()["status"] == "pending_approval" and edited.json()["invoice_number"] == "INV-1"
    )
    assert (await client.get(f"/reminders/{reminder_id}")).json()[
        "draft"
    ] == "Dear Acme,\nPlease pay."

    blank = await client.patch(f"/reminders/{reminder_id}", json={"draft": "   "})
    assert blank.status_code == 422

    approved = await client.post(f"/reminders/{reminder_id}/approve")
    assert approved.json()["reminder"]["draft"] == "Dear Acme,\nPlease pay."
    late = await client.patch(f"/reminders/{reminder_id}", json={"draft": "changed"})
    assert late.status_code == 409 and "no longer change" in late.json()["error"]["message"]


async def test_editing_an_unknown_reminder_is_404(client: httpx.AsyncClient) -> None:
    response = await client.patch(
        "/reminders/00000000-0000-4000-8000-000000000000", json={"draft": "x"}
    )
    assert response.status_code == 404
