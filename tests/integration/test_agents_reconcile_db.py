"""'Does INV-889 match its PO and contract rate?' end to end: router -> reconciler -> the
MCP server (in memory, langchain-mcp-adapters) -> the reconcile tool -> Postgres
(throwaway database). No model: the reply is the report itself.

    docker compose up -d --wait postgres
    pytest -m integration tests/integration/test_agents_reconcile_db.py
"""

from datetime import date
from decimal import Decimal

import pytest
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.shared.memory import create_connected_server_and_client_session
from sqlalchemy.orm import Session, sessionmaker

from ledgerlens.agents.graph import AgentNodes, build_graph
from ledgerlens.agents.nodes.reconciler import ReconcilerNode
from ledgerlens.agents.nodes.router import Router
from ledgerlens.agents.tools import ToolBox
from ledgerlens.core.schemas import Contract, Invoice, LineItem, PurchaseOrder, RateCardEntry
from ledgerlens.mcp_server.server import create_server
from ledgerlens.mcp_server.services import Services
from ledgerlens.retrieval.analyze import VendorFinder
from ledgerlens.storage.neo4j.linker import VendorCandidate
from ledgerlens.storage.postgres.db import session_scope
from ledgerlens.storage.postgres.repositories import Repositories

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

VENDOR = "Acme Supplies Ltd"
WRAP = "Pallet wrap 500m"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def line(qty: int, price: str) -> LineItem:
    amount = Decimal(qty) * Decimal(price)
    return LineItem(
        description=WRAP, quantity=Decimal(qty), unit_price=Decimal(price), amount=amount
    )


@pytest.fixture
def services(sessions: sessionmaker[Session]) -> Services:
    with session_scope(sessions) as session:
        repos = Repositories(session)
        repos.vendors.upsert("V001", VENDOR)
        repos.contracts.upsert(
            Contract(
                contract_number="CT-V001",
                vendor_name=VENDOR,
                start_date=date(2026, 1, 1),
                payment_terms_days=30,
                currency="GBP",
                rate_card=[RateCardEntry(item=WRAP, unit="roll", unit_price=Decimal("42.41"))],
            ),
            id="CT-V001",
            vendor_id="V001",
        )
        repos.purchase_orders.upsert(
            PurchaseOrder(
                po_number="PO-1042",
                vendor_name=VENDOR,
                order_date=date(2026, 3, 1),
                contract_number="CT-V001",
                currency="GBP",
                total=Decimal("84.82"),
                line_items=[line(2, "42.41")],
            ),
            id="PO-1042",
            vendor_id="V001",
        )
        repos.invoices.upsert(
            Invoice(
                invoice_number="INV-889",
                vendor_name=VENDOR,
                invoice_date=date(2026, 3, 14),
                due_date=date(2026, 4, 13),
                po_number="PO-1042",
                currency="GBP",
                subtotal=Decimal("90.00"),
                tax=Decimal(0),
                total=Decimal("90.00"),
                line_items=[line(2, "45.00")],  # 6.1 % over the contract rate
            ),
            id="I-889",
            vendor_id="V001",
        )
    return Services(
        sessions=sessions,
        vendors=VendorFinder([VendorCandidate("V001", VENDOR)]),
        today=date(2026, 3, 20),
    )


async def test_inv_889_mismatch_list(services: Services) -> None:
    server = create_server(services)
    async with create_connected_server_and_client_session(server) as session:
        box = ToolBox(await load_mcp_tools(session))
        graph = build_graph(
            AgentNodes(
                router=Router(None, vendors=services.vendors), reconciler=ReconcilerNode(box)
            )
        )
        result = await graph.ainvoke(
            {"messages": [HumanMessage("Does INV-889 match its PO and contract rate?")]}
        )
    report = result["reconciliation"]
    assert result["intent"] == "reconcile"
    assert report["status"] == "mismatch"
    assert report["mismatches"] == ["unit_price"]
    reply = result["messages"][-1].content
    assert reply.startswith("INV-889 does not match against PO PO-1042 and contract CT-V001.")
    assert "Pallet wrap 500m at 45.00 vs 42.41 (+6.1%)" in reply
    assert "Passed: po_exists, vendor_match, line_qty, total, payment, terms." in reply
