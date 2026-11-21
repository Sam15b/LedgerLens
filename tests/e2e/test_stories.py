"""The five user stories of docs/01-project-overview.md, end to end through the API.

Each test drives the FastAPI app the way the dashboard does (``/chat`` over SSE,
``/documents``, ``/anomalies``, ``/reminders``). Behind it everything is real except the
models and the vectors:

* the agent graph with every specialist node, **without a model** (the router's rules,
  QA's keyword plan and evidence answer, the rules extractor, the reconciler's report),
  over a Postgres checkpointer-free in-memory saver;
* the MCP server (in memory, through langchain-mcp-adapters) with its real tools;
* Postgres: a throwaway database (``tests/integration/conftest.py``);
* Neo4j: the compose graph, with this test's records under a vendor id no other data
  uses, deleted afterwards;
* the ingestion worker: arq on the compose Redis (its own queue, burst mode, in-process).

Vector search is a stand-in with no hits: these stories are answered from the graph and
Postgres, and retrieval quality is measured by the retrieval benchmark, not here.

    docker compose up -d --wait postgres neo4j redis
    pytest -m integration tests/e2e
"""

import json
import logging
import random
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from arq.worker import Worker
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.memory import InMemorySaver
from mcp.shared.memory import create_connected_server_and_client_session
from neo4j import Driver
from sqlalchemy.orm import Session, sessionmaker

from ledgerlens.agents.graph import AgentNodes, build_graph
from ledgerlens.agents.nodes.extractor import ExtractorNode, default_extractor
from ledgerlens.agents.nodes.qa import QANode
from ledgerlens.agents.nodes.reconciler import ReconcilerNode
from ledgerlens.agents.nodes.reminder_writer import ReminderWriterNode
from ledgerlens.agents.nodes.router import Router
from ledgerlens.agents.tools import ToolBox
from ledgerlens.anomaly.scorer import AnomalyScorer, score_stored
from ledgerlens.api.agent import AgentProvider, AgentRuntime
from ledgerlens.api.main import create_app
from ledgerlens.api.routers import anomalies, documents, reminders
from ledgerlens.core.config import get_config
from ledgerlens.core.schemas import (
    Contract,
    Invoice,
    LineItem,
    PurchaseOrder,
    RateCardEntry,
)
from ledgerlens.ingestion.loaders import FileLoader, PostgresRegistry
from ledgerlens.ingestion.pipeline import IngestionPipeline, make_extractor
from ledgerlens.ingestion.sinks import StoresSink
from ledgerlens.mcp_server.server import create_server
from ledgerlens.mcp_server.services import Services
from ledgerlens.retrieval import QueryFilters, RetrievalResult
from ledgerlens.retrieval.analyze import VendorFinder
from ledgerlens.retrieval.graph import Neo4jRunner
from ledgerlens.storage.neo4j.driver import get_driver
from ledgerlens.storage.neo4j.linker import GraphLinker, VendorCandidate
from ledgerlens.storage.postgres.db import session_scope
from ledgerlens.storage.postgres.repositories import Repositories
from ledgerlens.workers.ingest_worker import Ingestor, enqueue, ingest_document, redis_settings

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

TODAY = date(2026, 5, 20)
FIXTURE = Path(__file__).parents[1] / "fixtures" / "documents" / "invoice_digital.pdf"
VENDOR = "Northwind Parts Ltd"
WRAP = "Pallet wrap 500m"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# --------------------------------------------------------------------------- the ledger


@dataclass(frozen=True)
class Ledger:
    """This test's records. Numbers carry a random tag so they never meet the shared
    graph's (``INV-<tag>1`` ...)."""

    tag: str

    @property
    def vendor_id(self) -> str:
        return f"VE2E{self.tag}"

    def inv(self, n: int) -> str:
        return f"INV-{self.tag}{n}"

    def po(self, n: int) -> str:
        return f"PO-{self.tag}{n}"

    @property
    def contract(self) -> str:
        return f"CT-{self.tag}"

    def record_id(self, number: str) -> str:
        return f"E2E-{number}"


def item(description: str, qty: int, price: str) -> LineItem:
    amount = Decimal(qty) * Decimal(price)
    return LineItem(
        description=description, quantity=Decimal(qty), unit_price=Decimal(price), amount=amount
    )


def records(ledger: Ledger) -> list[Contract | PurchaseOrder | Invoice]:
    """Three unpaid invoices on three POs:

    1. 7,500.00, due 2026-04-10: **40 days overdue** on TODAY, over 5k;
    2. 90.00 for pallet wrap at 45.00 against a contract rate of 42.41: **a rate breach**,
       not due until 2026-05-31;
    3. 1,200.00, due 2026-05-10: 10 days overdue, under 5k.
    """
    brackets = item("Steel brackets", 300, "25.00")
    wrap_po, wrap_inv = item(WRAP, 2, "42.41"), item(WRAP, 2, "45.00")
    bolts = item("Hex bolts (box)", 40, "30.00")
    common: dict[str, Any] = {"vendor_name": VENDOR, "currency": "USD"}
    return [
        Contract(
            contract_number=ledger.contract,
            start_date=date(2026, 1, 1),
            payment_terms_days=30,
            rate_card=[RateCardEntry(item=WRAP, unit="roll", unit_price=Decimal("42.41"))],
            **common,
        ),
        PurchaseOrder(
            po_number=ledger.po(1),
            order_date=date(2026, 3, 1),
            total=brackets.amount,
            line_items=[brackets],
            **common,
        ),
        PurchaseOrder(
            po_number=ledger.po(2),
            order_date=date(2026, 4, 20),
            contract_number=ledger.contract,
            total=wrap_po.amount,
            line_items=[wrap_po],
            **common,
        ),
        PurchaseOrder(
            po_number=ledger.po(3),
            order_date=date(2026, 3, 25),
            total=bolts.amount,
            line_items=[bolts],
            **common,
        ),
        Invoice(
            invoice_number=ledger.inv(1),
            invoice_date=date(2026, 3, 11),
            due_date=date(2026, 4, 10),
            po_number=ledger.po(1),
            total=brackets.amount,
            line_items=[brackets],
            **common,
        ),
        Invoice(
            invoice_number=ledger.inv(2),
            invoice_date=date(2026, 5, 1),
            due_date=date(2026, 5, 31),
            po_number=ledger.po(2),
            subtotal=wrap_inv.amount,
            tax=Decimal(0),
            total=wrap_inv.amount,
            line_items=[wrap_inv],
            **common,
        ),
        Invoice(
            invoice_number=ledger.inv(3),
            invoice_date=date(2026, 4, 10),
            due_date=date(2026, 5, 10),
            po_number=ledger.po(3),
            total=bolts.amount,
            line_items=[bolts],
            **common,
        ),
    ]


def number_of(record: Contract | PurchaseOrder | Invoice) -> str:
    if isinstance(record, Invoice):
        return record.invoice_number
    if isinstance(record, PurchaseOrder):
        return record.po_number
    return record.contract_number or ""


def seed(sessions: sessionmaker[Session], driver: Driver, ledger: Ledger) -> None:
    """Postgres and Neo4j get the same records; every invoice is scored (rules only)."""
    with session_scope(sessions) as session:
        repos = Repositories(session)
        repos.vendors.upsert(ledger.vendor_id, VENDOR)
        for record in records(ledger):
            rid = ledger.record_id(number_of(record))
            repos.upsert_record(record, id=rid, vendor_id=ledger.vendor_id)
    linker = GraphLinker(driver)
    linker.resolver.upsert(ledger.vendor_id, VENDOR)
    for record in records(ledger):
        linker.link(record, id=ledger.record_id(number_of(record)), vendor_id=ledger.vendor_id)
    scorer = AnomalyScorer()  # rules only
    with session_scope(sessions) as session:
        for n in (1, 2, 3):
            score_stored(session, ledger.record_id(ledger.inv(n)), scorer)


def unseed(driver: Driver, ledger: Ledger) -> None:
    with driver.session() as session:
        session.run(
            """
            MATCH (n) WHERE n.vendor_id = $vendor OR (n:Vendor AND n.id = $vendor)
            OPTIONAL MATCH (n)-[:HAS_LINE]->(line:LineItem)
            OPTIONAL MATCH (u:UnresolvedLink {source_id: n.id})
            DETACH DELETE line, u, n
            """,
            vendor=ledger.vendor_id,
        ).consume()


# --------------------------------------------------------------------------- the stack


class NoVectors:
    """The vector store stand-in: no hits (see the module docstring)."""

    name = "none"

    def retrieve(
        self, query: str, *, filters: QueryFilters | None = None, k: int = 5
    ) -> RetrievalResult:
        return RetrievalResult.build(query, self.name, [], [])


class Stack:
    """The API app over the agents, the MCP server, the stores and the worker."""

    def __init__(
        self, sessions: sessionmaker[Session], ledger: Ledger, box: ToolBox, uploads: Path
    ) -> None:
        self.sessions = sessions
        self.ledger = ledger
        self.queue = f"e2e-{uuid.uuid4().hex[:8]}"
        vendors = VendorFinder([VendorCandidate(ledger.vendor_id, VENDOR)])
        nodes = AgentNodes(
            router=Router(None, vendors=vendors),
            qa=QANode(box, None),
            extractor=ExtractorNode(box, default_extractor(None)),
            reconciler=ReconcilerNode(box),
            reminder_writer=ReminderWriterNode(box, today=TODAY),
        )
        graph = build_graph(nodes, checkpointer=InMemorySaver())

        async def factory(stack: Any) -> AgentRuntime:
            return AgentRuntime(graph)

        app = create_app(AgentProvider(factory))
        overrides = app.dependency_overrides
        overrides[reminders.get_sessions] = lambda: sessions
        overrides[documents.get_registry] = lambda: PostgresRegistry(sessions)
        overrides[documents.get_raw_store] = lambda: documents.RawFileStore(uploads)
        overrides[documents.get_document_store] = lambda: documents.DocumentStore(sessions)
        overrides[documents.get_enqueue] = lambda: self.enqueue
        overrides[anomalies.get_anomaly_store] = lambda: anomalies.AnomalyStore(sessions)
        self.app = app

    # ------------------------------------------------------------------ worker

    async def enqueue(self, path: Path) -> str | None:
        (job_id,) = await enqueue([path], redis_settings(), queue_name=self.queue)
        return job_id

    async def work(self) -> None:
        """Run every queued ingestion job (rules extractor, Postgres only, scored)."""
        config = get_config().ingestion
        sink = StoresSink(sessions=self.sessions, scorer=AnomalyScorer())
        pipeline = IngestionPipeline(
            config,
            loader=FileLoader(config.loader, registry=PostgresRegistry(self.sessions)),
            extractor=make_extractor("rules"),
            on_status=sink.mark,
        )
        worker = Worker(
            functions=[ingest_document],
            queue_name=self.queue,
            redis_settings=redis_settings(),
            burst=True,
            handle_signals=False,
            ctx={"ingestor": Ingestor(pipeline, sink)},
            max_jobs=1,
            poll_delay=0.05,
        )
        try:
            await worker.main()
        finally:  # Worker.close() needs SIGUSR1, which Windows lacks
            if worker.pool is not None:
                await worker.pool.aclose()


@pytest.fixture(scope="module")
def driver() -> Driver:
    return get_driver()


@pytest.fixture
def ledger(sessions: sessionmaker[Session], driver: Driver) -> Iterator[Ledger]:
    ledger = Ledger(tag=f"{random.randrange(10**6):06d}")
    try:
        seed(sessions, driver, ledger)
        yield ledger
    finally:
        unseed(driver, ledger)


@pytest.fixture
async def stack(
    sessions: sessionmaker[Session], driver: Driver, ledger: Ledger, tmp_path: Path
) -> AsyncIterator[Stack]:
    services = Services(
        sessions=sessions,
        retriever=NoVectors(),
        graph=Neo4jRunner(driver),
        vendors=VendorFinder([VendorCandidate(ledger.vendor_id, VENDOR)]),
        record_sink=StoresSink(sessions=sessions, scorer=AnomalyScorer()),
        scorer=AnomalyScorer(),
        today=TODAY,
    )
    async with create_connected_server_and_client_session(create_server(services)) as mcp:
        box = ToolBox(await load_mcp_tools(mcp))
        yield Stack(sessions, ledger, box, tmp_path)


@pytest.fixture
async def api(stack: Stack) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=stack.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
        yield client


# --------------------------------------------------------------------------- helpers


def sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in text.strip().split("\n\n"):
        name, data = block.split("\n")
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


@dataclass
class Turn:
    events: list[tuple[str, dict[str, Any]]]

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.events]

    def of(self, name: str) -> list[dict[str, Any]]:
        return [data for event, data in self.events if event == name]

    @property
    def tools(self) -> list[str]:
        return [call["name"] for call in self.of("tool_call")]

    @property
    def reply(self) -> str:
        return "".join(token["text"] for token in self.of("token"))

    @property
    def done(self) -> dict[str, Any]:
        assert self.names[-1] == "done", self.names
        return self.events[-1][1]


async def chat(api: httpx.AsyncClient, message: str, **body: Any) -> Turn:
    response = await api.post("/chat", json={"message": message, **body})
    assert response.status_code == 200, response.text
    turn = Turn(sse(response.text))
    assert "error" not in turn.names, turn.of("error")
    return turn


def routed(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The router's hand-offs, as logged (``routed to qa by rules``)."""
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "ledgerlens.agents.nodes.router" and r.getMessage().startswith("routed")
    ]


# --------------------------------------------------------------------------- stories


async def test_unpaid_invoices_over_5k_from_a_vendor(
    api: httpx.AsyncClient, ledger: Ledger, caplog: pytest.LogCaptureFixture
) -> None:
    """AP clerk: "Show unpaid invoices from Acme over $5k" -- Router -> QA (GraphRAG)."""
    caplog.set_level(logging.INFO)
    turn = await chat(api, f"Show unpaid invoices from {VENDOR} over $5k")

    assert routed(caplog) == ["routed to question by rules"]
    assert turn.tools[:2] == ["search_documents", "graph_query"]
    graph_call = turn.of("tool_call")[1]["args"]
    assert graph_call["template"] == "unpaid_with_po"
    assert graph_call["params"]["vendor_ids"] == [ledger.vendor_id]
    assert len(turn.of("tool_result")) == len(turn.tools)
    assert ledger.inv(1) in turn.reply and "7,500.00" in turn.reply
    assert ledger.inv(3) not in turn.reply  # 1,200.00: under 5k
    assert ledger.inv(2) not in turn.reply
    assert turn.done["status"] == "completed"


async def test_upload_an_invoice_and_get_its_fields(
    api: httpx.AsyncClient, stack: Stack, caplog: pytest.LogCaptureFixture
) -> None:
    """AP clerk: upload a new invoice and get its fields -- Router -> Extractor."""
    files = {"file": ("invoice_digital.pdf", FIXTURE.read_bytes(), "application/pdf")}
    uploaded = await api.post("/documents", files=files)
    assert uploaded.status_code == 202
    document_id = uploaded.json()["document_id"]
    await stack.work()
    stored = (await api.get(f"/documents/{document_id}")).json()
    assert stored["status"] == "ready", stored["error"]

    caplog.set_level(logging.INFO)
    turn = await chat(api, "Extract the fields of this invoice", document_ids=[document_id])

    assert routed(caplog) == ["routed to extract by rules"]
    assert turn.tools[:2] == ["get_document_text", "save_extraction"]
    assert "INV-2026-0417" in turn.reply and "Acme Supplies Ltd" in turn.reply
    assert "498.00" in turn.reply
    assert turn.done["status"] == "completed"
    after = (await api.get(f"/documents/{document_id}")).json()
    assert after["record"]["invoice_number"] == "INV-2026-0417"
    assert after["record"]["total"] == "498.0000"


async def test_does_an_invoice_match_its_po_and_contract_rate(
    api: httpx.AsyncClient, ledger: Ledger, caplog: pytest.LogCaptureFixture
) -> None:
    """Finance controller: "Does INV-889 match its PO and contract rate?" -- Router ->
    Reconciler."""
    caplog.set_level(logging.INFO)
    turn = await chat(api, f"Does {ledger.inv(2)} match its PO and contract rate?")

    assert routed(caplog) == ["routed to reconcile by rules"]
    assert turn.tools == ["reconcile"]
    (result,) = turn.of("tool_result")
    assert result["name"] == "reconcile"
    reply = turn.reply
    against = f"against PO {ledger.po(2)} and contract {ledger.contract}."
    assert reply.startswith(f"{ledger.inv(2)} does not match {against}")
    assert "Mismatches: unit_price." in reply
    assert f"{WRAP} at 45.00 vs 42.41 (+6.1%)" in reply

    # prices and quantities agree with the PO, but it is 40 days past due and unpaid
    overdue = await chat(api, f"Does {ledger.inv(1)} match its PO?")
    assert overdue.tools == ["reconcile"]
    assert "Mismatches: payment." in overdue.reply
    assert "7500.00 USD outstanding, 40 day(s) past due 2026-04-10" in overdue.reply
    assert "Passed: po_exists, vendor_match, line_qty, total, terms." in overdue.reply


async def test_review_flagged_invoices(api: httpx.AsyncClient, ledger: Ledger) -> None:
    """Finance controller: review flagged invoices -- anomaly service -> dashboard."""
    flagged = (await api.get("/anomalies")).json()
    assert [f["invoice_number"] for f in flagged] == [ledger.inv(2)]  # the rate breach
    (flag,) = flagged
    assert flag["flagged"] and flag["score"] >= 0.5 and flag["model_version"] == "rules-only"
    assert any(r["source"] == "rule" and "42.41" in r["text"] for r in flag["reasons"])
    every = (await api.get("/anomalies", params={"min_score": 0})).json()
    assert {f["invoice_number"] for f in every} == {ledger.inv(n) for n in (1, 2, 3)}

    reviewed = await api.post(
        f"/anomalies/{flag['id']}/review",
        json={"verdict": "confirmed", "note": "billed above the contract rate"},
    )
    assert reviewed.status_code == 200 and reviewed.json()["review"] == "confirmed"
    assert (await api.get("/anomalies")).json() == []  # nothing left to review
    confirmed = (await api.get("/anomalies", params={"review": "confirmed"})).json()
    assert [f["id"] for f in confirmed] == [flag["id"]]
    assert confirmed[0]["review_note"] == "billed above the contract rate"
