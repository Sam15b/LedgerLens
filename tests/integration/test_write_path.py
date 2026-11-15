"""The unified write path: one ingest -> Postgres, Pinecone and Neo4j, safely repeatable.

Ingests the four fixture PDFs (an invoice, its PO, the PO's contract and the payment that
settles the invoice) through the real pipeline and :class:`StoresSink`, with stand-ins only
for the models: a deterministic dense encoder and an "oracle" extractor that returns each
fixture's expected record. Postgres is a throwaway database and Pinecone a throwaway
Pinecone Local index; in the shared Neo4j graph everything these documents created is
deleted after each test.

    docker compose up -d --wait postgres neo4j pinecone-local
    PINECONE_HOST=http://localhost:5080 pytest -m integration tests/integration/test_write_path.py
"""

import hashlib
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from neo4j import Driver
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from ledgerlens.core.config import PineconeConfig, get_config, get_settings
from ledgerlens.core.schemas import DocType, DocumentStatus, ExtractedRecord, parse_extracted
from ledgerlens.extraction.base import ExtractionResult
from ledgerlens.ingestion.embedding import BM25Encoder, ChunkEmbedder, DenseEmbedder
from ledgerlens.ingestion.loaders import FileLoader, PostgresRegistry
from ledgerlens.ingestion.pipeline import IngestionPipeline, IngestResult, find_inputs
from ledgerlens.ingestion.sinks import StoresSink
from ledgerlens.storage.neo4j.driver import apply_schema, get_driver, run_read, run_write
from ledgerlens.storage.neo4j.linker import GraphLinker, LinkResult
from ledgerlens.storage.pinecone_store import (
    LocalIndexAdmin,
    PineconeStore,
    create_client,
)
from ledgerlens.storage.postgres.models import (
    Base,
    ChunkRow,
    ContractRow,
    DocumentRow,
    InvoiceRow,
    PaymentRow,
    PurchaseOrderRow,
)
from ledgerlens.storage.postgres.repositories import InvoiceRepository

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not get_settings().pinecone_host,
        reason="creates and deletes an index: only run against Pinecone Local (PINECONE_HOST)",
    ),
]

FIXTURES = Path(__file__).parents[1] / "fixtures" / "documents"
NAMES = ["contract", "invoice_digital", "payment_advice", "purchase_order"]
DIM = 16
Sessions = sessionmaker[Session]


# --------------------------------------------------------------------------- stand-ins


class FakeEncoder:
    def encode(self, sentences: list[str], **kwargs: Any) -> Any:
        rows = [
            np.frombuffer(hashlib.sha256(s.encode()).digest()[:DIM], dtype=np.uint8) + 1.0
            for s in sentences
        ]
        return np.stack([row / np.linalg.norm(row) for row in rows])

    def get_embedding_dimension(self) -> int:
        return DIM


class OracleExtractor:
    """Returns the fixture's expected record for whichever fixture the text is from."""

    name = "oracle"
    MARKERS = {
        "INV-2026-0417  498.00": "payment_advice",  # the remittance lists the invoice
        "INVOICE": "invoice_digital",
        "PURCHASE ORDER": "purchase_order",
        "AGREEMENT": "contract",
    }

    def extract(self, text: str, doc_type: DocType | str) -> ExtractionResult:
        name = next(n for marker, n in self.MARKERS.items() if marker in text)
        record = parse_extracted((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
        return ExtractionResult(extractor=self.name, doc_type=DocType(doc_type), record=record)


class CrashingPinecone(PineconeStore):
    """Dies (like a killed process) on the ``crash_on``-th upsert."""

    def __init__(self, store: PineconeStore, crash_on: int) -> None:
        super().__init__(store.index, namespace=store.namespace, config=store.config)
        self.calls, self.crash_on = 0, crash_on

    def upsert_chunks(self, chunks: Any, **kwargs: Any) -> int:
        self.calls += 1
        if self.calls == self.crash_on:
            raise KeyboardInterrupt  # not an Exception: nothing marks the document failed
        return super().upsert_chunks(chunks, **kwargs)


class FlakyLinker(GraphLinker):
    """Neo4j is "down" for contracts until ``healthy`` is set."""

    healthy = False

    def link(self, record: ExtractedRecord, **kwargs: Any) -> LinkResult:
        if not self.healthy and record.doc_type == "contract":
            raise ConnectionError("neo4j unavailable")
        return super().link(record, **kwargs)


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def pinecone() -> Iterator[PineconeStore]:
    settings = get_settings()
    admin = LocalIndexAdmin(settings.pinecone_host)
    name = f"it-write-{uuid.uuid4().hex[:8]}"
    info = admin.create(name, dimension=DIM, metric="dotproduct", cloud="aws", region="us-east-1")
    assert info.host
    try:
        index = create_client(settings.pinecone_api_key, settings.pinecone_host).Index(
            host=info.host
        )
        store = PineconeStore(index, namespace="eval", config=PineconeConfig(dimension=DIM))  # type: ignore[arg-type]
        yield store
    finally:
        admin.delete(name)


@pytest.fixture(scope="module")
def driver() -> Driver:
    driver = get_driver()
    apply_schema(driver)
    return driver


@pytest.fixture
def stores(sessions: Sessions, pinecone: PineconeStore, driver: Driver) -> Iterator[dict[str, Any]]:
    yield {"sessions": sessions, "pinecone": pinecone, "driver": driver}
    with sessions() as session:
        ids = list(session.scalars(select(DocumentRow.id)))
    for document_id in ids:
        pinecone.delete_document(document_id)
    run_write(
        driver,
        """
        MATCH (n) WHERE n.document_id IN $ids
        OPTIONAL MATCH (n)-[:HAS_LINE]->(line:LineItem)
        OPTIONAL MATCH (u:UnresolvedLink {source_id: n.id})
        DETACH DELETE line, u, n
        """,
        {"ids": ids},
    )
    run_write(driver, "MATCH (v:Vendor {name: 'Acme Supplies Ltd'}) WHERE NOT (v)--() DELETE v")


@pytest.fixture
def raw(tmp_path: Path) -> list[Path]:
    out = tmp_path / "raw"
    out.mkdir()
    for name in NAMES:
        shutil.copy(FIXTURES / f"{name}.pdf", out / f"{name}.pdf")
    return find_inputs(out)


def sink_for(stores: dict[str, Any], **overrides: Any) -> StoresSink:
    parts = {
        "sessions": stores["sessions"],
        "pinecone": stores["pinecone"],
        "linker": GraphLinker(stores["driver"]),
    } | overrides
    return StoresSink(**parts)


def ingest(
    stores: dict[str, Any], paths: list[Path], sink: StoresSink, tmp: Path
) -> list[IngestResult]:
    """One run, as a fresh process would do it: new loader, pipeline and BM25 fit."""
    config = get_config().ingestion
    pipeline = IngestionPipeline(
        config,
        loader=FileLoader(config.loader, registry=PostgresRegistry(stores["sessions"])),
        extractor=OracleExtractor(),
        embedder=ChunkEmbedder(DenseEmbedder(model=FakeEncoder()), BM25Encoder()),
        on_status=sink.mark,
    )
    return pipeline.ingest(paths, sink, bm25_path=tmp / "bm25.json")


# --------------------------------------------------------------------------- state


def count(sessions: Sessions, model: type[Base]) -> int:
    with sessions() as session:
        return session.scalar(select(func.count()).select_from(model)) or 0


def state(stores: dict[str, Any]) -> dict[str, Any]:
    """Everything the write path produced, with document ids replaced by file names so
    two runs (different ids) can be compared."""
    sessions: Sessions = stores["sessions"]
    with sessions() as session:
        documents = {row.id: row for row in session.scalars(select(DocumentRow))}
        chunks = sorted(session.scalars(select(ChunkRow.id)))
    names = {doc_id: Path(row.filename).stem for doc_id, row in documents.items()}

    def named(value: str) -> str:
        for doc_id, name in names.items():
            value = value.replace(doc_id, name)
        return value

    vectors = sorted(v for d in names for v in stores["pinecone"].document_ids(d))
    edges = run_read(
        stores["driver"],
        "MATCH (a)-[r]->(b) WHERE a.document_id IN $ids OR b.document_id IN $ids "
        "RETURN a.id AS a, type(r) AS type, b.id AS b",
        {"ids": list(names)},
    )
    unresolved = run_read(
        stores["driver"],
        "MATCH (u:UnresolvedLink) WHERE u.source_id IN $ids RETURN u.id AS id",
        {"ids": list(names)},
    )
    return {
        "documents": sorted(
            (names[i], d.status, d.doc_type, d.error) for i, d in documents.items()
        ),
        "records": {
            m.__tablename__: count(sessions, m)
            for m in (InvoiceRow, PurchaseOrderRow, PaymentRow, ContractRow)
        },
        "chunks": [named(c) for c in chunks],
        "vectors": [named(v) for v in vectors],
        "edges": sorted((named(e["a"]), e["type"], named(e["b"])) for e in edges),
        "unresolved": sorted(named(u["id"]) for u in unresolved),
    }


def assert_complete(snapshot: dict[str, Any]) -> None:
    """The final state a correct ingest of the four fixtures must reach."""
    assert snapshot["documents"] == [
        ("contract", "ready", "contract", None),
        ("invoice_digital", "ready", "invoice", None),
        ("payment_advice", "ready", "payment", None),
        ("purchase_order", "ready", "purchase_order", None),
    ]
    assert snapshot["records"] == {
        "invoices": 1,
        "purchase_orders": 1,
        "payments": 1,
        "contracts": 1,
    }
    assert snapshot["chunks"] and snapshot["vectors"] == snapshot["chunks"]  # stores agree
    kinds = {(a.split("#")[0], t, b.split("#")[0]) for a, t, b in snapshot["edges"]}
    assert {
        ("invoice_digital", "REFERENCES", "purchase_order"),
        ("payment_advice", "SETTLES", "invoice_digital"),
        ("contract", "GOVERNS", "purchase_order"),
    } <= kinds
    assert {t for _, t, _ in snapshot["edges"]} >= {"ISSUED", "PARTY_TO", "HAS_LINE"}
    assert len(snapshot["edges"]) == len(set(snapshot["edges"]))  # no duplicate edges
    assert snapshot["unresolved"] == []


# --------------------------------------------------------------------------- tests


def test_one_ingest_writes_all_three_stores_consistently(
    stores: dict[str, Any], raw: list[Path], tmp_path: Path
) -> None:
    results = ingest(stores, raw, sink_for(stores), tmp_path)

    assert [r.document.status for r in results] == ["ready"] * 4
    assert_complete(state(stores))
    with stores["sessions"]() as session:
        invoice = InvoiceRepository(session).find_by_number("INV-2026-0417")[0]
    assert invoice.status == "paid"  # the remittance settled it in Postgres too
    hits = stores["pinecone"].query([1.0] * DIM, top_k=10, filter={"doc_number": "INV-2026-0417"})
    assert {h.document_id for h in hits} == {invoice.document_id}


def stamps(sessions: Sessions) -> dict[str, Any]:
    with sessions() as session:
        rows = session.execute(select(DocumentRow.id, DocumentRow.updated_at))
        return {row.id: row.updated_at for row in rows}


def test_reingest_is_a_no_op(stores: dict[str, Any], raw: list[Path], tmp_path: Path) -> None:
    sink = sink_for(stores)
    first = ingest(stores, raw, sink, tmp_path)
    before, touched = state(stores), stamps(stores["sessions"])

    again = ingest(stores, raw, sink, tmp_path)  # a new run over the same files

    assert all(r.duplicate for r in again)
    assert [r.document.status for r in again] == ["ready"] * 4
    assert stamps(stores["sessions"]) == touched  # not a single row was rewritten
    assert state(stores) == before

    for result in first:  # writing the same results again converges on the same state
        assert sink.write(result).status is DocumentStatus.READY
    assert state(stores) == before


def test_a_failed_ingest_is_recorded_and_can_be_retried(
    stores: dict[str, Any], raw: list[Path], tmp_path: Path
) -> None:
    flaky = FlakyLinker(stores["driver"])
    results = ingest(stores, raw, sink_for(stores, linker=flaky), tmp_path)

    failed = {
        r.document.filename: r.document.error for r in results if r.document.status == "failed"
    }
    assert failed == {"contract.pdf": "neo4j: ConnectionError: neo4j unavailable"}
    snapshot = state(stores)
    assert ("contract", "failed", "contract", failed["contract.pdf"]) in snapshot["documents"]
    assert any(u.endswith("|GOVERNS|MSA-2026-07") for u in snapshot["unresolved"])

    flaky.healthy = True
    retried = ingest(stores, raw, sink_for(stores, linker=flaky), tmp_path)

    by_name = {r.document.filename: r for r in retried}
    assert by_name["contract.pdf"].retry and not by_name["contract.pdf"].duplicate
    assert all(by_name[n].duplicate for n in by_name if n != "contract.pdf")
    assert_complete(state(stores))


def test_killed_mid_batch_then_rerun_reaches_the_correct_state(
    stores: dict[str, Any], raw: list[Path], tmp_path: Path
) -> None:
    crashing = CrashingPinecone(stores["pinecone"], crash_on=2)
    with pytest.raises(KeyboardInterrupt):
        ingest(stores, raw, sink_for(stores, pinecone=crashing), tmp_path)

    interrupted = state(stores)
    statuses = [status for _, status, _, _ in interrupted["documents"]]
    assert statuses.count("ready") == 1 and "processing" in statuses  # died part-way

    rerun = ingest(stores, raw, sink_for(stores), tmp_path)

    assert sum(r.retry for r in rerun) == 3 and sum(r.duplicate for r in rerun) == 1
    assert_complete(state(stores))
    assert count(stores["sessions"], DocumentRow) == 4  # same documents, not new ones
