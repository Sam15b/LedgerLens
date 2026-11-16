"""Cross-store parity after a bulk load: Postgres rows == Neo4j nodes, Postgres chunks ==
Pinecone vectors (``describe_index_stats``) -- the Phase 3 exit check, on a small ledger.

The ledger is the four fixture documents' records (a contract, its PO, an invoice and the
payment settling it) plus structured-only records -- two more invoices, and one from a
vendor that is not in the master -- loaded the way ``scripts/load_synthetic.py`` loads
the synthetic set: records first, then the fixture PDFs as their rendered documents.
Postgres is a throwaway database and Pinecone a throwaway Pinecone Local index. Neo4j is
shared, so every id is prefixed per test and the graph is compared by how much it grew;
the test's nodes are deleted afterwards.

    docker compose up -d --wait postgres neo4j pinecone-local
    PINECONE_HOST=http://localhost:5080 pytest -m integration tests/integration/test_parity.py
"""

import hashlib
import json
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from neo4j import Driver
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from ledgerlens.core.config import PineconeConfig, get_settings
from ledgerlens.ingestion.embedding import BM25Encoder, ChunkEmbedder, DenseEmbedder
from ledgerlens.storage.bulk_load import (
    PARITY,
    LedgerLine,
    LedgerLoader,
    LoadReport,
    Rendered,
    RenderedReport,
    StoreCounts,
    count_stores,
    load_rendered,
    parse_line,
)
from ledgerlens.storage.neo4j.driver import apply_schema, get_driver, run_read, run_write
from ledgerlens.storage.neo4j.linker import GraphLinker, new_vendor_id
from ledgerlens.storage.pinecone_store import LocalIndexAdmin, PineconeStore, create_client
from ledgerlens.storage.postgres.models import InvoiceRow, PaymentRow

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not get_settings().pinecone_host,
        reason="creates and deletes an index: only run against Pinecone Local (PINECONE_HOST)",
    ),
]

FIXTURES = Path(__file__).parents[1] / "fixtures" / "documents"
DIM = 16
Sessions = sessionmaker[Session]


class FakeEncoder:
    def encode(self, sentences: list[str], **kwargs: Any) -> Any:
        rows = [
            np.frombuffer(hashlib.sha256(s.encode()).digest()[:DIM], dtype=np.uint8) + 1.0
            for s in sentences
        ]
        return np.stack([row / np.linalg.norm(row) for row in rows])

    def get_embedding_dimension(self) -> int:
        return DIM


# --------------------------------------------------------------------------- the ledger


def fixture(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    return data


def extra_invoice(number: str, vendor_name: str) -> dict[str, Any]:
    return {
        "doc_type": "invoice",
        "invoice_number": number,
        "vendor_name": vendor_name,
        "invoice_date": "2026-10-20",
        "due_date": "2026-11-19",
        "currency": "GBP",
        "total": "120.00",
    }


class Ledger:
    """A prefixed ledger: ``lines`` to load, ``rendered`` fixture PDFs named by record."""

    def __init__(self, tmp: Path) -> None:
        self.prefix = f"it{uuid.uuid4().hex[:8]}-"
        p = self.prefix
        self.vendor_id = f"{p}V1"
        self.stranger = f"Stranger {p} Trading"  # matches nothing: becomes a new vendor
        rendered = {
            f"{p}CT": "contract",
            f"{p}PO": "purchase_order",
            f"{p}INV": "invoice_digital",
            f"{p}PAY": "payment_advice",
        }
        raw: list[dict[str, Any]] = [
            {
                "record_type": "vendor",
                "id": self.vendor_id,
                "data": {"vendor_id": self.vendor_id, "name": "Acme Supplies Ltd"},
            }
        ]
        for record_id, name in rendered.items():
            data = fixture(name)
            kind = data["doc_type"]
            raw.append({"record_type": kind, "id": record_id, "vendor_id": p + "V1", "data": data})
        raw += [
            {
                "record_type": "invoice",
                "id": f"{p}X{n}",
                "vendor_id": self.vendor_id,
                "data": extra_invoice(f"{p}INV-{n}", "ACME Supplies"),  # an alias
            }
            for n in (1, 2)
        ]
        raw.append(
            {
                "record_type": "invoice",
                "id": f"{p}X3",
                "vendor_id": None,
                "data": extra_invoice(f"{p}INV-3", self.stranger),
            }
        )
        self.lines: list[LedgerLine] = [parse_line(obj) for obj in raw]
        out = tmp / "rendered"
        out.mkdir()
        self.rendered: list[Rendered] = []
        for record_id, name in rendered.items():
            path = out / f"{record_id}.pdf"
            shutil.copy(FIXTURES / f"{name}.pdf", path)
            self.rendered.append(Rendered(record_id, fixture(name)["doc_type"], path))

    @property
    def vendor_ids(self) -> list[str]:
        return [self.vendor_id, new_vendor_id(self.stranger)]


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def pinecone() -> Iterator[PineconeStore]:
    settings = get_settings()
    admin = LocalIndexAdmin(settings.pinecone_host)
    name = f"it-parity-{uuid.uuid4().hex[:8]}"
    info = admin.create(name, dimension=DIM, metric="dotproduct", cloud="aws", region="us-east-1")
    assert info.host
    try:
        index = create_client(settings.pinecone_api_key, settings.pinecone_host).Index(
            host=info.host
        )
        yield PineconeStore(index, namespace="eval", config=PineconeConfig(dimension=DIM))  # type: ignore[arg-type]
    finally:
        admin.delete(name)


@pytest.fixture
def driver() -> Driver:
    driver = get_driver()
    apply_schema(driver)
    return driver


@pytest.fixture
def ledger(tmp_path: Path, driver: Driver) -> Iterator[Ledger]:
    ledger = Ledger(tmp_path)
    yield ledger
    run_write(
        driver,
        """
        MATCH (n) WHERE n.id STARTS WITH $prefix OR n.id IN $vendors
        OPTIONAL MATCH (u:UnresolvedLink {source_id: n.id})
        DETACH DELETE u, n
        """,
        {"prefix": ledger.prefix, "vendors": ledger.vendor_ids},
    )


def graph_nodes(driver: Driver) -> dict[str, int]:
    rows = run_read(driver, "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS n")
    return {row["label"]: int(row["n"]) for row in rows}


def load(
    ledger: Ledger, sessions: Sessions, pinecone: PineconeStore, driver: Driver, tmp: Path
) -> tuple[LoadReport, RenderedReport]:
    linker = GraphLinker(driver)
    loaded = LedgerLoader(sessions, linker).load(ledger.lines)
    docs = load_rendered(
        ledger.rendered,
        loaded,
        sessions=sessions,
        pinecone=pinecone,
        linker=linker,
        embedder=ChunkEmbedder(DenseEmbedder(model=FakeEncoder()), BM25Encoder()),
        bm25_path=tmp / "bm25.json",
    )
    return loaded, docs


def grown(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {label: after.get(label, 0) - before.get(label, 0) for label in after}


# --------------------------------------------------------------------------- tests


def test_counts_match_across_all_three_stores(
    ledger: Ledger, sessions: Sessions, pinecone: PineconeStore, driver: Driver, tmp_path: Path
) -> None:
    before = graph_nodes(driver)

    loaded, docs = load(ledger, sessions, pinecone, driver, tmp_path)

    assert loaded.records == {"contract": 1, "purchase_order": 1, "invoice": 4, "payment": 1}
    assert (loaded.vendors, loaded.new_vendors) == (2, 1)
    assert (docs.ready, docs.failed, docs.duplicates) == (4, {}, 0)
    counts = count_stores(sessions, pinecone, driver)
    local = StoreCounts(counts.postgres, counts.pinecone, grown(before, graph_nodes(driver)))
    assert local.mismatches() == []
    assert {table: counts.postgres[table] for table, _ in PARITY} == {
        "vendors": 2,
        "contracts": 1,
        "purchase_orders": 1,
        "invoices": 4,
        "payments": 1,
    }
    assert counts.postgres["documents"] == 4
    assert counts.pinecone == counts.postgres["chunks"] == docs.chunks > 0


def test_rendered_documents_attach_to_the_loaded_records(
    ledger: Ledger, sessions: Sessions, pinecone: PineconeStore, driver: Driver, tmp_path: Path
) -> None:
    load(ledger, sessions, pinecone, driver, tmp_path)
    p = ledger.prefix

    with sessions() as session:
        invoices = {row.id: row for row in session.scalars(select(InvoiceRow))}
        payment = session.get(PaymentRow, f"{p}PAY")
    assert set(invoices) == {f"{p}INV", f"{p}X1", f"{p}X2", f"{p}X3"}  # no second copy
    assert invoices[f"{p}INV"].document_id is not None  # the PDF is attached to the record
    assert invoices[f"{p}X1"].document_id is None  # structured only
    assert invoices[f"{p}INV"].status == "paid"  # the payment matched it by number
    assert invoices[f"{p}X3"].vendor_id == new_vendor_id(ledger.stranger)
    assert payment is not None and payment.document_id is not None

    nodes = run_read(
        driver,
        "MATCH (n) WHERE n.id STARTS WITH $p AND n.document_id IS NOT NULL RETURN n.id AS id",
        {"p": p},
    )
    assert sorted(row["id"] for row in nodes) == sorted(r.record_id for r in ledger.rendered)
    vendor = run_read(
        driver, "MATCH (v:Vendor {id: $id}) RETURN v.aliases AS aliases", {"id": ledger.vendor_id}
    )
    assert vendor == [{"aliases": ["ACME Supplies"]}]
    hits = pinecone.query([1.0] * DIM, top_k=50, filter={"vendor_id": ledger.vendor_id})
    assert {h.metadata.get("doc_number") for h in hits} == {
        "MSA-2026-07",
        "PO-88231",
        "INV-2026-0417",
        "RA-5531",
    }


def test_loading_again_changes_nothing(
    ledger: Ledger, sessions: Sessions, pinecone: PineconeStore, driver: Driver, tmp_path: Path
) -> None:
    load(ledger, sessions, pinecone, driver, tmp_path)
    first = count_stores(sessions, pinecone, driver)

    _, again = load(ledger, sessions, pinecone, driver, tmp_path)

    assert again.duplicates == 4 and again.ready == 0
    assert count_stores(sessions, pinecone, driver) == first
