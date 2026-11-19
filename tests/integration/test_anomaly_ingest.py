"""Anomaly scoring on ingest: every invoice the stores sink writes gets an
``anomaly_scores`` row with a score and reasons (throwaway Postgres database).

The sink runs without Pinecone and Neo4j, and resolves vendors to known master ids
instead of the graph (as the bulk loader does). The autoencoder is a stand-in, so the
test runs without Keras: it reconstructs every feature as the training mean.

    docker compose up -d --wait postgres
    pytest -m integration tests/integration/test_anomaly_ingest.py
"""

import hashlib
from datetime import date
from decimal import Decimal

import numpy as np
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from ledgerlens.anomaly.features import FEATURE_NAMES, Matrix, VendorHistory
from ledgerlens.anomaly.rules import InvoiceRef, RuleContext
from ledgerlens.anomaly.scorer import AnomalyScore, AnomalyScorer
from ledgerlens.core.schemas import (
    DocType,
    Document,
    DocumentStatus,
    ExtractedRecord,
    Invoice,
    LineItem,
    PurchaseOrder,
)
from ledgerlens.extraction.base import ExtractionResult
from ledgerlens.ingestion.pipeline import IngestResult
from ledgerlens.ingestion.sinks import StoresSink
from ledgerlens.storage.neo4j.linker import VendorMatch
from ledgerlens.storage.postgres.db import session_scope
from ledgerlens.storage.postgres.models import AnomalyScoreRow
from ledgerlens.storage.postgres.repositories import AnomalyScoreRepository, VendorRepository

pytestmark = pytest.mark.integration

Sessions = sessionmaker[Session]
VENDOR = "Acme Supplies Ltd"


class MeanModel:
    """Reconstructs every scaled feature as 0 (the training mean)."""

    version = "ae-test"
    threshold = 50.0  # high: only the rules flag these small invoices
    features = FEATURE_NAMES

    def reconstruct(self, x_raw: Matrix) -> tuple[Matrix, Matrix]:
        return x_raw, np.zeros_like(x_raw)


class MasterSink(StoresSink):
    """Every document's vendor is the master vendor V001."""

    def resolve_vendor(self, result: IngestResult, record: ExtractedRecord | None) -> VendorMatch:
        return VendorMatch("V001", "given", 100.0)


def ingested(record: Invoice | PurchaseOrder, name: str) -> IngestResult:
    doc_type = DocType.INVOICE if isinstance(record, Invoice) else DocType.PURCHASE_ORDER
    document = Document(
        sha256=hashlib.sha256(name.encode()).hexdigest(),
        filename=f"{name}.pdf",
        doc_type=doc_type,
    ).with_status(DocumentStatus.READY)
    extraction = ExtractionResult(extractor="test", doc_type=doc_type, record=record)
    return IngestResult(document=document, extraction=extraction)


def invoice(number: str, day: date, total: str = "100.00") -> Invoice:
    return Invoice(
        invoice_number=number,
        vendor_name=VENDOR,
        invoice_date=day,
        po_number="PO-77",
        currency="GBP",
        total=Decimal(total),
        line_items=[LineItem(description="Pallet wrap 500m", amount=Decimal(total))],
    )


@pytest.fixture
def sink(sessions: Sessions) -> StoresSink:
    with session_scope(sessions) as session:
        VendorRepository(session).upsert("V001", VENDOR, tax_id="GB123456789")
    return MasterSink(sessions=sessions, scorer=AnomalyScorer(MeanModel()))


def scores(sessions: Sessions, invoice_id: str) -> list[AnomalyScoreRow]:
    with sessions() as session:
        rows = AnomalyScoreRepository(session).for_invoice(invoice_id)
        session.expunge_all()
        return rows


def test_every_ingested_invoice_gets_a_score_with_reasons(
    sessions: Sessions, sink: StoresSink
) -> None:
    po = PurchaseOrder(
        po_number="PO-77",
        vendor_name=VENDOR,
        order_date=date(2026, 3, 1),
        currency="GBP",
        total=Decimal("100.00"),
    )
    assert sink.write(ingested(po, "po-77")).status is DocumentStatus.READY
    first = ingested(invoice("INV-500", date(2026, 3, 4)), "inv-500")
    copy = ingested(invoice("INV-500", date(2026, 3, 6)), "inv-500-again")
    assert sink.write(first).status is DocumentStatus.READY
    assert sink.write(copy).status is DocumentStatus.READY

    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(AnomalyScoreRow)) == 2

    (clean,) = scores(sessions, first.document.id)
    assert clean.model_version == "ae-test"
    assert clean.score < 0.25
    assert [r["code"] for r in clean.reasons] == ["nothing_unusual"]

    (dup,) = scores(sessions, copy.document.id)
    assert dup.score >= 0.5  # hard hits rank first
    by_code = {r["code"]: r for r in dup.reasons}
    assert set(by_code) == {"exact_duplicate", "over_po"}
    assert by_code["exact_duplicate"]["related_ids"] == [first.document.id]
    assert "already received" in by_code["exact_duplicate"]["text"]
    assert by_code["over_po"]["text"] == (
        "total 100.00 GBP is 100.00 over what is left on PO-77 (0.00 of 100.00; "
        "100.00 already billed)"
    )


def test_rewriting_an_invoice_replaces_its_score(sessions: Sessions, sink: StoresSink) -> None:
    result = ingested(invoice("INV-600", date(2026, 3, 9)), "inv-600")
    sink.write(result)
    sink.write(result)  # a retry or re-ingest of the same document
    assert len(scores(sessions, result.document.id)) == 1


def test_scoring_failure_does_not_fail_the_document(sessions: Sessions) -> None:
    class Broken(AnomalyScorer):
        def score(
            self, ref: InvoiceRef, context: RuleContext, history: VendorHistory | None = None
        ) -> AnomalyScore:
            raise RuntimeError("model exploded")

    with session_scope(sessions) as session:
        VendorRepository(session).upsert("V001", VENDOR)
    result = ingested(invoice("INV-700", date(2026, 3, 9)), "inv-700")
    stored = MasterSink(sessions=sessions, scorer=Broken(None)).write(result)
    assert stored.status is DocumentStatus.READY
    assert scores(sessions, result.document.id) == []
