"""The documents and anomalies stores behind the API, against Postgres (throwaway database).

docker compose up -d --wait postgres
pytest -m integration tests/integration/test_api_documents.py
"""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import update
from sqlalchemy.orm import Session, sessionmaker

from ledgerlens.api.routers.anomalies import AnomalyStore
from ledgerlens.api.routers.documents import DocumentStore
from ledgerlens.core.schemas import DocType, Document, DocumentStatus, Invoice, LineItem, Receipt
from ledgerlens.storage.postgres.db import session_scope
from ledgerlens.storage.postgres.models import AnomalyScoreRow
from ledgerlens.storage.postgres.repositories import Repositories

pytestmark = pytest.mark.integration

T0 = datetime(2026, 11, 1, tzinfo=UTC)
DOCS = [f"{n}1c2a8e-4b7d-4c1e-9a3f-0d2b5e7c9a10" for n in ("6f", "7f", "8f")]
RULE = [{"source": "rule", "code": "unknown_vendor", "text": "not in the vendor master"}]


def invoice(number: str, total: str) -> Invoice:
    return Invoice(
        invoice_number=number,
        vendor_name="Acme Supplies Ltd",
        invoice_date=date(2026, 3, 1),
        currency="GBP",
        total=Decimal(total),
        line_items=[LineItem(description="Widgets", amount=Decimal(total))],
    )


@pytest.fixture
def ledger(sessions: sessionmaker[Session]) -> sessionmaker[Session]:
    with session_scope(sessions) as session:
        repos = Repositories(session)
        repos.vendors.upsert("V001", "Acme Supplies Ltd")
        for i, (doc_id, doc_type) in enumerate(
            zip(DOCS, (DocType.INVOICE, DocType.RECEIPT, DocType.UNKNOWN), strict=True)
        ):
            status = DocumentStatus.QUEUED if doc_type is DocType.UNKNOWN else DocumentStatus.READY
            repos.documents.upsert(
                Document(
                    id=doc_id,
                    sha256=f"{i}" * 64,
                    filename=f"doc{i}.pdf",
                    doc_type=doc_type,
                    status=status,
                    created_at=T0 + timedelta(days=i),
                )
            )
        repos.invoices.upsert(
            invoice("INV-1", "100.00"), id="I-1", document_id=DOCS[0], vendor_id="V001"
        )
        repos.invoices.upsert_receipt(
            Receipt(
                vendor_name="Acme Supplies Ltd",
                receipt_number="R-9",
                receipt_date=date(2026, 3, 2),
                currency="GBP",
                total=Decimal("9.50"),
            ),
            id="I-2",
            document_id=DOCS[1],
        )
        repos.invoices.upsert(invoice("INV-3", "50.00"), id="I-3", vendor_id="V001")
        scores = repos.anomaly_scores
        scores.upsert("I-1", "ae-old", 0.10, [])
        scores.upsert("I-1", "ae-new", 0.80, RULE)
        scores.upsert("I-2", "ae-new", 0.30, [])
        scores.upsert("I-3", "ae-new", 0.05, [])
        session.flush()
        session.execute(  # the newer version was scored later
            update(AnomalyScoreRow)
            .where(AnomalyScoreRow.model_version == "ae-old")
            .values(created_at=T0)
        )
    return sessions


def test_documents_are_listed_newest_first_and_filtered(ledger: sessionmaker[Session]) -> None:
    store = DocumentStore(ledger)
    assert [d.id for d in store.list()] == DOCS[::-1]
    assert [d.id for d in store.list(status=DocumentStatus.QUEUED)] == [DOCS[2]]
    assert [d.id for d in store.list(doc_type=DocType.INVOICE)] == [DOCS[0]]
    assert [d.id for d in store.list(limit=1, offset=1)] == [DOCS[1]]


def test_detail_carries_the_extracted_record(ledger: sessionmaker[Session]) -> None:
    store = DocumentStore(ledger)
    detail = store.detail(DOCS[0])
    assert detail is not None and detail.record is not None
    assert (detail.record_type, detail.record_id) == ("invoice", "I-1")
    assert detail.record["total"] == "100.0000"
    assert detail.record["line_items"][0]["description"] == "Widgets"
    receipt = store.detail(DOCS[1])
    assert receipt is not None and receipt.record_type == "receipt"
    assert receipt.record is not None and receipt.record["receipt_number"] == "R-9"
    queued = store.detail(DOCS[2])
    assert queued is not None and queued.record is None and queued.chunks == 0
    assert store.detail("00000000-0000-4000-8000-000000000000") is None


def test_flags_use_each_invoices_newest_score(ledger: sessionmaker[Session]) -> None:
    store = AnomalyStore(ledger)
    flags = store.list()
    assert [(f.invoice_id, f.model_version, f.score) for f in flags] == [
        ("I-1", "ae-new", 0.80),
        ("I-2", "ae-new", 0.30),
    ]
    assert flags[0].reasons == RULE and flags[0].document_id == DOCS[0]
    assert [f.invoice_id for f in store.list(min_score=0)] == ["I-1", "I-2", "I-3"]
    old = store.list(model_version="ae-old", min_score=0)
    assert [(f.invoice_id, f.score) for f in old] == [("I-1", 0.10)]
    assert [f.invoice_id for f in store.list(vendor_id="V001", min_score=0)] == ["I-1", "I-3"]


def test_reviews_are_stored_filtered_and_survive_rescoring(
    ledger: sessionmaker[Session],
) -> None:
    store = AnomalyStore(ledger)
    first, second = store.list()
    reviewed = store.review(first.id, "false_positive", "known vendor, new entity")
    assert reviewed is not None and reviewed.review == "false_positive"
    assert reviewed.reviewed_at is not None
    assert [f.id for f in store.list()] == [second.id]  # unreviewed by default
    assert [f.id for f in store.list(review="reviewed")] == [first.id]
    assert [f.id for f in store.list(review="confirmed")] == []
    assert {f.id for f in store.list(review="all")} == {first.id, second.id}

    with session_scope(ledger) as session:  # same model version, scored again
        Repositories(session).anomaly_scores.upsert("I-1", "ae-new", 0.85, RULE)
    (again,) = store.list(review="false_positive")
    assert again.score == 0.85 and again.review_note == "known vendor, new entity"
    assert store.review(10_000, "confirmed", None) is None
