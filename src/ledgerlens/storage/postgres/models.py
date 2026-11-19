"""SQLAlchemy ORM models: the PostgreSQL system of record.

Tables follow docs/layers/02-storage.md. Conventions:

* ``documents.id`` is the ``document_id`` shared with Pinecone vector ids and Neo4j nodes.
  ``chunks.id`` is the chunk's Pinecone vector id (``{document_id}#c{n}``).
* Business records (invoices, ...) have a text ``id`` chosen by the writer -- the
  ``document_id`` for a record extracted from a document, the record's own id for imported
  or synthetic data -- so a re-run writes the same key and an upsert is a no-op. A record
  extracted from a document points back with a unique ``document_id``; deleting the
  document deletes the record.
* ``vendor_id`` is set once the vendor is resolved; ``vendor_name`` keeps the name as it
  appeared on the document either way.
* Amounts are ``NUMERIC(18, 4)`` (``Decimal`` in Python): exact, and wide enough for
  three-decimal currencies and four-decimal unit prices.
* Enumerations are ``VARCHAR`` + ``CHECK``, not native Postgres enums, so adding a value is
  a one-line migration instead of an ``ALTER TYPE``.
* Constraint and index names come from :data:`NAMING_CONVENTION`, so Alembic migrations
  name them the same way on every database.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from datetime import date, datetime
from decimal import Decimal
from typing import Any, get_args

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy import text as sql_text  # a column below is called ``text``
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from ledgerlens.core.schemas import DocType, DocumentStatus
from ledgerlens.core.schemas.payment import PaymentMethod

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

MONEY = Numeric(18, 4)
RECORD_ID = String(64)
INVOICE_STATUSES = ("unpaid", "partially_paid", "paid", "disputed", "void")
REMINDER_STATUSES = ("pending_approval", "approved", "rejected", "sent")
PAYMENT_METHODS: tuple[str, ...] = get_args(PaymentMethod)


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def one_of(column: str, values: Iterable[str]) -> CheckConstraint:
    """``CHECK (column IN (...))`` named ``ck_<table>_<column>``."""
    allowed = ", ".join(f"'{value}'" for value in values)
    return CheckConstraint(f"{column} IN ({allowed})", name=column)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _created_at() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now())


def _updated_at() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


def _document_fk() -> Mapped[str | None]:
    return mapped_column(
        Uuid(as_uuid=False), ForeignKey("documents.id", ondelete="CASCADE"), unique=True
    )


def _vendor_fk() -> Mapped[str | None]:
    return mapped_column(RECORD_ID, ForeignKey("vendors.id", ondelete="SET NULL"), index=True)


# --------------------------------------------------------------------------- documents


class DocumentRow(Base):
    """An uploaded file and its pipeline status (mirrors ``core.schemas.Document``)."""

    __tablename__ = "documents"
    __table_args__ = (
        one_of("doc_type", (t.value for t in DocType)),
        one_of("status", (s.value for s in DocumentStatus)),
        CheckConstraint("status <> 'failed' OR error IS NOT NULL", name="failed_has_error"),
    )

    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), unique=True)
    """Content hash; the unique index is what makes duplicate uploads a no-op."""
    filename: Mapped[str] = mapped_column(Text)
    mime_type: Mapped[str | None] = mapped_column(String(100))
    doc_type: Mapped[str] = mapped_column(String(20), default=DocType.UNKNOWN.value)
    status: Mapped[str] = mapped_column(String(20), default=DocumentStatus.QUEUED.value)
    error: Mapped[str | None] = mapped_column(Text)
    pages: Mapped[int | None] = mapped_column(Integer)
    ocr_engine: Mapped[str | None] = mapped_column(String(20))
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()

    chunks: Mapped[list[ChunkRow]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ChunkRow.chunk_index",
    )


class ChunkRow(Base):
    """A retrieval chunk; its full text lives here, Pinecone keeps a truncated copy."""

    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("document_id", "chunk_index"),)

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    """The Pinecone vector id, ``{document_id}#c{chunk_index}``."""
    document_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False), ForeignKey("documents.id", ondelete="CASCADE")
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(20))
    title: Mapped[str | None] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int | None] = mapped_column(Integer)
    pages: Mapped[list[int]] = mapped_column(ARRAY(Integer), server_default=sql_text("'{}'"))

    document: Mapped[DocumentRow] = relationship(back_populates="chunks")


# --------------------------------------------------------------------------- vendors


class VendorRow(Base):
    __tablename__ = "vendors"

    id: Mapped[str] = mapped_column(RECORD_ID, primary_key=True)
    canonical_name: Mapped[str] = mapped_column(Text, index=True)
    aliases: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default=sql_text("'{}'"))
    """Other spellings seen on documents; fuzzy-matched when resolving a vendor."""
    tax_id: Mapped[str | None] = mapped_column(String(64), unique=True)
    email: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()


# --------------------------------------------------------------------------- invoices


class InvoiceRow(Base):
    __tablename__ = "invoices"
    __table_args__ = (
        one_of("status", INVOICE_STATUSES),
        Index("ix_invoices_status_due_date", "status", "due_date"),  # overdue lists
    )

    id: Mapped[str] = mapped_column(RECORD_ID, primary_key=True)
    document_id: Mapped[str | None] = _document_fk()
    vendor_id: Mapped[str | None] = _vendor_fk()
    vendor_name: Mapped[str] = mapped_column(Text)
    vendor_tax_id: Mapped[str | None] = mapped_column(String(64))
    customer_name: Mapped[str | None] = mapped_column(Text)
    invoice_number: Mapped[str] = mapped_column(String(64), index=True)
    invoice_date: Mapped[date] = mapped_column(Date)
    due_date: Mapped[date | None] = mapped_column(Date)
    po_number: Mapped[str | None] = mapped_column(String(64), index=True)
    """The PO this invoice bills against, as printed (linked by number, not by key)."""
    currency: Mapped[str] = mapped_column(String(3))
    subtotal: Mapped[Decimal | None] = mapped_column(MONEY)
    tax: Mapped[Decimal | None] = mapped_column(MONEY)
    total: Mapped[Decimal] = mapped_column(MONEY)
    status: Mapped[str] = mapped_column(String(20), default="unpaid")
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()

    vendor: Mapped[VendorRow | None] = relationship()
    line_items: Mapped[list[InvoiceLineItemRow]] = relationship(
        back_populates="invoice",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="InvoiceLineItemRow.line_no",
    )


class InvoiceLineItemRow(Base):
    __tablename__ = "invoice_line_items"
    __table_args__ = (UniqueConstraint("invoice_id", "line_no"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    invoice_id: Mapped[str] = mapped_column(
        RECORD_ID, ForeignKey("invoices.id", ondelete="CASCADE")
    )
    line_no: Mapped[int] = mapped_column(Integer)
    """Position on the document, from 0."""
    description: Mapped[str] = mapped_column(Text)
    quantity: Mapped[Decimal | None] = mapped_column(MONEY)
    unit_price: Mapped[Decimal | None] = mapped_column(MONEY)
    amount: Mapped[Decimal] = mapped_column(MONEY)

    invoice: Mapped[InvoiceRow] = relationship(back_populates="line_items")


# --------------------------------------------------------------------------- contracts


class ContractRow(Base):
    __tablename__ = "contracts"
    __table_args__ = (CheckConstraint("payment_terms_days >= 0", name="payment_terms_days"),)

    id: Mapped[str] = mapped_column(RECORD_ID, primary_key=True)
    document_id: Mapped[str | None] = _document_fk()
    vendor_id: Mapped[str | None] = _vendor_fk()
    vendor_name: Mapped[str] = mapped_column(Text)
    customer_name: Mapped[str | None] = mapped_column(Text)
    contract_number: Mapped[str | None] = mapped_column(String(64), index=True)
    title: Mapped[str | None] = mapped_column(Text)
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date | None] = mapped_column(Date)
    """``NULL`` for an open-ended contract."""
    payment_terms_days: Mapped[int | None] = mapped_column(Integer)
    currency: Mapped[str] = mapped_column(String(3))
    rate_card: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, server_default=sql_text("'[]'::jsonb")
    )
    """``[{"item", "unit", "unit_price"}, ...]`` -- the prices invoices are checked against."""
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()


# --------------------------------------------------------------------------- purchase orders


class PurchaseOrderRow(Base):
    __tablename__ = "purchase_orders"

    id: Mapped[str] = mapped_column(RECORD_ID, primary_key=True)
    document_id: Mapped[str | None] = _document_fk()
    vendor_id: Mapped[str | None] = _vendor_fk()
    vendor_name: Mapped[str] = mapped_column(Text)
    vendor_tax_id: Mapped[str | None] = mapped_column(String(64))
    buyer_name: Mapped[str | None] = mapped_column(Text)
    po_number: Mapped[str] = mapped_column(String(64), index=True)
    order_date: Mapped[date] = mapped_column(Date)
    delivery_date: Mapped[date | None] = mapped_column(Date)
    contract_id: Mapped[str | None] = mapped_column(
        RECORD_ID, ForeignKey("contracts.id", ondelete="SET NULL"), index=True
    )
    contract_number: Mapped[str | None] = mapped_column(String(64))
    """The governing contract as printed on the PO; ``contract_id`` once resolved."""
    currency: Mapped[str] = mapped_column(String(3))
    subtotal: Mapped[Decimal | None] = mapped_column(MONEY)
    tax: Mapped[Decimal | None] = mapped_column(MONEY)
    total: Mapped[Decimal] = mapped_column(MONEY)
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()

    vendor: Mapped[VendorRow | None] = relationship()
    contract: Mapped[ContractRow | None] = relationship()
    line_items: Mapped[list[POLineItemRow]] = relationship(
        back_populates="purchase_order",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="POLineItemRow.line_no",
    )


class POLineItemRow(Base):
    __tablename__ = "po_line_items"
    __table_args__ = (UniqueConstraint("purchase_order_id", "line_no"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    purchase_order_id: Mapped[str] = mapped_column(
        RECORD_ID, ForeignKey("purchase_orders.id", ondelete="CASCADE")
    )
    line_no: Mapped[int] = mapped_column(Integer)
    description: Mapped[str] = mapped_column(Text)
    quantity: Mapped[Decimal | None] = mapped_column(MONEY)
    unit_price: Mapped[Decimal | None] = mapped_column(MONEY)
    amount: Mapped[Decimal] = mapped_column(MONEY)

    purchase_order: Mapped[PurchaseOrderRow] = relationship(back_populates="line_items")


# --------------------------------------------------------------------------- payments


class PaymentRow(Base):
    __tablename__ = "payments"
    __table_args__ = (one_of("method", PAYMENT_METHODS),)

    id: Mapped[str] = mapped_column(RECORD_ID, primary_key=True)
    document_id: Mapped[str | None] = _document_fk()
    vendor_id: Mapped[str | None] = _vendor_fk()
    vendor_name: Mapped[str] = mapped_column(Text)
    """Who was paid."""
    payer_name: Mapped[str | None] = mapped_column(Text)
    payment_reference: Mapped[str] = mapped_column(String(64), index=True)
    paid_at: Mapped[date] = mapped_column(Date, index=True)
    method: Mapped[str | None] = mapped_column(String(20))
    currency: Mapped[str] = mapped_column(String(3))
    amount: Mapped[Decimal] = mapped_column(MONEY)
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()

    vendor: Mapped[VendorRow | None] = relationship()
    allocations: Mapped[list[PaymentAllocationRow]] = relationship(
        back_populates="payment",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="PaymentAllocationRow.id",
    )


class PaymentAllocationRow(Base):
    """The part of a payment applied to one invoice (a payment can settle several)."""

    __tablename__ = "payment_allocations"
    __table_args__ = (UniqueConstraint("payment_id", "invoice_number"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    payment_id: Mapped[str] = mapped_column(
        RECORD_ID, ForeignKey("payments.id", ondelete="CASCADE")
    )
    invoice_number: Mapped[str] = mapped_column(String(64), index=True)
    """As printed on the remittance advice."""
    invoice_id: Mapped[str | None] = mapped_column(
        RECORD_ID, ForeignKey("invoices.id", ondelete="SET NULL"), index=True
    )
    """Set once the invoice number is matched to an invoice."""
    amount: Mapped[Decimal] = mapped_column(MONEY)

    payment: Mapped[PaymentRow] = relationship(back_populates="allocations")


# --------------------------------------------------------------------------- audit + workflow


class AnomalyScoreRow(Base):
    """One anomaly model's verdict on an invoice; re-scoring with a model version replaces it."""

    __tablename__ = "anomaly_scores"
    __table_args__ = (UniqueConstraint("invoice_id", "model_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    invoice_id: Mapped[str] = mapped_column(
        RECORD_ID, ForeignKey("invoices.id", ondelete="CASCADE")
    )
    model_version: Mapped[str] = mapped_column(String(64))
    score: Mapped[float] = mapped_column(Float)
    reasons: Mapped[list[Any]] = mapped_column(JSONB, server_default=sql_text("'[]'::jsonb"))
    created_at: Mapped[datetime] = _created_at()


class ReminderRow(Base):
    """A drafted payment reminder; nothing is sent until a human approves it."""

    __tablename__ = "reminders"
    __table_args__ = (one_of("status", REMINDER_STATUSES),)

    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True, default=_new_uuid)
    invoice_id: Mapped[str] = mapped_column(
        RECORD_ID, ForeignKey("invoices.id", ondelete="CASCADE"), index=True
    )
    draft: Mapped[str] = mapped_column(Text)
    tone: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20), default="pending_approval", index=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    thread_id: Mapped[str | None] = mapped_column(String(100), index=True)
    """The agent run (LangGraph thread) waiting for a decision on this draft."""
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()


class LLMCallRow(Base):
    """One completed LLM call (mirrors ``observability.cost.LLMCallRecord``)."""

    __tablename__ = "llm_calls"
    __table_args__ = (Index("ix_llm_calls_provider_model", "provider", "model"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    request_id: Mapped[str | None] = mapped_column(String(64), index=True)
    alias: Mapped[str] = mapped_column(String(64))
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(128))
    tokens_in: Mapped[int] = mapped_column(Integer)
    tokens_out: Mapped[int] = mapped_column(Integer)
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(18, 8))
    """``NULL`` when the model has no price in ``models.yaml``."""
    latency_ms: Mapped[float] = mapped_column(Float)
    streamed: Mapped[bool] = mapped_column(Boolean, default=False)


class EvalRunRow(Base):
    """One evaluation run: what was run (suite, config, git sha) and what it scored."""

    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True, default=_new_uuid)
    suite: Mapped[str] = mapped_column(String(64))
    git_sha: Mapped[str | None] = mapped_column(String(40))
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=sql_text("'{}'::jsonb"))
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=sql_text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
