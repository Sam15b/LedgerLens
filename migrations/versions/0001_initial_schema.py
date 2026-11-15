"""Initial schema: every table in docs/layers/02-storage.md.

Documents and chunks; vendors; invoices, purchase orders, payments (+ allocations) and
contracts with their line items; anomaly scores, reminders, LLM call costs and eval runs.

Revision ID: 0001
Revises:
Create Date: 2026-11-15 12:19:49.416099+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("mime_type", sa.String(length=100), nullable=True),
        sa.Column("doc_type", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("pages", sa.Integer(), nullable=True),
        sa.Column("ocr_engine", sa.String(length=20), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "doc_type IN ('invoice', 'receipt', 'purchase_order', 'payment', 'contract', "
            "'unknown')",
            name=op.f("ck_documents_doc_type"),
        ),
        sa.CheckConstraint(
            "status <> 'failed' OR error IS NOT NULL", name=op.f("ck_documents_failed_has_error")
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'processing', 'ready', 'failed')",
            name=op.f("ck_documents_status"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_documents")),
        sa.UniqueConstraint("sha256", name=op.f("uq_documents_sha256")),
    )
    op.create_table(
        "eval_runs",
        sa.Column("id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("suite", sa.String(length=64), nullable=False),
        sa.Column("git_sha", sa.String(length=40), nullable=True),
        sa.Column(
            "config",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "metrics",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_eval_runs")),
    )
    op.create_index(op.f("ix_eval_runs_created_at"), "eval_runs", ["created_at"], unique=False)
    op.create_table(
        "llm_calls",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=True),
        sa.Column("alias", sa.String(length=64), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("tokens_in", sa.Integer(), nullable=False),
        sa.Column("tokens_out", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("streamed", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_llm_calls")),
    )
    op.create_index("ix_llm_calls_provider_model", "llm_calls", ["provider", "model"], unique=False)
    op.create_index(op.f("ix_llm_calls_request_id"), "llm_calls", ["request_id"], unique=False)
    op.create_index(op.f("ix_llm_calls_ts"), "llm_calls", ["ts"], unique=False)
    op.create_table(
        "vendors",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("canonical_name", sa.Text(), nullable=False),
        sa.Column(
            "aliases", postgresql.ARRAY(sa.Text()), server_default=sa.text("'{}'"), nullable=False
        ),
        sa.Column("tax_id", sa.String(length=64), nullable=True),
        sa.Column("email", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_vendors")),
        sa.UniqueConstraint("tax_id", name=op.f("uq_vendors_tax_id")),
    )
    op.create_index(op.f("ix_vendors_canonical_name"), "vendors", ["canonical_name"], unique=False)
    op.create_table(
        "chunks",
        sa.Column("id", sa.String(length=80), nullable=False),
        sa.Column("document_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=20), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column(
            "pages", postgresql.ARRAY(sa.Integer()), server_default=sa.text("'{}'"), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_chunks_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chunks")),
        sa.UniqueConstraint(
            "document_id", "chunk_index", name=op.f("uq_chunks_document_id_chunk_index")
        ),
    )
    op.create_table(
        "contracts",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("document_id", sa.Uuid(as_uuid=False), nullable=True),
        sa.Column("vendor_id", sa.String(length=64), nullable=True),
        sa.Column("vendor_name", sa.Text(), nullable=False),
        sa.Column("customer_name", sa.Text(), nullable=True),
        sa.Column("contract_number", sa.String(length=64), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=True),
        sa.Column("payment_terms_days", sa.Integer(), nullable=True),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column(
            "rate_card",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("payment_terms_days >= 0", name=op.f("ck_contracts_payment_terms_days")),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_contracts_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["vendor_id"],
            ["vendors.id"],
            name=op.f("fk_contracts_vendor_id_vendors"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_contracts")),
        sa.UniqueConstraint("document_id", name=op.f("uq_contracts_document_id")),
    )
    op.create_index(
        op.f("ix_contracts_contract_number"), "contracts", ["contract_number"], unique=False
    )
    op.create_index(op.f("ix_contracts_vendor_id"), "contracts", ["vendor_id"], unique=False)
    op.create_table(
        "invoices",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("document_id", sa.Uuid(as_uuid=False), nullable=True),
        sa.Column("vendor_id", sa.String(length=64), nullable=True),
        sa.Column("vendor_name", sa.Text(), nullable=False),
        sa.Column("vendor_tax_id", sa.String(length=64), nullable=True),
        sa.Column("customer_name", sa.Text(), nullable=True),
        sa.Column("invoice_number", sa.String(length=64), nullable=False),
        sa.Column("invoice_date", sa.Date(), nullable=False),
        sa.Column("due_date", sa.Date(), nullable=True),
        sa.Column("po_number", sa.String(length=64), nullable=True),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("subtotal", sa.Numeric(precision=18, scale=4), nullable=True),
        sa.Column("tax", sa.Numeric(precision=18, scale=4), nullable=True),
        sa.Column("total", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('unpaid', 'partially_paid', 'paid', 'disputed', 'void')",
            name=op.f("ck_invoices_status"),
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_invoices_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["vendor_id"],
            ["vendors.id"],
            name=op.f("fk_invoices_vendor_id_vendors"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_invoices")),
        sa.UniqueConstraint("document_id", name=op.f("uq_invoices_document_id")),
    )
    op.create_index(
        op.f("ix_invoices_invoice_number"), "invoices", ["invoice_number"], unique=False
    )
    op.create_index(op.f("ix_invoices_po_number"), "invoices", ["po_number"], unique=False)
    op.create_index("ix_invoices_status_due_date", "invoices", ["status", "due_date"], unique=False)
    op.create_index(op.f("ix_invoices_vendor_id"), "invoices", ["vendor_id"], unique=False)
    op.create_table(
        "payments",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("document_id", sa.Uuid(as_uuid=False), nullable=True),
        sa.Column("vendor_id", sa.String(length=64), nullable=True),
        sa.Column("vendor_name", sa.Text(), nullable=False),
        sa.Column("payer_name", sa.Text(), nullable=True),
        sa.Column("payment_reference", sa.String(length=64), nullable=False),
        sa.Column("paid_at", sa.Date(), nullable=False),
        sa.Column("method", sa.String(length=20), nullable=True),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("amount", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "method IN ('bank_transfer', 'card', 'check', 'direct_debit', 'cash', 'other')",
            name=op.f("ck_payments_method"),
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_payments_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["vendor_id"],
            ["vendors.id"],
            name=op.f("fk_payments_vendor_id_vendors"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_payments")),
        sa.UniqueConstraint("document_id", name=op.f("uq_payments_document_id")),
    )
    op.create_index(op.f("ix_payments_paid_at"), "payments", ["paid_at"], unique=False)
    op.create_index(
        op.f("ix_payments_payment_reference"), "payments", ["payment_reference"], unique=False
    )
    op.create_index(op.f("ix_payments_vendor_id"), "payments", ["vendor_id"], unique=False)
    op.create_table(
        "anomaly_scores",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("invoice_id", sa.String(length=64), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column(
            "reasons",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["invoice_id"],
            ["invoices.id"],
            name=op.f("fk_anomaly_scores_invoice_id_invoices"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_anomaly_scores")),
        sa.UniqueConstraint(
            "invoice_id", "model_version", name=op.f("uq_anomaly_scores_invoice_id_model_version")
        ),
    )
    op.create_table(
        "invoice_line_items",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("invoice_id", sa.String(length=64), nullable=False),
        sa.Column("line_no", sa.Integer(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("quantity", sa.Numeric(precision=18, scale=4), nullable=True),
        sa.Column("unit_price", sa.Numeric(precision=18, scale=4), nullable=True),
        sa.Column("amount", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.ForeignKeyConstraint(
            ["invoice_id"],
            ["invoices.id"],
            name=op.f("fk_invoice_line_items_invoice_id_invoices"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_invoice_line_items")),
        sa.UniqueConstraint(
            "invoice_id", "line_no", name=op.f("uq_invoice_line_items_invoice_id_line_no")
        ),
    )
    op.create_table(
        "payment_allocations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("payment_id", sa.String(length=64), nullable=False),
        sa.Column("invoice_number", sa.String(length=64), nullable=False),
        sa.Column("invoice_id", sa.String(length=64), nullable=True),
        sa.Column("amount", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.ForeignKeyConstraint(
            ["invoice_id"],
            ["invoices.id"],
            name=op.f("fk_payment_allocations_invoice_id_invoices"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["payment_id"],
            ["payments.id"],
            name=op.f("fk_payment_allocations_payment_id_payments"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_payment_allocations")),
        sa.UniqueConstraint(
            "payment_id",
            "invoice_number",
            name=op.f("uq_payment_allocations_payment_id_invoice_number"),
        ),
    )
    op.create_index(
        op.f("ix_payment_allocations_invoice_id"),
        "payment_allocations",
        ["invoice_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_payment_allocations_invoice_number"),
        "payment_allocations",
        ["invoice_number"],
        unique=False,
    )
    op.create_table(
        "purchase_orders",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("document_id", sa.Uuid(as_uuid=False), nullable=True),
        sa.Column("vendor_id", sa.String(length=64), nullable=True),
        sa.Column("vendor_name", sa.Text(), nullable=False),
        sa.Column("vendor_tax_id", sa.String(length=64), nullable=True),
        sa.Column("buyer_name", sa.Text(), nullable=True),
        sa.Column("po_number", sa.String(length=64), nullable=False),
        sa.Column("order_date", sa.Date(), nullable=False),
        sa.Column("delivery_date", sa.Date(), nullable=True),
        sa.Column("contract_id", sa.String(length=64), nullable=True),
        sa.Column("contract_number", sa.String(length=64), nullable=True),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("subtotal", sa.Numeric(precision=18, scale=4), nullable=True),
        sa.Column("tax", sa.Numeric(precision=18, scale=4), nullable=True),
        sa.Column("total", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["contract_id"],
            ["contracts.id"],
            name=op.f("fk_purchase_orders_contract_id_contracts"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_purchase_orders_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["vendor_id"],
            ["vendors.id"],
            name=op.f("fk_purchase_orders_vendor_id_vendors"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_purchase_orders")),
        sa.UniqueConstraint("document_id", name=op.f("uq_purchase_orders_document_id")),
    )
    op.create_index(
        op.f("ix_purchase_orders_contract_id"), "purchase_orders", ["contract_id"], unique=False
    )
    op.create_index(
        op.f("ix_purchase_orders_po_number"), "purchase_orders", ["po_number"], unique=False
    )
    op.create_index(
        op.f("ix_purchase_orders_vendor_id"), "purchase_orders", ["vendor_id"], unique=False
    )
    op.create_table(
        "reminders",
        sa.Column("id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("invoice_id", sa.String(length=64), nullable=False),
        sa.Column("draft", sa.Text(), nullable=False),
        sa.Column("tone", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('pending_approval', 'approved', 'rejected', 'sent')",
            name=op.f("ck_reminders_status"),
        ),
        sa.ForeignKeyConstraint(
            ["invoice_id"],
            ["invoices.id"],
            name=op.f("fk_reminders_invoice_id_invoices"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_reminders")),
    )
    op.create_index(op.f("ix_reminders_invoice_id"), "reminders", ["invoice_id"], unique=False)
    op.create_index(op.f("ix_reminders_status"), "reminders", ["status"], unique=False)
    op.create_table(
        "po_line_items",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("purchase_order_id", sa.String(length=64), nullable=False),
        sa.Column("line_no", sa.Integer(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("quantity", sa.Numeric(precision=18, scale=4), nullable=True),
        sa.Column("unit_price", sa.Numeric(precision=18, scale=4), nullable=True),
        sa.Column("amount", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.ForeignKeyConstraint(
            ["purchase_order_id"],
            ["purchase_orders.id"],
            name=op.f("fk_po_line_items_purchase_order_id_purchase_orders"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_po_line_items")),
        sa.UniqueConstraint(
            "purchase_order_id", "line_no", name=op.f("uq_po_line_items_purchase_order_id_line_no")
        ),
    )


def downgrade() -> None:
    op.drop_table("po_line_items")
    op.drop_index(op.f("ix_reminders_status"), table_name="reminders")
    op.drop_index(op.f("ix_reminders_invoice_id"), table_name="reminders")
    op.drop_table("reminders")
    op.drop_index(op.f("ix_purchase_orders_vendor_id"), table_name="purchase_orders")
    op.drop_index(op.f("ix_purchase_orders_po_number"), table_name="purchase_orders")
    op.drop_index(op.f("ix_purchase_orders_contract_id"), table_name="purchase_orders")
    op.drop_table("purchase_orders")
    op.drop_index(op.f("ix_payment_allocations_invoice_number"), table_name="payment_allocations")
    op.drop_index(op.f("ix_payment_allocations_invoice_id"), table_name="payment_allocations")
    op.drop_table("payment_allocations")
    op.drop_table("invoice_line_items")
    op.drop_table("anomaly_scores")
    op.drop_index(op.f("ix_payments_vendor_id"), table_name="payments")
    op.drop_index(op.f("ix_payments_payment_reference"), table_name="payments")
    op.drop_index(op.f("ix_payments_paid_at"), table_name="payments")
    op.drop_table("payments")
    op.drop_index(op.f("ix_invoices_vendor_id"), table_name="invoices")
    op.drop_index("ix_invoices_status_due_date", table_name="invoices")
    op.drop_index(op.f("ix_invoices_po_number"), table_name="invoices")
    op.drop_index(op.f("ix_invoices_invoice_number"), table_name="invoices")
    op.drop_table("invoices")
    op.drop_index(op.f("ix_contracts_vendor_id"), table_name="contracts")
    op.drop_index(op.f("ix_contracts_contract_number"), table_name="contracts")
    op.drop_table("contracts")
    op.drop_table("chunks")
    op.drop_index(op.f("ix_vendors_canonical_name"), table_name="vendors")
    op.drop_table("vendors")
    op.drop_index(op.f("ix_llm_calls_ts"), table_name="llm_calls")
    op.drop_index(op.f("ix_llm_calls_request_id"), table_name="llm_calls")
    op.drop_index("ix_llm_calls_provider_model", table_name="llm_calls")
    op.drop_table("llm_calls")
    op.drop_index(op.f("ix_eval_runs_created_at"), table_name="eval_runs")
    op.drop_table("eval_runs")
    op.drop_table("documents")
