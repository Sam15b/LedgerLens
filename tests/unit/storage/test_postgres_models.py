"""ORM model shape, checked without a database by compiling the DDL for PostgreSQL."""

from typing import get_args

import pytest
from sqlalchemy import Table
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable

from ledgerlens.core.schemas import DocType, DocumentStatus
from ledgerlens.core.schemas.payment import PaymentMethod
from ledgerlens.observability.cost import LLMCallRecord
from ledgerlens.storage.postgres.models import Base

TABLES = Base.metadata.tables


def ddl(table: Table) -> str:
    dialect = postgresql.dialect()  # type: ignore[no-untyped-call]
    statements = [CreateTable(table)] + [CreateIndex(ix) for ix in table.indexes]
    return "\n".join(str(s.compile(dialect=dialect)) for s in statements)


@pytest.mark.parametrize("name", sorted(TABLES))
def test_every_table_compiles_for_postgres(name: str) -> None:
    assert f"CREATE TABLE {name}" in ddl(TABLES[name])


def test_document_id_is_a_uuid_shared_with_chunks_and_records() -> None:
    assert "id UUID NOT NULL" in ddl(TABLES["documents"])
    assert "document_id UUID NOT NULL" in ddl(TABLES["chunks"])
    assert "document_id UUID," in ddl(TABLES["invoices"])  # nullable: imported records


def test_documents_are_deduplicated_by_sha256() -> None:
    sql = ddl(TABLES["documents"])

    assert "CONSTRAINT uq_documents_sha256 UNIQUE (sha256)" in sql


def test_document_enums_match_the_pydantic_schema() -> None:
    sql = ddl(TABLES["documents"])

    assert all(f"'{t.value}'" in sql for t in DocType)
    assert all(f"'{s.value}'" in sql for s in DocumentStatus)
    assert "CONSTRAINT ck_documents_failed_has_error" in sql


def test_chunk_index_is_unique_per_document() -> None:
    assert "UNIQUE (document_id, chunk_index)" in ddl(TABLES["chunks"])


def test_records_cascade_from_their_document_and_keep_unresolved_vendors() -> None:
    sql = ddl(TABLES["invoices"])

    assert "REFERENCES documents (id) ON DELETE CASCADE" in sql
    assert "REFERENCES vendors (id) ON DELETE SET NULL" in sql
    assert "vendor_name TEXT NOT NULL" in sql


def test_amounts_are_exact_decimals() -> None:
    sql = ddl(TABLES["invoices"])

    assert "total NUMERIC(18, 4) NOT NULL" in sql
    assert "subtotal NUMERIC(18, 4)," in sql


def test_overdue_lookups_are_indexed() -> None:
    assert "ix_invoices_status_due_date ON invoices (status, due_date)" in ddl(TABLES["invoices"])


def test_constraint_names_follow_the_naming_convention() -> None:
    for table in TABLES.values():
        for constraint in table.constraints:
            assert constraint.name, f"unnamed constraint on {table.name}"
            assert str(constraint.name).split("_", 1)[0] in {"pk", "fk", "uq", "ck"}


def test_every_planned_table_exists() -> None:
    planned = {
        "documents",
        "chunks",
        "vendors",
        "invoices",
        "invoice_line_items",
        "purchase_orders",
        "po_line_items",
        "payments",
        "payment_allocations",
        "contracts",
        "anomaly_scores",
        "reminders",
        "llm_calls",
        "eval_runs",
    }

    assert set(TABLES) == planned


def test_payment_methods_match_the_pydantic_schema() -> None:
    sql = ddl(TABLES["payments"])

    for method in get_args(PaymentMethod):
        assert f"'{method}'" in sql


def test_a_payment_can_settle_several_invoices() -> None:
    sql = ddl(TABLES["payment_allocations"])

    assert "REFERENCES payments (id) ON DELETE CASCADE" in sql
    assert "REFERENCES invoices (id) ON DELETE SET NULL" in sql  # unmatched numbers survive
    assert "UNIQUE (payment_id, invoice_number)" in sql


def test_purchase_orders_link_to_their_contract() -> None:
    assert "REFERENCES contracts (id) ON DELETE SET NULL" in ddl(TABLES["purchase_orders"])


def test_rate_cards_and_eval_payloads_are_jsonb() -> None:
    assert "rate_card JSONB DEFAULT '[]'::jsonb NOT NULL" in ddl(TABLES["contracts"])
    eval_sql = ddl(TABLES["eval_runs"])
    assert "config JSONB DEFAULT '{}'::jsonb NOT NULL" in eval_sql
    assert "metrics JSONB DEFAULT '{}'::jsonb NOT NULL" in eval_sql


def test_one_score_per_invoice_and_model_version() -> None:
    assert "UNIQUE (invoice_id, model_version)" in ddl(TABLES["anomaly_scores"])


def test_reminders_wait_for_approval() -> None:
    sql = ddl(TABLES["reminders"])

    assert "'pending_approval'" in sql
    assert TABLES["reminders"].c.status.default.arg == "pending_approval"  # type: ignore[union-attr]


def test_llm_calls_mirror_the_cost_record() -> None:
    columns = set(TABLES["llm_calls"].c.keys())

    assert set(LLMCallRecord.model_fields) <= columns
