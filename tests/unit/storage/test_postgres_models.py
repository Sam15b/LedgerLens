"""ORM model shape, checked without a database by compiling the DDL for PostgreSQL."""

import pytest
from sqlalchemy import Table
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable

from ledgerlens.core.schemas import DocType, DocumentStatus
from ledgerlens.storage.postgres.models import Base

TABLES = Base.metadata.tables


def ddl(table: Table) -> str:
    dialect = postgresql.dialect()  # type: ignore[no-untyped-call]
    statements = [CreateTable(table)] + [CreateIndex(ix) for ix in table.indexes]
    return "\n".join(str(s.compile(dialect=dialect)) for s in statements)


@pytest.mark.parametrize("name", sorted(TABLES))
def test_every_table_compiles_for_postgres(name: str) -> None:
    assert f"CREATE TABLE {name}" in ddl(TABLES[name])


def test_core_tables_exist() -> None:
    assert {"documents", "chunks", "vendors", "invoices", "invoice_line_items"} <= set(TABLES)


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
