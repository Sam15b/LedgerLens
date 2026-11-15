"""Alembic migrations against a real PostgreSQL (``docker compose up -d postgres``)."""

from collections.abc import Callable

import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import Engine, inspect

from ledgerlens.storage.postgres.models import Base

pytestmark = pytest.mark.integration

TABLES = set(Base.metadata.tables)
Alembic = Callable[[Engine, str, str], None]


def _tables(engine: Engine) -> set[str]:
    return set(inspect(engine).get_table_names()) - {"alembic_version"}


def test_upgrade_creates_every_table_and_downgrade_removes_them(
    engine: Engine, alembic: Alembic
) -> None:
    alembic(engine, "upgrade", "head")
    assert _tables(engine) == TABLES

    alembic(engine, "downgrade", "base")
    assert _tables(engine) == set()

    alembic(engine, "upgrade", "head")  # and the round trip is repeatable
    assert _tables(engine) == TABLES


def test_migrations_match_the_models(migrated_engine: Engine) -> None:
    """Fails when a model changes without a migration (what ``alembic check`` reports)."""
    with migrated_engine.connect() as conn:
        context = MigrationContext.configure(
            conn, opts={"compare_type": True, "compare_server_default": True}
        )
        diff = compare_metadata(context, Base.metadata)

    assert diff == []


def test_upgrade_head_is_idempotent(migrated_engine: Engine, alembic: Alembic) -> None:
    alembic(migrated_engine, "upgrade", "head")

    assert _tables(migrated_engine) == TABLES
