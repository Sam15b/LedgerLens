"""Fixtures for integration tests (``pytest -m integration``, needs ``docker compose up``).

PostgreSQL: every test module gets its own throwaway database on the same server, so migrations and
writes never touch the ``ledgerlens`` development data, and tests can run in any order.
"""

import uuid
from collections.abc import Callable, Iterator

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import Session, sessionmaker

from ledgerlens.core.config import get_settings, repo_path
from ledgerlens.storage.postgres.db import make_engine, make_session_factory


def alembic_config(engine: Engine) -> Config:
    """``alembic.ini`` from the repo, running on ``engine`` instead of ``DATABASE_URL``."""
    config = Config(str(repo_path("alembic.ini")))
    config.attributes["configure_logger"] = False  # keep pytest's log capture
    return config


def run_alembic(engine: Engine, action: str, revision: str) -> None:
    """``alembic upgrade|downgrade <revision>`` on ``engine``."""
    config = alembic_config(engine)
    with engine.begin() as conn:
        config.attributes["connection"] = conn
        getattr(command, action)(config, revision)


AlembicRunner = Callable[[Engine, str, str], None]


@pytest.fixture(scope="session")
def alembic() -> AlembicRunner:
    """``alembic(engine, "upgrade" | "downgrade", revision)``."""
    return run_alembic


@pytest.fixture(scope="module")
def database_url() -> Iterator[URL]:
    """A new, empty database, dropped after the module's tests."""
    server = make_url(get_settings().database_url)
    name = f"ledgerlens_it_{uuid.uuid4().hex[:10]}"
    admin = make_engine(server.set(database="postgres").render_as_string(hide_password=False))
    admin = admin.execution_options(isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    try:
        yield server.set(database=name)
    finally:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


@pytest.fixture(scope="module")
def engine(database_url: URL) -> Iterator[Engine]:
    """An engine on the throwaway database (not migrated)."""
    engine = make_engine(database_url.render_as_string(hide_password=False))
    yield engine
    engine.dispose()


@pytest.fixture(scope="module")
def migrated_engine(engine: Engine) -> Engine:
    """The throwaway database at ``alembic upgrade head``."""
    run_alembic(engine, "upgrade", "head")
    return engine


@pytest.fixture
def sessions(migrated_engine: Engine) -> Iterator[sessionmaker[Session]]:
    """Sessions on the migrated database; every table is emptied after the test."""
    yield make_session_factory(migrated_engine)
    with migrated_engine.begin() as conn:
        tables = conn.execute(
            text(
                "SELECT string_agg(quote_ident(tablename), ', ') FROM pg_tables "
                "WHERE schemaname = 'public' AND tablename <> 'alembic_version'"
            )
        ).scalar_one()
        if tables:
            conn.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
