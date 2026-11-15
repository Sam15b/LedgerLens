"""Neo4j driver, sessions and schema management.

* :func:`create_driver` / :func:`get_driver` -- a driver (lazy connections; one per process
  from ``NEO4J_*`` settings).
* :func:`write_session` / :func:`read_session` -- sessions with the matching access mode
  (routes to the leader or a follower in a cluster).
* :func:`run_write` / :func:`run_read` -- one query in a managed transaction, retried on
  transient errors, returning plain dicts.
* :func:`run_read_only` -- for Cypher the application did not write itself (Text2Cypher):
  the query is ``EXPLAIN``-ed first and refused unless Neo4j classifies it as read-only.
  A READ access mode alone does not stop writes on a single instance.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import lru_cache
from importlib.resources import files
from typing import Any, Protocol

from neo4j import READ_ACCESS, WRITE_ACCESS, Driver, GraphDatabase, ManagedTransaction, Session

from ledgerlens.core.config import get_settings


class WriteQueryError(PermissionError):
    """A query given to :func:`run_read_only` would change the graph."""


def create_driver(
    uri: str, user: str, password: str, *, connection_timeout_s: float = 5.0
) -> Driver:
    """Create a driver. Connections are opened lazily on first use."""
    return GraphDatabase.driver(
        uri,
        auth=(user, password),
        connection_timeout=connection_timeout_s,
        connection_acquisition_timeout=connection_timeout_s,
    )


@lru_cache
def get_driver() -> Driver:
    """The process-wide driver for ``NEO4J_URI`` / ``NEO4J_USER`` / ``NEO4J_PASSWORD``."""
    settings = get_settings()
    return create_driver(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        connection_timeout_s=settings.health_check_timeout_s,
    )


@contextmanager
def write_session(driver: Driver, *, database: str | None = None) -> Iterator[Session]:
    with driver.session(database=database, default_access_mode=WRITE_ACCESS) as session:
        yield session


@contextmanager
def read_session(driver: Driver, *, database: str | None = None) -> Iterator[Session]:
    with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
        yield session


def _records(tx: ManagedTransaction, query: str, params: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [record.data() for record in tx.run(query, dict(params))]


def run_write(
    driver: Driver,
    query: str,
    params: Mapping[str, Any] | None = None,
    *,
    database: str | None = None,
) -> list[dict[str, Any]]:
    """Run ``query`` in a write transaction (retried on transient errors)."""
    with write_session(driver, database=database) as session:
        return session.execute_write(_records, query, params or {})


def run_read(
    driver: Driver,
    query: str,
    params: Mapping[str, Any] | None = None,
    *,
    database: str | None = None,
) -> list[dict[str, Any]]:
    """Run ``query`` in a read transaction (retried on transient errors)."""
    with read_session(driver, database=database) as session:
        return session.execute_read(_records, query, params or {})


def query_type(
    driver: Driver,
    query: str,
    params: Mapping[str, Any] | None = None,
    *,
    database: str | None = None,
) -> str:
    """Neo4j's classification of ``query`` without running it: ``r`` (read only), ``rw``,
    ``w`` (write only) or ``s`` (schema)."""
    with read_session(driver, database=database) as session:
        summary = session.run(f"EXPLAIN {query}", dict(params or {})).consume()
    return str(summary.query_type)


def run_read_only(
    driver: Driver,
    query: str,
    params: Mapping[str, Any] | None = None,
    *,
    database: str | None = None,
) -> list[dict[str, Any]]:
    """Run untrusted Cypher only if it cannot write; raises :class:`WriteQueryError`."""
    kind = query_type(driver, query, params, database=database)
    if kind != "r":
        raise WriteQueryError(f"refusing a {kind!r} query: only read-only Cypher may run here")
    return run_read(driver, query, params, database=database)


def split_cypher(script: str) -> list[str]:
    """Split a Cypher script into statements.

    Drops ``//`` comment lines and splits on ``;``. Intended for schema files, which
    contain no string literals with semicolons.
    """
    lines = [line for line in script.splitlines() if not line.strip().startswith("//")]
    return [stmt.strip() for stmt in "\n".join(lines).split(";") if stmt.strip()]


def load_schema() -> list[str]:
    """Statements from the packaged ``schema.cypher``."""
    script = files("ledgerlens.storage.neo4j").joinpath("schema.cypher").read_text("utf-8")
    return split_cypher(script)


class _QueryRunner(Protocol):
    def execute_query(self, query: str, /, *, database_: str | None = None) -> object: ...


def apply_schema(
    driver: _QueryRunner, statements: list[str] | None = None, *, database: str | None = None
) -> int:
    """Apply schema statements (default: the packaged schema). Returns how many ran.

    Every statement uses ``IF NOT EXISTS``, so re-applying is a no-op.
    """
    to_run = load_schema() if statements is None else statements
    for statement in to_run:
        driver.execute_query(statement, database_=database)
    return len(to_run)
