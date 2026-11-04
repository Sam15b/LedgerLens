"""Neo4j driver factory and schema management."""

from __future__ import annotations

from importlib.resources import files
from typing import Protocol

from neo4j import Driver, GraphDatabase


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
