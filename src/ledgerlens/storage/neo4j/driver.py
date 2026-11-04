"""Neo4j driver factory."""

from __future__ import annotations

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
