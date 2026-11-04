"""PostgreSQL engine factory (sessions and ORM wiring arrive with the storage layer)."""

from __future__ import annotations

from sqlalchemy import Engine, create_engine


def make_engine(database_url: str, *, connect_timeout_s: float = 5.0) -> Engine:
    """Create a pooled engine. Nothing connects until the engine is first used."""
    return create_engine(
        database_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": max(1, round(connect_timeout_s))},
    )
