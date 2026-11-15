"""Alembic environment: runs migrations against ``DATABASE_URL`` with the app's models.

The URL comes from, in order:

1. a connection handed over in ``config.attributes["connection"]`` (tests, scripts);
2. ``alembic -x url=...`` on the command line;
3. ``Settings.database_url`` -- ``DATABASE_URL`` from the environment or ``.env``.
"""

from __future__ import annotations

from logging.config import fileConfig
from typing import Any

from alembic import context
from sqlalchemy import Connection

from ledgerlens.core.config import get_settings
from ledgerlens.storage.postgres.db import make_engine
from ledgerlens.storage.postgres.models import Base

config = context.config

if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def database_url() -> str:
    return context.get_x_argument(as_dictionary=True).get("url") or get_settings().database_url


def _configure(**kwargs: Any) -> None:
    context.configure(
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        **kwargs,
    )


def run_migrations_offline() -> None:
    """``alembic upgrade head --sql``: print the SQL instead of running it."""
    _configure(url=database_url(), literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def _run(connection: Connection) -> None:
    _configure(connection=connection)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        _run(connection)
        return
    engine = make_engine(database_url())
    try:
        with engine.connect() as conn:
            _run(conn)
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
