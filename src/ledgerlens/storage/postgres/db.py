"""PostgreSQL engine and session management.

* :func:`make_engine` -- a pooled engine for one database URL.
* :func:`make_session_factory` -- a ``sessionmaker`` bound to an engine.
* :func:`session_scope` -- one unit of work: commit on success, roll back on any error.
* :func:`get_engine` / :func:`get_session_factory` -- process-wide instances built from
  ``DATABASE_URL`` the first time they are used.

Nothing here connects at import time; the first query opens the first connection.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from ledgerlens.core.config import get_settings


def make_engine(database_url: str, *, connect_timeout_s: float = 5.0) -> Engine:
    """Create a pooled engine. Nothing connects until the engine is first used.

    ``connect_timeout_s`` applies to PostgreSQL drivers; other backends (SQLite in unit
    tests) ignore it.
    """
    connect_args: dict[str, object] = {}
    if make_url(database_url).get_backend_name() == "postgresql":
        connect_args["connect_timeout"] = max(1, round(connect_timeout_s))
    return create_engine(database_url, pool_pre_ping=True, connect_args=connect_args)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Sessions bound to ``engine``.

    ``expire_on_commit=False`` keeps loaded rows readable after the unit of work commits,
    so a repository can return ORM objects to a caller that no longer holds the session.
    """
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session_scope(factory: sessionmaker[Session] | None = None) -> Iterator[Session]:
    """One transaction: commit if the block succeeds, roll back if it raises.

    ::

        with session_scope() as session:
            session.add(row)
    """
    session = (factory or get_session_factory())()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


@lru_cache
def get_engine() -> Engine:
    """The process-wide engine for ``DATABASE_URL``."""
    settings = get_settings()
    return make_engine(settings.database_url, connect_timeout_s=settings.health_check_timeout_s)


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    """Sessions on :func:`get_engine`."""
    return make_session_factory(get_engine())
