from collections.abc import Iterator
from typing import Any

import pytest
import sqlalchemy
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session, sessionmaker

from ledgerlens.core.config import get_settings
from ledgerlens.storage.postgres import db
from ledgerlens.storage.postgres.db import make_engine, make_session_factory, session_scope


def _spy_create_engine(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    real = sqlalchemy.create_engine

    def spy(url: str, **kwargs: Any) -> Engine:
        seen.update(kwargs, url=url)
        return real(url, **kwargs)

    monkeypatch.setattr(db, "create_engine", spy)
    return seen


def test_postgres_engine_gets_a_connect_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _spy_create_engine(monkeypatch)

    make_engine("postgresql+psycopg://u:p@localhost:1/db", connect_timeout_s=2.4).dispose()

    assert seen["connect_args"] == {"connect_timeout": 2}
    assert seen["pool_pre_ping"] is True


def test_postgres_connect_timeout_is_at_least_one_second(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _spy_create_engine(monkeypatch)

    make_engine("postgresql+psycopg://u:p@localhost:1/db", connect_timeout_s=0.1).dispose()

    assert seen["connect_args"] == {"connect_timeout": 1}


def test_sqlite_engine_ignores_the_postgres_timeout() -> None:
    engine = make_engine("sqlite://", connect_timeout_s=3)

    with engine.connect() as conn:
        assert conn.execute(text("SELECT 1")).scalar_one() == 1
    engine.dispose()


@pytest.fixture
def factory() -> Iterator[sessionmaker[Session]]:
    engine = make_engine("sqlite://")  # in-memory; one connection per thread keeps the table
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE t (x INTEGER)"))
    yield make_session_factory(engine)
    engine.dispose()


def _count(factory: sessionmaker[Session]) -> int:
    with factory() as session:
        return int(session.execute(text("SELECT count(*) FROM t")).scalar_one())


def test_session_scope_commits_on_success(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        session.execute(text("INSERT INTO t VALUES (1)"))

    assert _count(factory) == 1


def test_session_scope_rolls_back_and_reraises(factory: sessionmaker[Session]) -> None:
    with pytest.raises(RuntimeError, match="boom"), session_scope(factory) as session:
        session.execute(text("INSERT INTO t VALUES (1)"))
        raise RuntimeError("boom")

    assert _count(factory) == 0


def test_sessions_keep_rows_readable_after_commit(factory: sessionmaker[Session]) -> None:
    assert factory.kw["expire_on_commit"] is False


def test_get_engine_is_cached_and_built_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "sqlite://")
    get_settings.cache_clear()
    db.get_engine.cache_clear()
    db.get_session_factory.cache_clear()
    try:
        engine = db.get_engine()
        assert engine is db.get_engine()
        assert str(engine.url) == "sqlite://"
        assert db.get_session_factory().kw["bind"] is engine
    finally:
        db.get_engine().dispose()
        get_settings.cache_clear()
        db.get_engine.cache_clear()
        db.get_session_factory.cache_clear()
