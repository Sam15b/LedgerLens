"""Shared, lazily created clients for the API.

Each getter builds its client on first use and caches it for the process. Nothing here
opens a connection at import time, so the app (and unit tests) start without any store.
"""

from __future__ import annotations

from functools import lru_cache

import httpx
from neo4j import Driver
from redis import Redis
from sqlalchemy import Engine

from ledgerlens.core.config import get_settings
from ledgerlens.storage.neo4j import driver as neo4j_driver
from ledgerlens.storage.pinecone_store import IndexAdmin, create_index_admin
from ledgerlens.storage.postgres import db


def _timeout() -> float:
    return get_settings().health_check_timeout_s


def get_engine() -> Engine:
    """The shared Postgres engine (one pool per process, also used by repositories)."""
    return db.get_engine()


def get_neo4j_driver() -> Driver:
    """The shared Neo4j driver (one connection pool per process, also used by the linker)."""
    return neo4j_driver.get_driver()


@lru_cache
def get_redis() -> Redis:
    timeout = _timeout()
    return Redis.from_url(
        get_settings().redis_url, socket_connect_timeout=timeout, socket_timeout=timeout
    )


@lru_cache
def get_pinecone_admin() -> IndexAdmin:
    s = get_settings()
    return create_index_admin(s.pinecone_api_key, s.pinecone_host, timeout_s=_timeout())


@lru_cache
def get_ollama_http() -> httpx.Client:
    return httpx.Client(base_url=get_settings().ollama_base_url, timeout=_timeout())
