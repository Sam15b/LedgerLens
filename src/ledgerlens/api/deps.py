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
from ledgerlens.storage.neo4j.driver import create_driver
from ledgerlens.storage.pinecone_store import IndexAdmin, create_index_admin
from ledgerlens.storage.postgres.db import make_engine


def _timeout() -> float:
    return get_settings().health_check_timeout_s


@lru_cache
def get_engine() -> Engine:
    return make_engine(get_settings().database_url, connect_timeout_s=_timeout())


@lru_cache
def get_neo4j_driver() -> Driver:
    s = get_settings()
    return create_driver(
        s.neo4j_uri, s.neo4j_user, s.neo4j_password, connection_timeout_s=_timeout()
    )


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
