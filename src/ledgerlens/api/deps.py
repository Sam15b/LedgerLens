"""Shared, lazily created clients for the API, and its API-key check.

Each getter builds its client on first use and caches it for the process. Nothing here
opens a connection at import time, so the app (and unit tests) start without any store.

:func:`require_api_key` guards every route but ``/health`` and ``/ready``: with
``API_KEYS`` set, a request must carry one of them as ``X-API-Key`` (or
``Authorization: Bearer <key>``), else 401. With none set, auth is off (development).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from functools import lru_cache
from typing import Annotated

import httpx
from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from neo4j import Driver
from redis import Redis
from sqlalchemy import Engine

from ledgerlens.core.config import get_settings
from ledgerlens.storage.neo4j import driver as neo4j_driver
from ledgerlens.storage.pinecone_store import IndexAdmin, create_index_admin
from ledgerlens.storage.postgres import db

logger = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"
_api_key_header = APIKeyHeader(
    name=API_KEY_HEADER,
    auto_error=False,
    description="One of the keys in the API's API_KEYS setting.",
)


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


def key_id(key: str) -> str:
    """A short, non-secret name for a key (for logs and rate limits)."""
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _supplied_key(request: Request, header: str | None) -> str | None:
    if header:
        return header
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    return token.strip() or None if scheme.lower() == "bearer" else None


def require_api_key(
    request: Request, header: Annotated[str | None, Security(_api_key_header)] = None
) -> str | None:
    """The caller's key id, after checking its key; ``None`` when auth is off."""
    keys = get_settings().api_key_list
    if not keys:
        return None
    supplied = _supplied_key(request, header)
    if supplied is not None:
        given = supplied.encode()
        # compare against every key, so the time taken does not say which one was close
        matched = [hmac.compare_digest(given, key.encode()) for key in keys]
        if any(matched):
            caller = key_id(supplied)
            request.state.api_key_id = caller
            return caller
    logger.info(
        "refused %s %s: %s API key",
        request.method,
        request.url.path,
        "invalid" if supplied else "no",
    )
    raise HTTPException(
        status.HTTP_401_UNAUTHORIZED,
        f"a valid API key is required: send it in the {API_KEY_HEADER} header",
        headers={"WWW-Authenticate": "ApiKey"},
    )
