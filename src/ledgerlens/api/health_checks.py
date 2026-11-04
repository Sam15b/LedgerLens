"""Dependency health checks used by ``/health``, ``/ready`` and the integration tests.

A check is a callable that returns an optional detail string on success and raises on
failure. :func:`run_checks` runs them concurrently with a time limit, so one hanging store
cannot stall the endpoint.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel
from sqlalchemy import text

from ledgerlens.api import deps
from ledgerlens.core.config import get_settings

CheckFn = Callable[[], str | None]


@dataclass(frozen=True)
class HealthCheck:
    name: str
    check: CheckFn
    required: bool = True  # required checks gate /ready


class DependencyStatus(BaseModel):
    status: Literal["ok", "down"]
    required: bool
    latency_ms: float | None = None
    detail: str | None = None
    error: str | None = None


def check_postgres() -> str | None:
    with deps.get_engine().connect() as conn:
        conn.execute(text("SELECT 1"))
    return None


def check_neo4j() -> str | None:
    deps.get_neo4j_driver().verify_connectivity()
    return None


def check_redis() -> str | None:
    deps.get_redis().ping()
    return None


def check_pinecone() -> str | None:
    name = get_settings().pinecone_index
    info = deps.get_pinecone_admin().describe(name)
    if info is None:
        return f"index {name!r} missing; run scripts/create_pinecone_index.py"
    return f"index {name!r} ready" if info.ready else f"index {name!r} initializing"


def check_ollama() -> str | None:
    response = deps.get_ollama_http().get("/api/tags")
    response.raise_for_status()
    models = response.json().get("models", [])
    return f"{len(models)} model(s) pulled"


def default_checks() -> list[HealthCheck]:
    return [
        HealthCheck("postgres", check_postgres),
        HealthCheck("neo4j", check_neo4j),
        HealthCheck("redis", check_redis),
        HealthCheck("pinecone", check_pinecone),
        # Local LLMs are optional: cloud deployments route to hosted providers instead.
        HealthCheck("ollama", check_ollama, required=False),
    ]


def _describe_error(exc: BaseException) -> str:
    message = " ".join(str(exc).split()) or exc.__class__.__name__
    return f"{exc.__class__.__name__}: {message}"[:300]


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def _run_one(check: HealthCheck) -> DependencyStatus:
    start = time.perf_counter()
    try:
        detail = check.check()
    except Exception as exc:  # any failure means "down": report it, never raise
        return DependencyStatus(
            status="down",
            required=check.required,
            latency_ms=_elapsed_ms(start),
            error=_describe_error(exc),
        )
    return DependencyStatus(
        status="ok", required=check.required, latency_ms=_elapsed_ms(start), detail=detail
    )


def run_checks(checks: list[HealthCheck], *, timeout_s: float) -> dict[str, DependencyStatus]:
    """Run all checks concurrently; a check still running after ``timeout_s`` is down."""
    if not checks:
        return {}
    pool = ThreadPoolExecutor(max_workers=len(checks), thread_name_prefix="health")
    try:
        futures = {check.name: pool.submit(_run_one, check) for check in checks}
        wait(futures.values(), timeout=timeout_s)
        results: dict[str, DependencyStatus] = {}
        for check in checks:
            future = futures[check.name]
            if future.done():
                results[check.name] = future.result()
            else:
                results[check.name] = DependencyStatus(
                    status="down", required=check.required, error=f"timed out after {timeout_s}s"
                )
        return results
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
