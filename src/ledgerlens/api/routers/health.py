"""Liveness / readiness endpoints.

* ``GET /health`` -- always 200 while the process is up; reports every dependency.
  ``status`` is ``ok`` when all *required* dependencies are up, otherwise ``degraded``.
* ``GET /ready`` -- same body, but 503 unless every required dependency is up.
  Use it for container readiness probes.
* ``GET /livez`` -- 200 ``ok`` while the process serves requests, without touching any
  store. For probes that run every few seconds (the load balancer's and the container's
  health checks): ``/health`` calls all four stores, two of them over the internet.
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel

from ledgerlens import __version__
from ledgerlens.api.health_checks import DependencyStatus, HealthCheck, default_checks, run_checks
from ledgerlens.core.config import get_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    env: str
    dependencies: dict[str, DependencyStatus]


def get_health_checks() -> list[HealthCheck]:
    """Dependency hook; tests override it to avoid touching real stores."""
    return default_checks()


Checks = Annotated[list[HealthCheck], Depends(get_health_checks)]


def _build(checks: list[HealthCheck]) -> HealthResponse:
    settings = get_settings()
    results = run_checks(checks, timeout_s=settings.health_check_timeout_s + 1)
    healthy = all(r.status == "ok" for r in results.values() if r.required)
    return HealthResponse(
        status="ok" if healthy else "degraded",
        version=__version__,
        env=settings.app_env,
        dependencies=results,
    )


@router.get("/health")
def health(checks: Checks) -> HealthResponse:
    return _build(checks)


@router.get("/livez", include_in_schema=False)
def livez() -> Response:
    return Response("ok", media_type="text/plain")


@router.get("/ready")
def ready(checks: Checks, response: Response) -> HealthResponse:
    body = _build(checks)
    if body.status != "ok":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return body
