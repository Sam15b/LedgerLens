"""Health check endpoint."""

from fastapi import APIRouter

from ledgerlens import __version__
from ledgerlens.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__, "env": get_settings().app_env}
