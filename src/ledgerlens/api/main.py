"""FastAPI application factory."""

from fastapi import FastAPI

from ledgerlens import __version__
from ledgerlens.api.routers import health
from ledgerlens.core.config import get_settings
from ledgerlens.core.logging import configure_logging


def create_app() -> FastAPI:
    configure_logging(get_settings().log_level)
    app = FastAPI(
        title="LedgerLens API",
        version=__version__,
        description="Agentic GraphRAG over invoices, purchase orders and contracts.",
    )
    app.include_router(health.router)
    return app


app = create_app()
