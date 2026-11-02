"""FastAPI application factory."""

from fastapi import FastAPI

from ledgerlens import __version__
from ledgerlens.api.routers import health


def create_app() -> FastAPI:
    app = FastAPI(
        title="LedgerLens API",
        version=__version__,
        description="Agentic GraphRAG over invoices, purchase orders and contracts.",
    )
    app.include_router(health.router)
    return app


app = create_app()
