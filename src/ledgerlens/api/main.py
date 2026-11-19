"""FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ledgerlens import __version__
from ledgerlens.api.agent import AgentProvider
from ledgerlens.api.routers import chat, health, reminders
from ledgerlens.core.config import get_settings
from ledgerlens.core.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    await app.state.agents.aclose()


def create_app(agents: AgentProvider | None = None) -> FastAPI:
    """The app; ``agents`` builds the agent graph (default: the live one, on first use)."""
    configure_logging(get_settings().log_level)
    app = FastAPI(
        lifespan=lifespan,
        title="LedgerLens API",
        version=__version__,
        description="Agentic GraphRAG over invoices, purchase orders and contracts.",
    )
    app.state.agents = agents or AgentProvider()
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(reminders.router)
    return app


app = create_app()
