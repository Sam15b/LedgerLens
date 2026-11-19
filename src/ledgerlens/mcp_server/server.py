"""The LedgerLens MCP server: the ledger's data as tools any MCP client can call.

    python -m ledgerlens.mcp_server.server                      # stdio (Claude Desktop, Inspector)
    python -m ledgerlens.mcp_server.server --transport http     # streamable HTTP on :8765/mcp
    make mcp                                                    # the HTTP one

Built with the Python MCP SDK's ``FastMCP`` (docs/layers/04-agents-mcp.md). The agents
load the same tools through ``langchain-mcp-adapters``, so LangGraph, Claude Desktop and
the MCP Inspector all see one set of tools, and none of them touches a database directly.

:func:`create_server` builds a server over a :class:`~ledgerlens.mcp_server.services
.Services`: the live stores by default (each connected on first use), stand-ins in tests.
Every tool module has a ``register(server, services)`` function, listed in
:data:`TOOL_MODULES`. Over stdio, stdout carries the protocol, so logs go to stderr.
"""

from __future__ import annotations

import argparse
import importlib
import logging
from collections.abc import Sequence
from typing import Literal

from mcp.server.fastmcp import FastMCP

from ledgerlens.core.config import get_settings
from ledgerlens.core.logging import configure_logging
from ledgerlens.mcp_server.services import Services

logger = logging.getLogger(__name__)

NAME = "ledgerlens"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
INSTRUCTIONS = (
    "LedgerLens holds an accounts-payable ledger: invoices, purchase orders, contracts and "
    "payments, as documents (searchable text) and as a knowledge graph. Use the tools to "
    "look things up; cite documents by the citation tags the tools return (doc:INV-889)."
)
TOOL_MODULES: tuple[str, ...] = ("search", "graph", "records")
"""Modules under ``ledgerlens.mcp_server.tools`` whose tools the server offers."""

Transport = Literal["stdio", "http"]


def create_server(
    services: Services | None = None, *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> FastMCP:
    """A server with every tool of :data:`TOOL_MODULES`, reading from ``services``."""
    services = services or Services()
    server = FastMCP(NAME, instructions=INSTRUCTIONS, host=host, port=port)
    for name in TOOL_MODULES:
        module = importlib.import_module(f"ledgerlens.mcp_server.tools.{name}")
        module.register(server, services)
    return server


def run(
    transport: Transport = "stdio",
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    warm_up: bool = True,
) -> None:
    services = Services()
    server = create_server(services, host=host, port=port)
    if warm_up:
        services.warm_up()  # load the search models while the client connects
    if transport == "stdio":
        server.run("stdio")
    else:
        logger.info("MCP server on http://%s:%d/mcp", host, port)
        server.run("streamable-http")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ledgerlens.mcp_server.server", description="Run the LedgerLens MCP server."
    )
    parser.add_argument("--transport", choices=("stdio", "http"), default="stdio")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--no-warm-up",
        dest="warm_up",
        action="store_false",
        help="load the search models on the first search instead of at start",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(get_settings().log_level)  # stderr: stdout is the stdio transport
    run(args.transport, host=args.host, port=args.port, warm_up=args.warm_up)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
