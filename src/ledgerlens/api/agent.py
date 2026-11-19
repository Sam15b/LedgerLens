"""The agent graph behind the chat API, built once per process on first use.

:func:`build_runtime` wires the real thing: the MCP server's tools (``configs/agents.yaml``
``mcp``: start it with ``make mcp``), the model router, the vendor master, and the
Postgres checkpointer, so a run paused for reminder approval survives an API restart.
:class:`AgentProvider` builds it on the first request rather than at startup, so the API
(and ``/health``) come up even when the MCP server or Postgres is down; ``/chat`` then
answers 503 with the reason, and the next request tries again.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AgentRuntime:
    graph: Any
    """The compiled LangGraph graph (with a checkpointer)."""


Factory = Callable[[AsyncExitStack], Awaitable[AgentRuntime]]


class AgentUnavailableError(RuntimeError):
    pass


async def build_runtime(stack: AsyncExitStack) -> AgentRuntime:
    """Every agent over the live MCP server, with the Postgres checkpointer."""
    from ledgerlens.agents.graph import AgentNodes, build_graph, postgres_checkpointer
    from ledgerlens.agents.nodes.extractor import ExtractorNode, default_extractor
    from ledgerlens.agents.nodes.qa import QANode
    from ledgerlens.agents.nodes.reconciler import ReconcilerNode
    from ledgerlens.agents.nodes.reminder_writer import ReminderWriterNode
    from ledgerlens.agents.nodes.router import Router
    from ledgerlens.agents.tools import load_toolbox
    from ledgerlens.core.config import get_config
    from ledgerlens.llm.router import LLMClient
    from ledgerlens.retrieval.analyze import QueryAnalyzer

    config = get_config().agents
    tools = await load_toolbox()
    llm = LLMClient()
    vendors = await asyncio.to_thread(lambda: QueryAnalyzer.from_postgres().finder)
    saver = await stack.enter_async_context(postgres_checkpointer())
    models = {name: agent.model for name, agent in config.agents.items()}
    nodes = AgentNodes(
        router=Router(llm, model=models.get("router", "fast"), vendors=vendors),
        qa=QANode(tools, llm, model=models.get("qa", "smart"), max_steps=config.max_steps),
        extractor=ExtractorNode(tools, default_extractor(llm)),
        reconciler=ReconcilerNode(tools, llm, model=models.get("reconciler", "smart")),
        reminder_writer=ReminderWriterNode(tools),
    )
    return AgentRuntime(build_graph(nodes, checkpointer=saver, max_steps=config.max_steps))


class AgentProvider:
    """The process's :class:`AgentRuntime`, built on first use; failures are retried on
    the next request."""

    def __init__(self, factory: Factory = build_runtime) -> None:
        self.factory = factory
        self._runtime: AgentRuntime | None = None
        self._stack = AsyncExitStack()
        self._lock = asyncio.Lock()

    async def get(self) -> AgentRuntime:
        if self._runtime is not None:
            return self._runtime
        async with self._lock:
            if self._runtime is None:
                try:
                    self._runtime = await self.factory(self._stack)
                except Exception as exc:
                    await self._stack.aclose()
                    self._stack = AsyncExitStack()
                    logger.warning("agents unavailable: %s", exc)
                    raise AgentUnavailableError(f"{type(exc).__name__}: {exc}") from exc
            return self._runtime

    async def aclose(self) -> None:
        self._runtime = None
        await self._stack.aclose()
