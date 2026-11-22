"""The agents' tools: the LedgerLens MCP server's tools, loaded through
``langchain-mcp-adapters`` (docs/layers/04-agents-mcp.md).

    toolbox = await load_toolbox()                       # configs/agents.yaml: mcp
    qa_tools = toolbox.only("search_documents", "graph_query", "run_analytics")
    outcome, message = await qa_tools.call("search_documents", {"query": "..."})

The agents reach the data only through these tools, exactly as Claude Desktop or the MCP
Inspector would. By default they talk to the running server over streamable HTTP
(``make mcp``), so the search models stay loaded between calls; ``transport: stdio``
starts a server per session instead.

:class:`ToolBox` binds a node to the tools it may use and never raises: an unknown tool,
a tool error or a broken connection comes back as a failed :class:`ToolOutcome` and an
error ``ToolMessage``, so the agent can read the error and carry on.
"""

from __future__ import annotations

import logging
import sys
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ledgerlens.core.config import MCPConnectionConfig, get_config
from ledgerlens.observability.tracing import span

logger = logging.getLogger(__name__)

SERVER = "ledgerlens"


class ToolOutcome(BaseModel):
    """One tool call's result, as the agent sees it."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    call_id: str
    ok: bool
    data: dict[str, Any] | None = None
    """The tool's structured output (``None`` on error)."""
    text: str = ""
    """The text content (the error message when not ``ok``)."""


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "") if isinstance(block, Mapping) else str(block) for block in content
        )
    return str(content)


def tool_attributes(outcome: ToolOutcome) -> dict[str, Any]:
    """Span attributes for a tool call."""
    return {"tool.ok": outcome.ok}


class ToolBox:
    """Tools by name; :meth:`only` narrows them to what one agent may call."""

    def __init__(self, tools: Sequence[BaseTool]) -> None:
        self.tools = {tool.name: tool for tool in tools}

    @property
    def names(self) -> list[str]:
        return sorted(self.tools)

    def only(self, *names: str) -> ToolBox:
        """A toolbox with just ``names`` (those the server offers)."""
        missing = [name for name in names if name not in self.tools]
        if missing:
            logger.warning("the MCP server does not offer %s", ", ".join(missing))
        return ToolBox([self.tools[name] for name in names if name in self.tools])

    async def call(self, name: str, args: Mapping[str, Any]) -> tuple[ToolOutcome, ToolMessage]:
        """Call a tool, in a span; errors come back as a failed outcome and an error
        message."""
        with span(f"tool {name}", kind="tool", inputs=args, attributes={"tool.name": name}) as s:
            outcome, message = await self._call(name, args)
            s.set_attributes(tool_attributes(outcome))
            if outcome.ok:
                s.set_outputs(text=outcome.text)
            else:
                s.record_error(outcome.text)
            return outcome, message

    async def _call(self, name: str, args: Mapping[str, Any]) -> tuple[ToolOutcome, ToolMessage]:
        call_id = f"call_{uuid.uuid4().hex[:12]}"
        tool = self.tools.get(name)
        if tool is None:
            text = f"tool {name!r} is not available here (available: {', '.join(self.names)})"
            return self._failed(name, args, call_id, text)
        try:
            message = await tool.ainvoke(
                {"type": "tool_call", "id": call_id, "name": name, "args": dict(args)}
            )
        except Exception as exc:  # a broken connection, a bad argument the tool raised on
            return self._failed(name, args, call_id, f"{type(exc).__name__}: {exc}")
        if not isinstance(message, ToolMessage):  # pragma: no cover - adapters return one
            message = ToolMessage(_text(message), tool_call_id=call_id, name=name)
        artifact = message.artifact if isinstance(message.artifact, Mapping) else {}
        ok = message.status != "error"
        outcome = ToolOutcome(
            tool=name,
            args=dict(args),
            call_id=call_id,
            ok=ok,
            data=artifact.get("structured_content") if ok else None,
            text=_text(message.content),
        )
        return outcome, message

    @staticmethod
    def _failed(
        name: str, args: Mapping[str, Any], call_id: str, text: str
    ) -> tuple[ToolOutcome, ToolMessage]:
        logger.warning("tool %s failed: %s", name, text)
        outcome = ToolOutcome(tool=name, args=dict(args), call_id=call_id, ok=False, text=text)
        message = ToolMessage(text, tool_call_id=call_id, name=name, status="error")
        return outcome, message


def mcp_connections(config: MCPConnectionConfig | None = None) -> dict[str, Any]:
    """The ``MultiServerMCPClient`` connection for the LedgerLens server."""
    config = config or get_config().agents.mcp
    if config.transport == "stdio":
        return {
            SERVER: {
                "transport": "stdio",
                "command": config.command or sys.executable,
                "args": ["-m", "ledgerlens.mcp_server.server", "--transport", "stdio"],
            }
        }
    return {SERVER: {"transport": "streamable_http", "url": config.url}}


async def load_toolbox(connections: Mapping[str, Any] | None = None) -> ToolBox:
    """Every tool the MCP server offers (``configs/agents.yaml`` ``mcp`` by default)."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(dict(connections or mcp_connections()))
    tools = await client.get_tools()
    logger.info("loaded %d MCP tools", len(tools))
    return ToolBox(tools)
