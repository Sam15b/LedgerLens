"""Where the agents find the MCP server: configs/agents.yaml, unless MCP_URL says otherwise."""

from collections.abc import Iterator

import pytest

from ledgerlens.agents.tools import SERVER, mcp_connections
from ledgerlens.core.config import MCPConnectionConfig, get_settings

CONFIG = MCPConnectionConfig(url="http://localhost:8765/mcp")


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    get_settings.cache_clear()
    yield monkeypatch
    get_settings.cache_clear()


def test_the_configured_url_by_default(settings: pytest.MonkeyPatch) -> None:
    settings.setenv("MCP_URL", "")
    assert mcp_connections(CONFIG)[SERVER] == {
        "transport": "streamable_http",
        "url": "http://localhost:8765/mcp",
    }


def test_mcp_url_overrides_it(settings: pytest.MonkeyPatch) -> None:
    settings.setenv("MCP_URL", "http://mcp:8765/mcp")
    get_settings.cache_clear()
    assert mcp_connections(CONFIG)[SERVER]["url"] == "http://mcp:8765/mcp"


def test_stdio_ignores_it(settings: pytest.MonkeyPatch) -> None:
    settings.setenv("MCP_URL", "http://mcp:8765/mcp")
    get_settings.cache_clear()
    connection = mcp_connections(MCPConnectionConfig(transport="stdio"))[SERVER]
    assert connection["transport"] == "stdio" and "url" not in connection
