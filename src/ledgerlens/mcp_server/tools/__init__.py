"""The MCP server's tools, one module per area; each has ``register(server, services)``.

Shared here: :data:`READ_ONLY` annotations, :func:`in_thread` (the stores' clients block,
so a tool runs them off the server's event loop) and :func:`jsonable` (dates and
decimals in free-form rows, as JSON values).
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from datetime import date, datetime
from decimal import Decimal
from typing import Any, ParamSpec, TypeVar

import anyio.to_thread
from mcp.types import ToolAnnotations

P = ParamSpec("P")
T = TypeVar("T")

READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)
"""For tools that only read the ledger."""


async def in_thread(fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
    """``fn(*args, **kwargs)`` in a worker thread."""
    return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))


def jsonable(value: Any) -> Any:
    """``value`` with dates as ISO strings and decimals as strings (exact), inside lists
    and maps too; everything else as is."""
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    return value
