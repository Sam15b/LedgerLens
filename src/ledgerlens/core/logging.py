"""Logging setup.

* ``dev``: human-readable lines.  Any other ``APP_ENV``: one JSON object per line.
* Every record carries the current ``request_id`` (set per API request / job via
  :func:`request_context` or :func:`set_request_id`), so one request can be traced
  across API, agents, tools and LLM calls.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Any

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

# Attributes every LogRecord has; anything else was passed via ``extra=`` and is emitted.
_RESERVED = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {
    "message",
    "asctime",
    "request_id",
}

_DEV_FORMAT = "%(asctime)s %(levelname)-8s %(name)s [%(request_id)s] %(message)s"


def get_request_id() -> str | None:
    return _request_id.get()


def set_request_id(request_id: str | None) -> Token[str | None]:
    """Bind a request id to the current context; pass the token to :func:`reset_request_id`."""
    return _request_id.set(request_id)


def reset_request_id(token: Token[str | None]) -> None:
    _request_id.reset(token)


@contextmanager
def request_context(request_id: str) -> Iterator[None]:
    token = set_request_id(request_id)
    try:
        yield
    finally:
        reset_request_id(token)


class RequestIdFilter(logging.Filter):
    """Attach the context's request id (or ``-``) to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or "-"
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None) or get_request_id(),
        }
        for key, value in vars(record).items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str | int = "INFO", *, json_logs: bool | None = None) -> None:
    """Configure the root logger. Safe to call more than once (replaces our handler).

    ``json_logs`` defaults to ``True`` unless ``APP_ENV`` is ``dev``.
    """
    if json_logs is None:
        from ledgerlens.core.config import get_settings

        json_logs = get_settings().app_env != "dev"

    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(JsonFormatter() if json_logs else logging.Formatter(_DEV_FORMAT))
    handler.set_name("ledgerlens")

    root = logging.getLogger()
    for existing in [h for h in root.handlers if h.get_name() == "ledgerlens"]:
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper() if isinstance(level, str) else level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
