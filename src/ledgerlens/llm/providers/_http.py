"""HTTP helpers shared by the httpx-based provider adapters.

They turn transport failures and HTTP error statuses into the router's error classes, so
every adapter classifies "retry" vs "fall back" the same way.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx

from ledgerlens.llm.types import LLMProviderError, LLMTransientError

# Statuses worth retrying on the same provider; every 5xx is retried too.
_TRANSIENT_STATUS = frozenset({408, 425, 429})
_DETAIL_MAX_CHARS = 300


def is_transient_status(status_code: int) -> bool:
    return status_code in _TRANSIENT_STATUS or status_code >= 500


def check_response(provider: str, response: httpx.Response) -> None:
    """Raise the matching router error if ``response`` is an HTTP error. Reads the body."""
    if response.is_success:
        return
    response.read()
    code = response.status_code
    message = f"{provider} returned HTTP {code}: {_error_detail(response)}"
    if is_transient_status(code):
        raise LLMTransientError(
            message,
            provider=provider,
            status_code=code,
            retry_after_s=_retry_after_s(response),
        )
    raise LLMProviderError(message, provider=provider, status_code=code)


@contextmanager
def transport_errors(provider: str) -> Iterator[None]:
    """Map httpx timeouts and connection failures to :class:`LLMTransientError`."""
    try:
        yield
    except httpx.TimeoutException as exc:
        message = f"{provider} timed out: {exc!r}"
        raise LLMTransientError(message, provider=provider) from exc
    except httpx.TransportError as exc:
        message = f"{provider} connection failed: {exc!r}"
        raise LLMTransientError(message, provider=provider) from exc


def json_body(provider: str, response: httpx.Response) -> dict[str, Any]:
    """Decode a JSON-object response body or raise :class:`LLMProviderError`."""
    try:
        data = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise LLMProviderError(
            f"{provider} returned a non-JSON body: {response.text[:_DETAIL_MAX_CHARS]!r}",
            provider=provider,
        ) from exc
    if not isinstance(data, dict):
        raise LLMProviderError(
            f"{provider} returned {type(data).__name__}, expected an object", provider=provider
        )
    return data


def _error_detail(response: httpx.Response) -> str:
    try:
        data = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return response.text[:_DETAIL_MAX_CHARS] or response.reason_phrase
    error = data.get("error", data) if isinstance(data, dict) else data
    if isinstance(error, dict):
        error = error.get("message", error)
    return str(error)[:_DETAIL_MAX_CHARS]


def _retry_after_s(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None  # HTTP-date form; fall back to the configured backoff
    return max(seconds, 0.0)
