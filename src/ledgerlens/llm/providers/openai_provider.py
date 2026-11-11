"""OpenAI adapter (GPT-4o family) over the Chat Completions REST API.

* ``json_schema`` uses native structured outputs (``response_format.type = "json_schema"``,
  ``strict: true``). Strict mode needs every object closed (``additionalProperties: false``)
  with all properties required and no ``default`` keyword, so Pydantic schemas are
  converted by :func:`~ledgerlens.llm.providers._schema.strict_json_schema`.
* ``json_mode`` alone uses ``{"type": "json_object"}`` (the prompt must mention JSON).
* ``OPENAI_BASE_URL`` lets the same adapter talk to an OpenAI-compatible gateway.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from types import TracebackType
from typing import Any

import httpx

from ledgerlens.core.config import get_settings
from ledgerlens.llm.providers._http import check_response, json_body, transport_errors
from ledgerlens.llm.providers._schema import strict_json_schema
from ledgerlens.llm.types import (
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMTransientError,
    Usage,
)

CHAT_PATH = "/chat/completions"

# 429 codes that will not clear by waiting: retrying them only burns the backoff budget.
_NOT_RETRYABLE_429_CODES = frozenset({"insufficient_quota", "billing_hard_limit_reached"})


class OpenAIProvider:
    name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        client: httpx.Client | None = None,
        strict_schema: bool = True,
    ) -> None:
        settings = get_settings()
        self._api_key = api_key if api_key is not None else settings.openai_api_key
        self.base_url = (base_url or settings.openai_base_url).rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.Client()
        self._strict_schema = strict_schema

    # ------------------------------------------------------------------ LLMProvider

    def complete(self, req: LLMRequest) -> LLMResponse:
        start = time.perf_counter()
        with transport_errors(self.name):
            response = self._client.post(
                self._url(),
                json=self._payload(req, stream=False),
                headers=self._headers(),
                timeout=req.timeout_s,
            )
        self._check(response)
        data = json_body(self.name, response)
        latency_ms = (time.perf_counter() - start) * 1000

        choice = _first_choice(data)
        message = choice.get("message") or {}
        if message.get("refusal"):
            raise LLMProviderError(f"openai refused: {message['refusal']}", provider=self.name)
        content = message.get("content")
        if not isinstance(content, str):
            raise LLMProviderError("openai response has no message content", provider=self.name)

        return LLMResponse(
            text=content,
            provider=self.name,
            model=req.model,
            usage=_usage(data),
            latency_ms=round(latency_ms, 2),
            finish_reason=choice.get("finish_reason"),
        )

    def stream(self, req: LLMRequest) -> Iterator[str]:
        """Yield content deltas from the server-sent event stream."""
        with (
            transport_errors(self.name),
            self._client.stream(
                "POST",
                self._url(),
                json=self._payload(req, stream=True),
                headers=self._headers(),
                timeout=req.timeout_s,
            ) as response,
        ):
            self._check(response)
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue  # blank keep-alives and SSE comments
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    return
                event = _decode_event(data)
                if "error" in event:
                    raise LLMProviderError(f"openai: {event['error']}", provider=self.name)
                choices = event.get("choices") or [{}]
                content = (choices[0].get("delta") or {}).get("content")
                if content:
                    yield content

    # ------------------------------------------------------------------ lifecycle

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OpenAIProvider:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------ helpers

    def _url(self) -> str:
        return f"{self.base_url}{CHAT_PATH}"

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            raise LLMProviderError("OPENAI_API_KEY is not set", provider=self.name)
        return {"Authorization": f"Bearer {self._api_key}"}

    def _check(self, response: httpx.Response) -> None:
        try:
            check_response(self.name, response)
        except LLMTransientError as exc:
            if exc.status_code == 429 and _error_code(response) in _NOT_RETRYABLE_429_CODES:
                raise LLMProviderError(str(exc), provider=self.name, status_code=429) from exc
            raise

    def _payload(self, req: LLMRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": req.model,
            "messages": [{"role": m.role, "content": m.content} for m in req.messages],
            "temperature": req.temperature,
        }
        if req.max_tokens is not None:
            payload["max_completion_tokens"] = req.max_tokens
        if req.json_schema is not None:
            schema = strict_json_schema(req.json_schema) if self._strict_schema else req.json_schema
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": req.schema_name,
                    "schema": schema,
                    "strict": self._strict_schema,
                },
            }
        elif req.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if stream:
            payload["stream"] = True
        return payload


# --------------------------------------------------------------------------- parsing


def _first_choice(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise LLMProviderError("openai response has no choices", provider="openai")
    return choices[0]


def _usage(data: dict[str, Any]) -> Usage:
    usage = data.get("usage") or {}
    return Usage(
        tokens_in=int(usage.get("prompt_tokens") or 0),
        tokens_out=int(usage.get("completion_tokens") or 0),
    )


def _error_code(response: httpx.Response) -> str | None:
    try:
        error = response.json().get("error")
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        return None
    if isinstance(error, dict):
        code = error.get("code") or error.get("type")
        return str(code) if code else None
    return None


def _decode_event(data: str) -> dict[str, Any]:
    try:
        event = json.loads(data)
    except json.JSONDecodeError as exc:
        raise LLMProviderError(
            f"openai stream sent a non-JSON event: {data[:200]!r}", provider="openai"
        ) from exc
    if not isinstance(event, dict):
        raise LLMProviderError("openai stream sent a non-object event", provider="openai")
    return event
