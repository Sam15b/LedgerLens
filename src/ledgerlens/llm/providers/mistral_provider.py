"""Mistral adapter over the Chat Completions REST API (``POST /v1/chat/completions``).

* ``json_schema`` uses Mistral's custom structured outputs
  (``response_format.type = "json_schema"``, ``strict: true``) with the same strict schema
  conversion as OpenAI.
* ``json_mode`` alone uses JSON mode (``{"type": "json_object"}``); the prompt must ask for
  JSON.
* Reasoning models return ``content`` as a list of chunks; only ``text`` chunks are kept,
  ``thinking`` chunks are dropped.

The REST API is used through httpx rather than the ``mistralai`` SDK, so errors are
classified by the shared helpers and tests replay recorded responses through
``httpx.MockTransport``.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from types import TracebackType
from typing import Any

import httpx

from ledgerlens.core.config import get_settings
from ledgerlens.llm.providers._http import check_response, json_body, sse_data, transport_errors
from ledgerlens.llm.providers._schema import strict_json_schema
from ledgerlens.llm.types import LLMProviderError, LLMRequest, LLMResponse, Usage

CHAT_PATH = "/chat/completions"


class MistralProvider:
    name = "mistral"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        settings = get_settings()
        self._api_key = api_key if api_key is not None else settings.mistral_api_key
        self.base_url = (base_url or settings.mistral_base_url).rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.Client()

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
        check_response(self.name, response)
        data = json_body(self.name, response)
        latency_ms = (time.perf_counter() - start) * 1000

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise LLMProviderError("mistral response has no choices", provider=self.name)
        choice = choices[0]
        text = _content_text((choice.get("message") or {}).get("content"))
        if text is None:
            raise LLMProviderError("mistral response has no message content", provider=self.name)

        return LLMResponse(
            text=text,
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
            check_response(self.name, response)
            for event in sse_data(self.name, response):
                if "error" in event:
                    raise LLMProviderError(f"mistral: {event['error']}", provider=self.name)
                choices = event.get("choices") or [{}]
                text = _content_text((choices[0].get("delta") or {}).get("content"))
                if text:
                    yield text

    # ------------------------------------------------------------------ lifecycle

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> MistralProvider:
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
            raise LLMProviderError("MISTRAL_API_KEY is not set", provider=self.name)
        return {"Authorization": f"Bearer {self._api_key}"}

    def _payload(self, req: LLMRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": req.model,
            "messages": [{"role": m.role, "content": m.content} for m in req.messages],
            "temperature": req.temperature,
        }
        if req.max_tokens is not None:
            payload["max_tokens"] = req.max_tokens
        if req.json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": req.schema_name,
                    "schema": strict_json_schema(req.json_schema),
                    "strict": True,
                },
            }
        elif req.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if stream:
            payload["stream"] = True
        return payload


# --------------------------------------------------------------------------- parsing


def _content_text(content: Any) -> str | None:
    """Text of a message/delta ``content``: a string, or a list of typed chunks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            chunk["text"]
            for chunk in content
            if isinstance(chunk, dict)
            and chunk.get("type") == "text"
            and isinstance(chunk.get("text"), str)
        )
    return None


def _usage(data: dict[str, Any]) -> Usage:
    usage = data.get("usage") or {}
    return Usage(
        tokens_in=int(usage.get("prompt_tokens") or 0),
        tokens_out=int(usage.get("completion_tokens") or 0),
    )
