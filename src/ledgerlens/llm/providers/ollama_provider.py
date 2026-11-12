"""Ollama adapter (Llama 3 and the fine-tuned LedgerLens model, served locally for $0).

Talks to Ollama's REST API (``POST /api/chat``) with httpx. ``json_mode`` maps to
``format: "json"``; a JSON schema is passed as ``format`` itself, which Ollama uses to
constrain the output. Token counts come from ``prompt_eval_count`` / ``eval_count``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Generator
from types import TracebackType
from typing import Any

import httpx

from ledgerlens.core.config import get_settings
from ledgerlens.llm.providers._http import check_response, json_body, transport_errors
from ledgerlens.llm.types import LLMProviderError, LLMRequest, LLMResponse, Usage

CHAT_PATH = "/api/chat"


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        base_url: str | None = None,
        *,
        client: httpx.Client | None = None,
        keep_alive: str | None = None,
    ) -> None:
        """``client`` is injectable for tests; ``keep_alive`` (e.g. ``"10m"``) is passed
        through to Ollama to control how long the model stays loaded."""
        self.base_url = (base_url or get_settings().ollama_base_url).rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.Client(base_url=self.base_url)
        self._keep_alive = keep_alive

    # ------------------------------------------------------------------ LLMProvider

    def complete(self, req: LLMRequest) -> LLMResponse:
        start = time.perf_counter()
        with transport_errors(self.name):
            response = self._client.post(
                self._url(), json=self._payload(req, stream=False), timeout=req.timeout_s
            )
        check_response(self.name, response)
        data = json_body(self.name, response)
        latency_ms = (time.perf_counter() - start) * 1000

        if "error" in data:
            raise LLMProviderError(f"ollama: {data['error']}", provider=self.name)
        message = data.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise LLMProviderError("ollama response has no message content", provider=self.name)

        return LLMResponse(
            text=message["content"],
            provider=self.name,
            model=req.model,
            usage=_usage(data),
            latency_ms=round(latency_ms, 2),
            finish_reason=data.get("done_reason"),
        )

    def stream(self, req: LLMRequest) -> Generator[str, None, Usage]:
        """Yield the reply as it is generated (Ollama streams NDJSON, one chunk per line).

        The final ``done`` chunk carries the token counts, which are returned.
        """
        usage = Usage()
        with (
            transport_errors(self.name),
            self._client.stream(
                "POST", self._url(), json=self._payload(req, stream=True), timeout=req.timeout_s
            ) as response,
        ):
            check_response(self.name, response)
            for line in response.iter_lines():
                if not line.strip():
                    continue
                chunk = _decode_line(line)
                if "error" in chunk:
                    raise LLMProviderError(f"ollama: {chunk['error']}", provider=self.name)
                content = (chunk.get("message") or {}).get("content")
                if content:
                    yield content
                if chunk.get("done"):
                    usage = _usage(chunk)
                    break
        return usage

    # ------------------------------------------------------------------ lifecycle

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OllamaProvider:
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
        # An injected client may have no base_url, so always send an absolute URL.
        return f"{self.base_url}{CHAT_PATH}"

    def _payload(self, req: LLMRequest, *, stream: bool) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": req.temperature}
        if req.max_tokens is not None:
            options["num_predict"] = req.max_tokens
        payload: dict[str, Any] = {
            "model": req.model,
            "messages": [{"role": m.role, "content": m.content} for m in req.messages],
            "stream": stream,
            "options": options,
        }
        if req.json_schema is not None:
            payload["format"] = req.json_schema
        elif req.json_mode:
            payload["format"] = "json"
        if self._keep_alive is not None:
            payload["keep_alive"] = self._keep_alive
        return payload


def _usage(data: dict[str, Any]) -> Usage:
    # Ollama omits prompt_eval_count when the whole prompt was served from its cache.
    return Usage(
        tokens_in=int(data.get("prompt_eval_count") or 0),
        tokens_out=int(data.get("eval_count") or 0),
    )


def _decode_line(line: str) -> dict[str, Any]:
    try:
        chunk = json.loads(line)
    except json.JSONDecodeError as exc:
        raise LLMProviderError(
            f"ollama stream sent a non-JSON line: {line[:200]!r}", provider="ollama"
        ) from exc
    if not isinstance(chunk, dict):
        raise LLMProviderError("ollama stream sent a non-object line", provider="ollama")
    return chunk
