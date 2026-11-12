"""Gemini adapter over the Gemini API's REST ``generateContent`` endpoint.

* System messages become ``systemInstruction``; ``assistant`` turns use Gemini's ``model``
  role.
* ``json_schema`` is sent as ``generationConfig.responseJsonSchema`` with
  ``responseMimeType: application/json``. Gemini accepts JSON Schema directly (``$defs``,
  ``$ref``, ``anyOf``, nullable types) and ignores keywords it does not support, so Pydantic
  schemas are sent unchanged. ``json_mode`` alone only sets the MIME type.
* Thinking models bill thought tokens as output, so ``tokens_out`` includes
  ``thoughtsTokenCount``; thought parts are never returned as text.
* A 429 carries its wait time in the body (``google.rpc.RetryInfo.retryDelay``), not in a
  ``Retry-After`` header; it is passed on to the router's backoff.

The REST API is used through httpx (like the OpenAI and Ollama adapters) rather than the
``google-genai`` SDK, so errors are classified by the shared helpers and tests can replay
recorded responses through ``httpx.MockTransport``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Generator
from types import TracebackType
from typing import Any

import httpx

from ledgerlens.core.config import get_settings
from ledgerlens.llm.providers._http import check_response, json_body, sse_data, transport_errors
from ledgerlens.llm.types import (
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMTransientError,
    Usage,
)

_ROLES = {"user": "user", "assistant": "model"}
_RETRY_INFO = "type.googleapis.com/google.rpc.RetryInfo"


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        settings = get_settings()
        self._api_key = api_key if api_key is not None else settings.google_api_key
        self.base_url = (base_url or settings.gemini_base_url).rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.Client()

    # ------------------------------------------------------------------ LLMProvider

    def complete(self, req: LLMRequest) -> LLMResponse:
        start = time.perf_counter()
        with transport_errors(self.name):
            response = self._client.post(
                self._url(req.model, "generateContent"),
                json=self._payload(req),
                headers=self._headers(),
                timeout=req.timeout_s,
            )
        self._check(response)
        data = json_body(self.name, response)
        latency_ms = (time.perf_counter() - start) * 1000

        candidate = _first_candidate(data)
        text = _text(candidate)
        finish_reason = candidate.get("finishReason")
        if not text and finish_reason not in (None, "STOP"):
            raise LLMProviderError(
                f"gemini returned no text (finishReason={finish_reason})", provider=self.name
            )

        return LLMResponse(
            text=text,
            provider=self.name,
            model=req.model,
            usage=_usage(data),
            latency_ms=round(latency_ms, 2),
            finish_reason=finish_reason,
        )

    def stream(self, req: LLMRequest) -> Generator[str, None, Usage]:
        """Yield text as it is generated (``streamGenerateContent`` with ``alt=sse``).

        Every chunk repeats the ``usageMetadata`` so far; the last one is returned.
        """
        usage = Usage()
        with (
            transport_errors(self.name),
            self._client.stream(
                "POST",
                self._url(req.model, "streamGenerateContent"),
                params={"alt": "sse"},
                json=self._payload(req),
                headers=self._headers(),
                timeout=req.timeout_s,
            ) as response,
        ):
            self._check(response)
            for event in sse_data(self.name, response):
                if "error" in event:
                    raise LLMProviderError(f"gemini: {event['error']}", provider=self.name)
                if event.get("usageMetadata"):
                    usage = _usage(event)
                candidates = event.get("candidates") or []
                if candidates and isinstance(candidates[0], dict):
                    text = _text(candidates[0])
                    if text:
                        yield text
        return usage

    # ------------------------------------------------------------------ lifecycle

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> GeminiProvider:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------ helpers

    def _url(self, model: str, method: str) -> str:
        return f"{self.base_url}/models/{model}:{method}"

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            raise LLMProviderError("GOOGLE_API_KEY is not set", provider=self.name)
        return {"x-goog-api-key": self._api_key}

    def _check(self, response: httpx.Response) -> None:
        try:
            check_response(self.name, response)
        except LLMTransientError as exc:
            if exc.retry_after_s is None:
                exc.retry_after_s = _retry_delay_s(response)
            raise

    def _payload(self, req: LLMRequest) -> dict[str, Any]:
        system = [m.content for m in req.messages if m.role == "system"]
        contents = [
            {"role": _ROLES[m.role], "parts": [{"text": m.content}]}
            for m in req.messages
            if m.role != "system"
        ]
        if not contents:
            raise LLMProviderError(
                "gemini needs at least one user or assistant message", provider=self.name
            )

        config: dict[str, Any] = {"temperature": req.temperature}
        if req.max_tokens is not None:
            config["maxOutputTokens"] = req.max_tokens
        if req.json_schema is not None or req.json_mode:
            config["responseMimeType"] = "application/json"
        if req.json_schema is not None:
            config["responseJsonSchema"] = req.json_schema

        payload: dict[str, Any] = {"contents": contents, "generationConfig": config}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system)}]}
        return payload


# --------------------------------------------------------------------------- parsing


def _first_candidate(data: dict[str, Any]) -> dict[str, Any]:
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        block_reason = (data.get("promptFeedback") or {}).get("blockReason")
        detail = f"prompt blocked: {block_reason}" if block_reason else "no candidates"
        raise LLMProviderError(f"gemini response has {detail}", provider="gemini")
    return candidates[0]


def _text(candidate: dict[str, Any]) -> str:
    parts = (candidate.get("content") or {}).get("parts") or []
    return "".join(
        part["text"]
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str) and not part.get("thought")
    )


def _usage(data: dict[str, Any]) -> Usage:
    usage = data.get("usageMetadata") or {}
    return Usage(
        tokens_in=int(usage.get("promptTokenCount") or 0),
        tokens_out=int(usage.get("candidatesTokenCount") or 0)
        + int(usage.get("thoughtsTokenCount") or 0),
    )


def _retry_delay_s(response: httpx.Response) -> float | None:
    """Read ``retryDelay`` (e.g. ``"37s"``) from a ``google.rpc.RetryInfo`` error detail."""
    try:
        error = response.json().get("error")
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        return None
    if not isinstance(error, dict):
        return None
    for detail in error.get("details") or []:
        if isinstance(detail, dict) and detail.get("@type") == _RETRY_INFO:
            delay = str(detail.get("retryDelay", ""))
            try:
                return max(float(delay.removesuffix("s")), 0.0)
            except ValueError:
                return None
    return None
