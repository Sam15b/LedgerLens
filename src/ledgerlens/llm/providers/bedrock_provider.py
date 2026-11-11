"""AWS Bedrock adapter over the ``bedrock-runtime`` **Converse API** (via boto3).

Converse is one request shape for every model family on Bedrock (Anthropic, Llama,
Mistral, Amazon Nova, ...), so ``bedrock/<model-id>`` works for any of them; the model id
can be an inference profile such as ``global.anthropic.claude-sonnet-5-5``.

* Credentials come from the standard AWS chain (env vars, ``~/.aws``, SSO, instance role);
  the region from ``AWS_REGION``.
* botocore's own retries are switched off: the router already retries transient errors
  with the backoff in ``models.yaml``, and two retry loops would multiply the wait.
* ``json_schema`` uses Bedrock structured outputs (``outputConfig.textFormat``). Bedrock
  needs every object closed and rejects numeric/string-length constraints, lookaround
  regexes and ``oneOf``, so the schema is rewritten by :func:`bedrock_json_schema`.
  Converse has no schema-less JSON mode; with ``json_mode`` alone the prompt must ask for
  JSON.
* Newer Anthropic models reject ``temperature``; it is left out for model ids containing
  one of ``omit_temperature_for``.
* Reasoning blocks in the reply are skipped; only ``text`` blocks become the answer.

boto3 is imported on first use, so the router still loads (and falls back to the next
model) on an install without it.
"""

from __future__ import annotations

import json
import time
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from typing import Any, Protocol

from ledgerlens.core.config import get_config, get_settings
from ledgerlens.llm.providers._http import is_transient_status
from ledgerlens.llm.providers._schema import SchemaNode, strict_json_schema, walk_schema
from ledgerlens.llm.types import (
    LLMError,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMTransientError,
    Message,
    Usage,
)

# The first request with a new schema compiles a grammar, which AWS says can take ~90 s.
MIN_READ_TIMEOUT_S = 120.0
CONNECT_TIMEOUT_S = 10.0

_TRANSIENT_CODES = frozenset(
    {
        "ThrottlingException",
        "ServiceUnavailableException",
        "InternalServerException",
        "ModelNotReadyException",
        "ModelTimeoutException",
    }
)
# ConverseStream reports errors as events keyed by the exception name in camelCase.
_TRANSIENT_STREAM_EVENTS = frozenset(
    {"throttlingException", "serviceUnavailableException", "internalServerException"}
)
_STREAM_ERROR_EVENTS = _TRANSIENT_STREAM_EVENTS | {
    "modelStreamErrorException",
    "validationException",
}
_BLOCKED_STOP_REASONS = frozenset({"content_filtered", "guardrail_intervened"})

# JSON-schema keywords Bedrock structured outputs rejects (or Pydantic emits but Bedrock
# does not know). Values are still validated by Pydantic after the reply comes back.
_UNSUPPORTED_KEYWORDS = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "maxItems",
    "minProperties",
    "maxProperties",
    "uniqueItems",
    "pattern",  # Pydantic's Decimal pattern uses lookahead, which Bedrock rejects
    "discriminator",
)
_SUPPORTED_FORMATS = frozenset(
    {"date-time", "time", "date", "duration", "email", "hostname", "uri", "ipv4", "ipv6", "uuid"}
)


class ConverseClient(Protocol):
    """The two ``bedrock-runtime`` client methods this adapter uses."""

    def converse(self, **kwargs: Any) -> dict[str, Any]: ...

    def converse_stream(self, **kwargs: Any) -> dict[str, Any]: ...


class BedrockProvider:
    name = "bedrock"

    def __init__(
        self,
        *,
        region: str | None = None,
        client: ConverseClient | None = None,
        read_timeout_s: float | None = None,
        omit_temperature_for: Collection[str] = ("anthropic.",),
    ) -> None:
        """``client`` is injectable for tests; otherwise a boto3 client is built on first
        use with ``read_timeout_s`` (default: ``models.yaml`` ``timeout_s``, at least
        :data:`MIN_READ_TIMEOUT_S`)."""
        self.region = region or get_settings().aws_region
        self._client = client
        self._read_timeout_s = read_timeout_s or max(
            get_config().models.timeout_s, MIN_READ_TIMEOUT_S
        )
        self._omit_temperature_for = tuple(omit_temperature_for)

    # ------------------------------------------------------------------ LLMProvider

    def complete(self, req: LLMRequest) -> LLMResponse:
        start = time.perf_counter()
        with _bedrock_errors():
            data = self._runtime().converse(**self._request(req))
        latency_ms = (time.perf_counter() - start) * 1000

        content = ((data.get("output") or {}).get("message") or {}).get("content")
        if not isinstance(content, list):
            raise LLMProviderError("bedrock response has no message content", provider=self.name)
        text = "".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
        stop_reason = data.get("stopReason")
        if not text and stop_reason in _BLOCKED_STOP_REASONS:
            raise LLMProviderError(
                f"bedrock returned no text (stopReason={stop_reason})", provider=self.name
            )

        return LLMResponse(
            text=text,
            provider=self.name,
            model=req.model,
            usage=_usage(data.get("usage")),
            latency_ms=round(latency_ms, 2),
            finish_reason=stop_reason,
        )

    def stream(self, req: LLMRequest) -> Iterator[str]:
        """Yield text deltas from ``ConverseStream``."""
        with _bedrock_errors():
            events = self._runtime().converse_stream(**self._request(req)).get("stream") or []
            for event in events:
                _raise_stream_error(event)
                delta = (event.get("contentBlockDelta") or {}).get("delta") or {}
                text = delta.get("text")
                if isinstance(text, str) and text:
                    yield text

    # ------------------------------------------------------------------ helpers

    def _runtime(self) -> ConverseClient:
        if self._client is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError as exc:
                raise LLMProviderError(
                    "boto3 is not installed (pip install boto3)", provider=self.name
                ) from exc
            config = Config(
                retries={"total_max_attempts": 1, "mode": "standard"},
                connect_timeout=CONNECT_TIMEOUT_S,
                read_timeout=self._read_timeout_s,
            )
            self._client = boto3.client("bedrock-runtime", region_name=self.region, config=config)
        return self._client

    def _request(self, req: LLMRequest) -> dict[str, Any]:
        inference: dict[str, Any] = {}
        if not any(marker in req.model for marker in self._omit_temperature_for):
            inference["temperature"] = req.temperature
        if req.max_tokens is not None:
            inference["maxTokens"] = req.max_tokens

        request: dict[str, Any] = {"modelId": req.model, "messages": _messages(req.messages)}
        system = [{"text": m.content} for m in req.messages if m.role == "system"]
        if system:
            request["system"] = system
        if inference:
            request["inferenceConfig"] = inference
        if req.json_schema is not None:
            request["outputConfig"] = {
                "textFormat": {
                    "type": "json_schema",
                    "structure": {
                        "jsonSchema": {
                            "name": req.schema_name,
                            "schema": json.dumps(bedrock_json_schema(req.json_schema)),
                        }
                    },
                }
            }
        return request


# --------------------------------------------------------------------------- schema


def bedrock_json_schema(schema: SchemaNode) -> SchemaNode:
    """Rewrite a Pydantic JSON schema into the subset Bedrock structured outputs accepts."""
    strict = strict_json_schema(schema, drop_keywords=_UNSUPPORTED_KEYWORDS)

    def visit(node: SchemaNode) -> None:
        if "oneOf" in node:
            node["anyOf"] = node.pop("oneOf")
        if node.get("format") not in (None, *_SUPPORTED_FORMATS):
            node.pop("format")
        min_items = node.get("minItems")
        if isinstance(min_items, int) and min_items > 1:
            node["minItems"] = 1

    walk_schema(strict, visit)
    return strict


# --------------------------------------------------------------------------- conversion


def _messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Converse wants alternating turns, so consecutive same-role messages are merged."""
    turns: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            continue
        block = {"text": message.content}
        if turns and turns[-1]["role"] == message.role:
            turns[-1]["content"].append(block)
        else:
            turns.append({"role": message.role, "content": [block]})
    if not turns:
        raise LLMProviderError(
            "bedrock needs at least one user or assistant message", provider="bedrock"
        )
    return turns


def _usage(usage: Any) -> Usage:
    usage = usage if isinstance(usage, dict) else {}
    return Usage(
        tokens_in=int(usage.get("inputTokens") or 0),
        tokens_out=int(usage.get("outputTokens") or 0),
    )


# --------------------------------------------------------------------------- errors


@contextmanager
def _bedrock_errors() -> Iterator[None]:
    """Map botocore exceptions to the router's error classes."""
    try:
        yield
    except LLMError:
        raise
    except Exception as exc:
        mapped = _map_error(exc)
        if mapped is None:
            raise
        raise mapped from exc


def _map_error(exc: Exception) -> LLMError | None:
    try:
        from botocore.exceptions import (
            BotoCoreError,
            ClientError,
            HTTPClientError,
        )
        from botocore.exceptions import ConnectionError as BotoConnectionError
    except ImportError:  # pragma: no cover - a client exists, so botocore does too
        return None

    if isinstance(exc, ClientError):
        error = exc.response.get("Error") or {}
        code = str(error.get("Code") or "")
        status = (exc.response.get("ResponseMetadata") or {}).get("HTTPStatusCode")
        message = f"bedrock {code or 'error'}: {error.get('Message') or exc}"
        if code in _TRANSIENT_CODES or (isinstance(status, int) and is_transient_status(status)):
            return LLMTransientError(message, provider="bedrock", status_code=status)
        return LLMProviderError(message, provider="bedrock", status_code=status)
    if isinstance(exc, HTTPClientError | BotoConnectionError):
        return LLMTransientError(f"bedrock connection failed: {exc}", provider="bedrock")
    if isinstance(exc, BotoCoreError):  # no credentials, bad region, bad parameters, ...
        return LLMProviderError(f"bedrock: {exc}", provider="bedrock")
    return None


def _raise_stream_error(event: dict[str, Any]) -> None:
    for key in _STREAM_ERROR_EVENTS & event.keys():
        detail = (event[key] or {}).get("message") or key
        message = f"bedrock {key}: {detail}"
        if key in _TRANSIENT_STREAM_EVENTS:
            raise LLMTransientError(message, provider="bedrock")
        raise LLMProviderError(message, provider="bedrock")
