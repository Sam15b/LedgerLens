import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from ledgerlens.llm.providers.openai_provider import OpenAIProvider, strict_json_schema
from ledgerlens.llm.types import (
    LLMProviderError,
    LLMRequest,
    LLMTransientError,
    Message,
    Usage,
)

FIXTURES = Path(__file__).parents[2] / "fixtures" / "llm" / "openai"
BASE_URL = "https://openai.test/v1"

Handler = Callable[[httpx.Request], httpx.Response]


def recorded(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def make_provider(handler: Handler, *, api_key: str = "sk-test", **kwargs: Any) -> OpenAIProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAIProvider(api_key, base_url=BASE_URL, client=client, **kwargs)


def reply_with(
    body: str, *, status: int = 200, headers: dict[str, str] | None = None
) -> tuple[Handler, list[httpx.Request]]:
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{BASE_URL}/chat/completions"
        sent.append(request)
        return httpx.Response(status, text=body, headers=headers)

    return handler, sent


def payload(request: httpx.Request) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(request.content)
    return data


def request(**overrides: Any) -> LLMRequest:
    fields: dict[str, Any] = {"messages": [Message.user("What is a PO?")], "model": "gpt-4o"}
    return LLMRequest(**(fields | overrides))


class Invoice(BaseModel):
    vendor: str
    total: float
    currency: str = "USD"
    po_number: str | None = None


# --------------------------------------------------------------------------- complete


def test_complete_maps_recorded_response() -> None:
    handler, sent = reply_with(recorded("chat.json"))

    response = make_provider(handler).complete(request())

    assert response.text.startswith("A purchase order is")
    assert (response.provider, response.model) == ("openai", "gpt-4o")
    assert response.usage == Usage(tokens_in=24, tokens_out=15)
    assert response.finish_reason == "stop"
    assert sent[0].headers["authorization"] == "Bearer sk-test"


def test_complete_sends_chat_payload() -> None:
    handler, sent = reply_with(recorded("chat.json"))

    make_provider(handler).complete(request(temperature=0.3, max_tokens=50))

    assert payload(sent[0]) == {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "What is a PO?"}],
        "temperature": 0.3,
        "max_completion_tokens": 50,
    }


def test_json_mode_uses_json_object_format() -> None:
    handler, sent = reply_with(recorded("chat_structured.json"))

    make_provider(handler).complete(request(json_mode=True))

    assert payload(sent[0])["response_format"] == {"type": "json_object"}


def test_json_schema_uses_strict_structured_outputs() -> None:
    handler, sent = reply_with(recorded("chat_structured.json"))
    schema = Invoice.model_json_schema()

    response = make_provider(handler).complete(
        request(json_schema=schema, schema_name="Invoice", json_mode=True)
    )

    fmt = payload(sent[0])["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "Invoice"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"] == strict_json_schema(schema)
    assert Invoice.model_validate_json(response.text).vendor == "Acme Supplies"


def test_non_strict_mode_sends_schema_unchanged() -> None:
    handler, sent = reply_with(recorded("chat_structured.json"))
    schema = Invoice.model_json_schema()

    make_provider(handler, strict_schema=False).complete(request(json_schema=schema))

    fmt = payload(sent[0])["response_format"]["json_schema"]
    assert (fmt["strict"], fmt["schema"]) == (False, schema)


def test_missing_api_key_is_a_provider_error_without_a_request() -> None:
    handler, sent = reply_with(recorded("chat.json"))

    with pytest.raises(LLMProviderError, match="OPENAI_API_KEY"):
        make_provider(handler, api_key="").complete(request())

    assert sent == []


def test_refusal_is_a_provider_error() -> None:
    data = json.loads(recorded("chat.json"))
    data["choices"][0]["message"] |= {"content": None, "refusal": "I can't help with that."}
    handler, _ = reply_with(json.dumps(data))

    with pytest.raises(LLMProviderError, match="refused"):
        make_provider(handler).complete(request())


@pytest.mark.parametrize("body", ['{"choices": []}', '{"choices": [{"message": {}}]}', "[]"])
def test_malformed_body_is_a_provider_error(body: str) -> None:
    handler, _ = reply_with(body)

    with pytest.raises(LLMProviderError):
        make_provider(handler).complete(request())


def openai_error(message: str, code: str | None, type_: str = "invalid_request_error") -> str:
    return json.dumps({"error": {"message": message, "type": type_, "code": code}})


def test_bad_key_is_a_provider_error() -> None:
    handler, _ = reply_with(
        openai_error("Incorrect API key provided", "invalid_api_key"), status=401
    )

    with pytest.raises(LLMProviderError, match="Incorrect API key") as excinfo:
        make_provider(handler).complete(request())

    assert excinfo.value.status_code == 401


def test_rate_limit_is_transient_with_retry_after() -> None:
    body = openai_error("Rate limit reached for gpt-4o", "rate_limit_exceeded", "requests")
    handler, _ = reply_with(body, status=429, headers={"retry-after": "1.5"})

    with pytest.raises(LLMTransientError, match="Rate limit") as excinfo:
        make_provider(handler).complete(request())

    assert excinfo.value.retry_after_s == 1.5


def test_exhausted_quota_is_not_retried() -> None:
    body = openai_error("You exceeded your current quota", "insufficient_quota")
    handler, _ = reply_with(body, status=429)

    with pytest.raises(LLMProviderError, match="quota") as excinfo:
        make_provider(handler).complete(request())

    assert excinfo.value.status_code == 429


def test_server_error_is_transient() -> None:
    handler, _ = reply_with(
        openai_error("The server had an error", None, "server_error"), status=500
    )

    with pytest.raises(LLMTransientError):
        make_provider(handler).complete(request())


# --------------------------------------------------------------------------- stream


def test_stream_yields_recorded_deltas() -> None:
    handler, sent = reply_with(recorded("chat_stream.sse"))

    chunks = list(make_provider(handler).stream(request()))

    assert chunks == ["Net", " 30"]
    assert payload(sent[0])["stream"] is True


def test_stream_error_event_is_a_provider_error() -> None:
    handler, _ = reply_with('data: {"error": {"message": "overloaded"}}\n\n')

    with pytest.raises(LLMProviderError, match="overloaded"):
        list(make_provider(handler).stream(request()))


def test_stream_http_error_is_classified() -> None:
    handler, _ = reply_with(openai_error("busy", None, "server_error"), status=503)

    with pytest.raises(LLMTransientError):
        list(make_provider(handler).stream(request()))


# --------------------------------------------------------------------------- strict schema


def test_strict_schema_closes_objects_and_requires_every_field() -> None:
    class Line(BaseModel):
        sku: str
        qty: int = 1

    class Order(BaseModel):
        lines: list[Line]
        note: str | None = None

    strict = strict_json_schema(Order.model_json_schema())

    assert strict["additionalProperties"] is False
    assert strict["required"] == ["lines", "note"]
    line = strict["$defs"]["Line"]
    assert line["additionalProperties"] is False
    assert line["required"] == ["sku", "qty"]
    assert "default" not in line["properties"]["qty"]
    assert {"type": "null"} in strict["properties"]["note"]["anyOf"]  # optional stays nullable


def test_strict_schema_keeps_a_property_named_default() -> None:
    class Setting(BaseModel):
        default: str
        value: int = 0

    strict = strict_json_schema(Setting.model_json_schema())

    assert set(strict["properties"]) == {"default", "value"}
    assert strict["required"] == ["default", "value"]


def test_strict_schema_does_not_mutate_input() -> None:
    schema = Invoice.model_json_schema()
    before = json.dumps(schema, sort_keys=True)

    strict_json_schema(schema)

    assert json.dumps(schema, sort_keys=True) == before
