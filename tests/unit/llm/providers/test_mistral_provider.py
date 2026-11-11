import json
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from ledgerlens.llm.providers.mistral_provider import MistralProvider
from ledgerlens.llm.types import (
    LLMProviderError,
    LLMRequest,
    LLMTransientError,
    Message,
    Usage,
)

FIXTURES = Path(__file__).parents[3] / "fixtures" / "llm" / "mistral"
BASE_URL = "https://mistral.test/v1"
MODEL = "mistral-small-2603"


def drain(stream: Generator[str, None, Usage]) -> tuple[list[str], Usage]:
    """Collect a provider stream's chunks and its returned usage."""
    chunks: list[str] = []
    while True:
        try:
            chunks.append(next(stream))
        except StopIteration as stop:
            return chunks, stop.value


Handler = Callable[[httpx.Request], httpx.Response]


def recorded(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def make_provider(handler: Handler, *, api_key: str = "m-test") -> MistralProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return MistralProvider(api_key, base_url=BASE_URL, client=client)


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
    fields: dict[str, Any] = {"messages": [Message.user("What is a PO?")], "model": MODEL}
    return LLMRequest(**(fields | overrides))


def message(content: Any) -> str:
    return json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]})


class Invoice(BaseModel):
    vendor: str
    total: float
    po_number: str | None = None


# --------------------------------------------------------------------------- complete


def test_complete_maps_recorded_response() -> None:
    handler, sent = reply_with(recorded("chat.json"))

    response = make_provider(handler).complete(request())

    assert response.text.startswith("A purchase order is")
    assert (response.provider, response.model) == ("mistral", MODEL)
    assert response.usage == Usage(tokens_in=12, tokens_out=21)
    assert response.finish_reason == "stop"
    assert sent[0].headers["authorization"] == "Bearer m-test"


def test_complete_sends_chat_payload() -> None:
    handler, sent = reply_with(recorded("chat.json"))
    messages = [Message.system("Be brief."), Message.user("What is a PO?")]

    make_provider(handler).complete(request(messages=messages, temperature=0.2, max_tokens=64))

    assert payload(sent[0]) == {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "What is a PO?"},
        ],
        "temperature": 0.2,
        "max_tokens": 64,
    }


def test_json_schema_uses_strict_structured_output() -> None:
    handler, sent = reply_with(recorded("chat_structured.json"))

    make_provider(handler).complete(
        request(json_schema=Invoice.model_json_schema(), schema_name="Invoice")
    )

    response_format = payload(sent[0])["response_format"]
    assert response_format["type"] == "json_schema"
    spec = response_format["json_schema"]
    assert (spec["name"], spec["strict"]) == ("Invoice", True)
    assert spec["schema"]["additionalProperties"] is False
    assert spec["schema"]["required"] == ["vendor", "total", "po_number"]
    assert "default" not in spec["schema"]["properties"]["po_number"]


def test_json_mode_uses_json_object() -> None:
    handler, sent = reply_with(recorded("chat_structured.json"))

    make_provider(handler).complete(request(json_mode=True))

    assert payload(sent[0])["response_format"] == {"type": "json_object"}


def test_chunked_content_keeps_only_text_chunks() -> None:
    content = [
        {"type": "thinking", "thinking": [{"type": "text", "text": "Hmm."}]},
        {"type": "text", "text": "Net "},
        {"type": "text", "text": "30"},
    ]
    handler, _ = reply_with(message(content))

    assert make_provider(handler).complete(request()).text == "Net 30"


def test_missing_content_is_a_provider_error() -> None:
    handler, _ = reply_with(message(None))

    with pytest.raises(LLMProviderError, match="no message content"):
        make_provider(handler).complete(request())


def test_no_choices_is_a_provider_error() -> None:
    handler, _ = reply_with(json.dumps({"choices": []}))

    with pytest.raises(LLMProviderError, match="no choices"):
        make_provider(handler).complete(request())


def test_rate_limit_is_transient() -> None:
    body = json.dumps({"object": "error", "message": "Requests rate limit exceeded"})
    handler, _ = reply_with(body, status=429, headers={"retry-after": "3"})

    with pytest.raises(LLMTransientError) as info:
        make_provider(handler).complete(request())

    assert (info.value.status_code, info.value.retry_after_s) == (429, 3.0)


def test_unauthorized_is_a_provider_error() -> None:
    handler, _ = reply_with(json.dumps({"message": "Unauthorized"}), status=401)

    with pytest.raises(LLMProviderError, match="401"):
        make_provider(handler).complete(request())


def test_missing_api_key_fails_before_sending() -> None:
    handler, sent = reply_with(recorded("chat.json"))

    with pytest.raises(LLMProviderError, match="MISTRAL_API_KEY"):
        make_provider(handler, api_key="").complete(request())
    assert sent == []


# --------------------------------------------------------------------------- stream


def test_stream_yields_content_deltas() -> None:
    handler, sent = reply_with(recorded("chat_stream.sse"))

    chunks, usage = drain(make_provider(handler).stream(request()))

    assert chunks == ["Net", " 30"]
    assert usage == Usage(tokens_in=11, tokens_out=3)
    assert payload(sent[0])["stream"] is True


def test_stream_rejects_non_json_event() -> None:
    handler, _ = reply_with("data: not-json\n\n")

    with pytest.raises(LLMProviderError, match="non-JSON event"):
        list(make_provider(handler).stream(request()))


def test_stream_error_event_raises() -> None:
    handler, _ = reply_with('data: {"error": "overloaded"}\n\n')

    with pytest.raises(LLMProviderError, match="overloaded"):
        list(make_provider(handler).stream(request()))
