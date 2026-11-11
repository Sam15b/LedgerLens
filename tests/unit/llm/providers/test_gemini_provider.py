import json
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import httpx
import pytest

from ledgerlens.llm.providers.gemini_provider import GeminiProvider
from ledgerlens.llm.types import (
    LLMProviderError,
    LLMRequest,
    LLMTransientError,
    Message,
    Usage,
)

FIXTURES = Path(__file__).parents[3] / "fixtures" / "llm" / "gemini"
BASE_URL = "https://gemini.test/v1beta"
MODEL = "gemini-3.8-flash"


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


def make_provider(handler: Handler, *, api_key: str = "g-test") -> GeminiProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return GeminiProvider(api_key, base_url=BASE_URL, client=client)


def reply_with(body: str, *, status: int = 200) -> tuple[Handler, list[httpx.Request]]:
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(status, text=body)

    return handler, sent


def payload(request: httpx.Request) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(request.content)
    return data


def request(**overrides: Any) -> LLMRequest:
    fields: dict[str, Any] = {"messages": [Message.user("What is a PO?")], "model": MODEL}
    return LLMRequest(**(fields | overrides))


def candidate(parts: list[dict[str, Any]], finish_reason: str = "STOP") -> str:
    return json.dumps(
        {
            "candidates": [
                {"content": {"parts": parts, "role": "model"}, "finishReason": finish_reason}
            ]
        }
    )


# --------------------------------------------------------------------------- complete


def test_complete_maps_recorded_response() -> None:
    handler, sent = reply_with(recorded("generate.json"))

    response = make_provider(handler).complete(request())

    assert response.text.startswith("A purchase order is")
    assert (response.provider, response.model) == ("gemini", MODEL)
    assert response.usage == Usage(tokens_in=9, tokens_out=19 + 24)  # thinking billed as output
    assert response.finish_reason == "STOP"
    assert str(sent[0].url) == f"{BASE_URL}/models/{MODEL}:generateContent"
    assert sent[0].headers["x-goog-api-key"] == "g-test"


def test_complete_converts_roles_and_system_instruction() -> None:
    handler, sent = reply_with(recorded("generate.json"))
    messages = [
        Message.system("You are an AP clerk."),
        Message.system("Answer briefly."),
        Message.user("Hi"),
        Message.assistant("Hello"),
        Message.user("What is a PO?"),
    ]

    make_provider(handler).complete(request(messages=messages, temperature=0.3, max_tokens=50))

    body = payload(sent[0])
    assert body["systemInstruction"] == {
        "parts": [{"text": "You are an AP clerk.\n\nAnswer briefly."}]
    }
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
    assert body["contents"][1]["parts"] == [{"text": "Hello"}]
    assert body["generationConfig"] == {"temperature": 0.3, "maxOutputTokens": 50}


def test_json_schema_is_sent_as_response_json_schema() -> None:
    handler, sent = reply_with(recorded("generate_structured.json"))
    schema = {"type": "object", "properties": {"vendor": {"type": "string"}}, "$defs": {}}

    make_provider(handler).complete(request(json_schema=schema))

    config = payload(sent[0])["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseJsonSchema"] == schema  # sent unchanged


def test_json_mode_without_schema_only_sets_mime_type() -> None:
    handler, sent = reply_with(recorded("generate_structured.json"))

    make_provider(handler).complete(request(json_mode=True))

    config = payload(sent[0])["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert "responseJsonSchema" not in config


def test_thought_parts_are_not_returned_as_text() -> None:
    body = candidate([{"text": "Let me think.", "thought": True}, {"text": "Net 30"}])
    handler, _ = reply_with(body)

    assert make_provider(handler).complete(request()).text == "Net 30"


def test_blocked_prompt_is_a_provider_error() -> None:
    handler, _ = reply_with(json.dumps({"promptFeedback": {"blockReason": "SAFETY"}}))

    with pytest.raises(LLMProviderError, match="prompt blocked: SAFETY"):
        make_provider(handler).complete(request())


def test_no_text_with_non_stop_finish_reason_is_a_provider_error() -> None:
    handler, _ = reply_with(candidate([], finish_reason="RECITATION"))

    with pytest.raises(LLMProviderError, match="finishReason=RECITATION"):
        make_provider(handler).complete(request())


def test_rate_limit_is_transient_with_retry_delay_from_body() -> None:
    handler, _ = reply_with(recorded("error_429.json"), status=429)

    with pytest.raises(LLMTransientError) as info:
        make_provider(handler).complete(request())

    assert info.value.status_code == 429
    assert info.value.retry_after_s == 37.0
    assert "exceeded your current quota" in str(info.value)


def test_bad_request_is_a_provider_error() -> None:
    body = json.dumps(
        {"error": {"code": 400, "message": "API key not valid.", "status": "INVALID_ARGUMENT"}}
    )
    handler, _ = reply_with(body, status=400)

    with pytest.raises(LLMProviderError, match="API key not valid"):
        make_provider(handler).complete(request())


def test_timeout_is_transient() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(LLMTransientError, match="timed out"):
        make_provider(handler).complete(request())


def test_missing_api_key_fails_before_sending() -> None:
    handler, sent = reply_with(recorded("generate.json"))

    with pytest.raises(LLMProviderError, match="GOOGLE_API_KEY"):
        make_provider(handler, api_key="").complete(request())
    assert sent == []


def test_system_only_request_is_rejected() -> None:
    handler, sent = reply_with(recorded("generate.json"))

    with pytest.raises(LLMProviderError, match="at least one user"):
        make_provider(handler).complete(request(messages=[Message.system("Be terse.")]))
    assert sent == []


# --------------------------------------------------------------------------- stream


def test_stream_yields_text_deltas() -> None:
    handler, sent = reply_with(recorded("stream.sse"))

    chunks, usage = drain(make_provider(handler).stream(request()))

    assert chunks == ["Net", " 30"]
    assert usage == Usage(tokens_in=11, tokens_out=3)
    assert sent[0].url.path.endswith(f"/models/{MODEL}:streamGenerateContent")
    assert sent[0].url.params["alt"] == "sse"


def test_stream_error_event_raises() -> None:
    handler, _ = reply_with('data: {"error": {"code": 500, "message": "boom"}}\n\n')

    with pytest.raises(LLMProviderError, match="boom"):
        list(make_provider(handler).stream(request()))


def test_stream_http_error_is_classified() -> None:
    handler, _ = reply_with(recorded("error_429.json"), status=429)

    with pytest.raises(LLMTransientError):
        list(make_provider(handler).stream(request()))
