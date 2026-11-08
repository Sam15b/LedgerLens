import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from ledgerlens.core.config import get_settings
from ledgerlens.llm.providers.ollama_provider import OllamaProvider
from ledgerlens.llm.types import (
    LLMProviderError,
    LLMRequest,
    LLMTransientError,
    Message,
    Usage,
)

FIXTURES = Path(__file__).parents[2] / "fixtures" / "llm" / "ollama"
BASE_URL = "http://ollama.test:11434"

Handler = Callable[[httpx.Request], httpx.Response]


def recorded(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def make_provider(handler: Handler, **kwargs: Any) -> OllamaProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OllamaProvider(BASE_URL, client=client, **kwargs)


def reply_with(
    body: str, *, status: int = 200, headers: dict[str, str] | None = None
) -> tuple[Handler, list[dict[str, Any]]]:
    """A handler that returns ``body`` and records every request payload it receives."""
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == f"{BASE_URL}/api/chat"
        sent.append(json.loads(request.content))
        return httpx.Response(status, text=body, headers=headers)

    return handler, sent


def request(**overrides: Any) -> LLMRequest:
    fields: dict[str, Any] = {
        "messages": [Message.system("Be brief."), Message.user("What is an invoice?")],
        "model": "llama3:8b",
    }
    return LLMRequest(**(fields | overrides))


# --------------------------------------------------------------------------- complete


def test_complete_maps_recorded_response() -> None:
    handler, _ = reply_with(recorded("chat.json"))

    response = make_provider(handler).complete(request())

    assert response.text.startswith("An invoice is a request for payment")
    assert response.provider == "ollama"
    assert response.model == "llama3:8b"
    assert response.usage == Usage(tokens_in=26, tokens_out=17)
    assert response.finish_reason == "stop"
    assert response.latency_ms >= 0
    assert response.cost_usd is None  # the router prices calls, not the adapter


def test_complete_sends_chat_payload() -> None:
    handler, sent = reply_with(recorded("chat.json"))

    make_provider(handler).complete(request(temperature=0.2))

    assert sent == [
        {
            "model": "llama3:8b",
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "What is an invoice?"},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        }
    ]


def test_max_tokens_and_keep_alive_are_passed_through() -> None:
    handler, sent = reply_with(recorded("chat.json"))

    make_provider(handler, keep_alive="10m").complete(request(max_tokens=64))

    assert sent[0]["options"] == {"temperature": 0.0, "num_predict": 64}
    assert sent[0]["keep_alive"] == "10m"


def test_json_mode_sets_format_json() -> None:
    handler, sent = reply_with(recorded("chat_json.json"))

    response = make_provider(handler).complete(request(json_mode=True))

    assert sent[0]["format"] == "json"
    assert json.loads(response.text) == {"vendor": "Acme Supplies", "total": 1250.0}


def test_json_schema_is_sent_as_format() -> None:
    schema = {"type": "object", "properties": {"total": {"type": "number"}}}
    handler, sent = reply_with(recorded("chat_json.json"))

    make_provider(handler).complete(request(json_mode=True, json_schema=schema))

    assert sent[0]["format"] == schema


def test_missing_prompt_eval_count_counts_as_zero_input_tokens() -> None:
    handler, _ = reply_with(recorded("chat_json.json"))

    response = make_provider(handler).complete(request())

    assert response.usage == Usage(tokens_in=0, tokens_out=21)


def test_unknown_model_is_a_provider_error() -> None:
    body = json.dumps({"error": "model 'llama9' not found, try pulling it first"})
    handler, _ = reply_with(body, status=404)

    with pytest.raises(LLMProviderError, match="not found") as excinfo:
        make_provider(handler).complete(request(model="llama9"))

    assert excinfo.value.status_code == 404
    assert excinfo.value.provider == "ollama"


@pytest.mark.parametrize("status", [500, 502, 503, 408])
def test_server_errors_are_transient(status: int) -> None:
    handler, _ = reply_with(json.dumps({"error": "busy"}), status=status)

    with pytest.raises(LLMTransientError) as excinfo:
        make_provider(handler).complete(request())

    assert excinfo.value.status_code == status


def test_rate_limit_keeps_retry_after() -> None:
    handler, _ = reply_with("slow down", status=429, headers={"Retry-After": "2"})

    with pytest.raises(LLMTransientError, match="slow down") as excinfo:
        make_provider(handler).complete(request())

    assert excinfo.value.retry_after_s == 2.0


@pytest.mark.parametrize(
    "exc", [httpx.ConnectError("refused"), httpx.ReadTimeout("slow"), httpx.ReadError("reset")]
)
def test_transport_failures_are_transient(exc: httpx.TransportError) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    with pytest.raises(LLMTransientError, match="ollama"):
        make_provider(handler).complete(request())


@pytest.mark.parametrize(
    "body", ["<html>proxy error</html>", "[]", json.dumps({"done": True}), '{"error": "boom"}']
)
def test_malformed_body_is_a_provider_error(body: str) -> None:
    handler, _ = reply_with(body)

    with pytest.raises(LLMProviderError):
        make_provider(handler).complete(request())


# --------------------------------------------------------------------------- stream


def test_stream_yields_recorded_chunks_until_done() -> None:
    handler, sent = reply_with(recorded("chat_stream.ndjson"))

    chunks = list(make_provider(handler).stream(request()))

    assert chunks == ["Net", " 30", " days."]
    assert sent[0]["stream"] is True


def test_stream_error_line_is_a_provider_error() -> None:
    body = '{"message":{"content":"Hi"},"done":false}\n{"error":"out of memory"}\n'
    handler, _ = reply_with(body)

    with pytest.raises(LLMProviderError, match="out of memory"):
        list(make_provider(handler).stream(request()))


def test_stream_http_error_is_classified() -> None:
    handler, _ = reply_with('{"error":"model not found"}', status=404)

    with pytest.raises(LLMProviderError, match="model not found"):
        list(make_provider(handler).stream(request()))


def test_stream_non_json_line_is_a_provider_error() -> None:
    handler, _ = reply_with("not json\n")

    with pytest.raises(LLMProviderError, match="non-JSON"):
        list(make_provider(handler).stream(request()))


# --------------------------------------------------------------------------- lifecycle


@pytest.fixture
def fresh_settings() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.usefixtures("fresh_settings")
def test_base_url_defaults_to_settings_and_drops_trailing_slash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434/")

    with OllamaProvider() as provider:
        assert provider.base_url == "http://gpu-box:11434"


def test_injected_client_is_not_closed() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))

    with OllamaProvider(BASE_URL, client=client):
        pass

    assert not client.is_closed
