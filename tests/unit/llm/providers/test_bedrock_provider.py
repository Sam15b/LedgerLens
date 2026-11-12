"""Bedrock adapter contract tests.

Requests go through a real boto3 ``bedrock-runtime`` client with botocore's ``Stubber``, so
every request is validated against the Converse service model shipped with botocore;
nothing is sent over the network.
"""

import json
import sys
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any

import boto3
import pytest
from botocore.exceptions import EndpointConnectionError, NoCredentialsError, ReadTimeoutError
from botocore.stub import Stubber

from ledgerlens.llm.providers.bedrock_provider import BedrockProvider, bedrock_json_schema
from ledgerlens.llm.types import (
    LLMProviderError,
    LLMRequest,
    LLMTransientError,
    Message,
    Usage,
)

FIXTURES = Path(__file__).parents[3] / "fixtures" / "llm" / "bedrock"
CLAUDE = "global.anthropic.claude-sonnet-5-5"
NOVA = "us.amazon.nova-pro-v1:0"


def drain(stream: Generator[str, None, Usage]) -> tuple[list[str], Usage]:
    """Collect a provider stream's chunks and its returned usage."""
    chunks: list[str] = []
    while True:
        try:
            chunks.append(next(stream))
        except StopIteration as stop:
            return chunks, stop.value


def recorded(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
def stubbed() -> Iterator[tuple[BedrockProvider, Stubber]]:
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    with Stubber(client) as stubber:
        yield BedrockProvider(client=client), stubber
        stubber.assert_no_pending_responses()


def request(**overrides: Any) -> LLMRequest:
    fields: dict[str, Any] = {"messages": [Message.user("What is a PO?")], "model": CLAUDE}
    return LLMRequest(**(fields | overrides))


class FakeClient:
    """Stands in for boto3 where Stubber cannot (event streams, transport errors)."""

    def __init__(
        self, *, events: list[dict[str, Any]] | None = None, error: Exception | None = None
    ):
        self.events = events or []
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        response: dict[str, Any] = recorded("converse.json")
        return response

    def converse_stream(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return {"stream": iter(self.events)}


# --------------------------------------------------------------------------- complete


def test_complete_maps_recorded_response(stubbed: tuple[BedrockProvider, Stubber]) -> None:
    provider, stubber = stubbed
    stubber.add_response(
        "converse",
        recorded("converse.json"),
        {"modelId": CLAUDE, "messages": [{"role": "user", "content": [{"text": "What is a PO?"}]}]},
    )

    response = provider.complete(request())

    assert response.text.startswith("A purchase order is")
    assert (response.provider, response.model) == ("bedrock", CLAUDE)
    assert response.usage == Usage(tokens_in=14, tokens_out=22)
    assert response.finish_reason == "end_turn"


def test_request_shape_for_a_model_that_takes_temperature(
    stubbed: tuple[BedrockProvider, Stubber],
) -> None:
    provider, stubber = stubbed
    stubber.add_response(
        "converse",
        recorded("converse.json"),
        {
            "modelId": NOVA,
            "system": [{"text": "You are an AP clerk."}],
            "messages": [
                {"role": "user", "content": [{"text": "Hi"}, {"text": "What is a PO?"}]},
                {"role": "assistant", "content": [{"text": "A purchase order."}]},
                {"role": "user", "content": [{"text": "Thanks"}]},
            ],
            "inferenceConfig": {"temperature": 0.2, "maxTokens": 128},
        },
    )
    messages = [
        Message.system("You are an AP clerk."),
        Message.user("Hi"),
        Message.user("What is a PO?"),  # merged: Converse wants alternating turns
        Message.assistant("A purchase order."),
        Message.user("Thanks"),
    ]

    provider.complete(request(model=NOVA, messages=messages, temperature=0.2, max_tokens=128))


def test_temperature_is_left_out_for_anthropic_models(
    stubbed: tuple[BedrockProvider, Stubber],
) -> None:
    provider, stubber = stubbed
    stubber.add_response(
        "converse",
        recorded("converse.json"),
        {
            "modelId": CLAUDE,
            "messages": [{"role": "user", "content": [{"text": "What is a PO?"}]}],
            "inferenceConfig": {"maxTokens": 64},
        },
    )

    provider.complete(request(temperature=0.0, max_tokens=64))


def test_json_schema_uses_output_config_and_skips_reasoning(
    stubbed: tuple[BedrockProvider, Stubber],
) -> None:
    provider, stubber = stubbed
    schema = {"type": "object", "properties": {"vendor": {"type": "string"}}}
    stubber.add_response(
        "converse",
        recorded("converse_structured.json"),
        {
            "modelId": CLAUDE,
            "messages": [{"role": "user", "content": [{"text": "What is a PO?"}]}],
            "outputConfig": {
                "textFormat": {
                    "type": "json_schema",
                    "structure": {
                        "jsonSchema": {
                            "name": "Invoice",
                            "schema": json.dumps(bedrock_json_schema(schema)),
                        }
                    },
                }
            },
        },
    )

    response = provider.complete(request(json_schema=schema, schema_name="Invoice"))

    assert json.loads(response.text)["vendor"] == "Acme Supplies"


def test_blocked_reply_without_text_is_a_provider_error(
    stubbed: tuple[BedrockProvider, Stubber],
) -> None:
    provider, stubber = stubbed
    blocked = recorded("converse.json") | {
        "output": {"message": {"role": "assistant", "content": []}},
        "stopReason": "guardrail_intervened",
    }
    stubber.add_response("converse", blocked)

    with pytest.raises(LLMProviderError, match="guardrail_intervened"):
        provider.complete(request())


# --------------------------------------------------------------------------- errors


@pytest.mark.parametrize(
    ("code", "status"),
    [
        ("ThrottlingException", 429),
        ("ServiceUnavailableException", 503),
        ("ModelNotReadyException", 429),
        ("InternalServerException", 500),
    ],
)
def test_retryable_client_errors_are_transient(
    stubbed: tuple[BedrockProvider, Stubber], code: str, status: int
) -> None:
    provider, stubber = stubbed
    stubber.add_client_error("converse", code, "slow down", http_status_code=status)

    with pytest.raises(LLMTransientError, match=code) as info:
        provider.complete(request())
    assert info.value.status_code == status


@pytest.mark.parametrize(
    ("code", "status"),
    [
        ("ValidationException", 400),
        ("AccessDeniedException", 403),
        ("ResourceNotFoundException", 404),
    ],
)
def test_other_client_errors_are_provider_errors(
    stubbed: tuple[BedrockProvider, Stubber], code: str, status: int
) -> None:
    provider, stubber = stubbed
    stubber.add_client_error("converse", code, "no", http_status_code=status)

    with pytest.raises(LLMProviderError, match=code):
        provider.complete(request())


@pytest.mark.parametrize(
    "error",
    [
        ReadTimeoutError(endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com"),
        EndpointConnectionError(endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com"),
    ],
)
def test_transport_errors_are_transient(error: Exception) -> None:
    provider = BedrockProvider(client=FakeClient(error=error))

    with pytest.raises(LLMTransientError, match="connection failed"):
        provider.complete(request())


def test_missing_credentials_is_a_provider_error() -> None:
    provider = BedrockProvider(client=FakeClient(error=NoCredentialsError()))

    with pytest.raises(LLMProviderError, match="credentials"):
        provider.complete(request())


def test_unrelated_exceptions_propagate() -> None:
    provider = BedrockProvider(client=FakeClient(error=KeyError("bug")))

    with pytest.raises(KeyError):
        provider.complete(request())


def test_missing_boto3_is_a_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "boto3", None)

    with pytest.raises(LLMProviderError, match="boto3 is not installed"):
        BedrockProvider().complete(request())


def test_system_only_request_is_rejected() -> None:
    client = FakeClient()

    with pytest.raises(LLMProviderError, match="at least one user"):
        BedrockProvider(client=client).complete(request(messages=[Message.system("Be terse.")]))
    assert client.calls == []


# --------------------------------------------------------------------------- stream


def test_stream_yields_text_deltas_only() -> None:
    client = FakeClient(events=recorded("converse_stream.json"))

    chunks, usage = drain(BedrockProvider(client=client).stream(request()))

    assert chunks == ["Net", " 30"]
    assert usage == Usage(tokens_in=11, tokens_out=9)
    assert client.calls[0]["modelId"] == CLAUDE


def test_stream_throttling_event_is_transient() -> None:
    events = [{"messageStart": {"role": "assistant"}}, {"throttlingException": {"message": "slow"}}]

    with pytest.raises(LLMTransientError, match="slow"):
        list(BedrockProvider(client=FakeClient(events=events)).stream(request()))


def test_stream_model_error_event_is_a_provider_error() -> None:
    events = [{"modelStreamErrorException": {"message": "bad output"}}]

    with pytest.raises(LLMProviderError, match="bad output"):
        list(BedrockProvider(client=FakeClient(events=events)).stream(request()))


# --------------------------------------------------------------------------- schema


def test_bedrock_json_schema_keeps_only_supported_keywords() -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "pattern": "^(?!x)", "maxLength": 5},
            "amount": {
                "anyOf": [
                    {"type": "number", "minimum": 0},
                    {"type": "string", "format": "decimal"},
                ],
                "default": None,
            },
            "issued": {"type": "string", "format": "date"},
            "lines": {"type": "array", "items": {"$ref": "#/$defs/Line"}, "minItems": 2},
            "doc": {"oneOf": [{"$ref": "#/$defs/Line"}], "discriminator": {"propertyName": "k"}},
        },
        "$defs": {"Line": {"type": "object", "properties": {"qty": {"type": "integer"}}}},
    }

    out = bedrock_json_schema(schema)

    props = out["properties"]
    assert out["additionalProperties"] is False
    assert out["required"] == ["pattern", "amount", "issued", "lines", "doc"]
    assert props["pattern"] == {"type": "string"}  # property named "pattern" survives
    assert props["amount"] == {"anyOf": [{"type": "number"}, {"type": "string"}]}
    assert props["issued"]["format"] == "date"
    assert props["lines"]["minItems"] == 1
    assert props["doc"] == {"anyOf": [{"$ref": "#/$defs/Line"}]}
    assert out["$defs"]["Line"]["additionalProperties"] is False
    assert schema["properties"]["lines"]["minItems"] == 2  # input left untouched
