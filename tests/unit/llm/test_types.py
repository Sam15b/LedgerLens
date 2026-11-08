import pytest
from pydantic import ValidationError

from ledgerlens.llm.types import (
    LLMError,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMTransientError,
    LLMValidationError,
    Message,
    Usage,
)


def test_message_helpers_set_the_role() -> None:
    assert Message.system("s") == Message(role="system", content="s")
    assert Message.user("u").role == "user"
    assert Message.assistant("a").role == "assistant"


def test_unknown_role_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Message.model_validate({"role": "tool", "content": "x"})


def test_usage_total() -> None:
    assert Usage(tokens_in=3, tokens_out=4).total == 7
    with pytest.raises(ValidationError):
        Usage(tokens_in=-1)


@pytest.mark.parametrize(
    "fields",
    [
        {"messages": []},
        {"model": ""},
        {"temperature": 2.5},
        {"max_tokens": 0},
        {"timeout_s": 0},
        {"unknown": True},
    ],
)
def test_invalid_request_is_rejected(fields: dict[str, object]) -> None:
    base: dict[str, object] = {"messages": [Message.user("hi")], "model": "llama3:8b"}

    with pytest.raises(ValidationError):
        LLMRequest.model_validate(base | fields)


def test_request_and_response_are_immutable() -> None:
    req = LLMRequest(messages=[Message.user("hi")], model="m")
    resp = LLMResponse(text="hi", provider="ollama", model="m")

    with pytest.raises(ValidationError):
        req.model = "other"
    assert resp.usage == Usage() and resp.cost_usd is None


def test_error_hierarchy_carries_routing_details() -> None:
    transient = LLMTransientError("busy", provider="openai", status_code=429, retry_after_s=1.5)
    provider = LLMProviderError("bad key", provider="openai", status_code=401)
    invalid = LLMValidationError("not json", provider="ollama", raw="{oops")

    assert all(isinstance(e, LLMError) for e in (transient, provider, invalid))
    assert (transient.status_code, transient.retry_after_s) == (429, 1.5)
    assert provider.status_code == 401
    assert invalid.raw == "{oops"
