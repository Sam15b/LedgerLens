import logging
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel

from ledgerlens.core.config import ModelsConfig
from ledgerlens.llm.providers.openai_provider import OpenAIProvider
from ledgerlens.llm.router import MAX_RETRY_AFTER_S, AllProvidersFailedError, LLMClient
from ledgerlens.llm.types import (
    LLMError,
    LLMProvider,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMTransientError,
    LLMValidationError,
    Message,
    Usage,
)

FIXTURES = Path(__file__).parents[2] / "fixtures" / "llm"
MESSAGES = [Message.user("Extract the invoice.")]

CONFIG = ModelsConfig.model_validate(
    {
        "aliases": {
            "smart": ["openai/big", "ollama/small"],
            "local": ["ollama/small"],
            "future": ["gemini/flash", "ollama/small"],
            "empty": [],
        },
        "pricing_usd_per_1m_tokens": {
            "openai/big": {"in": 2.0, "out": 10.0},
            "ollama/small": {"in": 0.0, "out": 0.0},
        },
        "retry": {"max_attempts": 3, "backoff_s": [1, 2]},
        "timeout_s": 12,
    }
)

Outcome = str | LLMError


class FakeProvider:
    """Plays back scripted outcomes: a string is a reply, an exception is raised."""

    def __init__(self, name: str, *outcomes: Outcome) -> None:
        self.name = name
        self.outcomes = list(outcomes)
        self.requests: list[LLMRequest] = []

    def complete(self, req: LLMRequest) -> LLMResponse:
        self.requests.append(req)
        outcome = self.outcomes.pop(0) if self.outcomes else "ok"
        if isinstance(outcome, LLMError):
            raise outcome
        return LLMResponse(
            text=outcome,
            provider=self.name,
            model=req.model,
            usage=Usage(tokens_in=1_000, tokens_out=100),
            latency_ms=5.0,
        )

    def stream(self, req: LLMRequest) -> Iterator[str]:
        yield from ()


def make_client(*providers: FakeProvider, sleeps: list[float] | None = None) -> LLMClient:
    pinned: dict[str, LLMProvider] = {p.name: p for p in providers}
    sink = sleeps if sleeps is not None else []
    return LLMClient(CONFIG, providers=pinned, factories={}, sleep=sink.append)


def transient(status: int = 503, retry_after_s: float | None = None) -> LLMTransientError:
    return LLMTransientError(
        f"HTTP {status}", provider="x", status_code=status, retry_after_s=retry_after_s
    )


class Invoice(BaseModel):
    vendor: str
    total: float
    currency: str = "USD"


# --------------------------------------------------------------------------- complete


def test_first_model_answers_and_is_priced() -> None:
    openai, ollama = FakeProvider("openai", "hello"), FakeProvider("ollama")

    response = make_client(openai, ollama).complete(MESSAGES, model="smart", max_tokens=50)

    assert (response.text, response.provider, response.model) == ("hello", "openai", "big")
    assert response.cost_usd == pytest.approx((1_000 * 2.0 + 100 * 10.0) / 1_000_000)
    assert ollama.requests == []
    sent = openai.requests[0]
    assert (sent.model, sent.max_tokens, sent.timeout_s) == ("big", 50, 12)
    assert sent.messages == MESSAGES


def test_explicit_model_ref_skips_alias_lookup() -> None:
    ollama = FakeProvider("ollama")

    response = make_client(ollama).complete(MESSAGES, model="ollama/other")

    assert response.model == "other"
    assert response.cost_usd is None  # not in the price table


def test_every_call_is_logged_with_usage_cost_and_latency(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="ledgerlens.llm.router"):
        make_client(FakeProvider("ollama")).complete(MESSAGES, model="local")

    (record,) = [r for r in caplog.records if r.getMessage().startswith("llm call alias")]
    assert "provider=ollama model=small tokens_in=1000 tokens_out=100" in record.getMessage()
    assert record.__dict__["cost_usd"] == 0.0
    assert record.__dict__["latency_ms"] == 5.0


def test_json_schema_implies_json_mode() -> None:
    ollama = FakeProvider("ollama")

    make_client(ollama).complete(MESSAGES, model="local", json_schema={"type": "object"})

    assert ollama.requests[0].json_mode is True


def test_unknown_alias_and_empty_alias_fail_fast() -> None:
    client = make_client(FakeProvider("ollama"))

    with pytest.raises(KeyError, match="unknown model alias"):
        client.complete(MESSAGES, model="nope")
    with pytest.raises(ValueError, match="no models"):
        client.complete(MESSAGES, model="empty")


# --------------------------------------------------------------------------- fallback


def test_provider_error_falls_back_without_retry(caplog: pytest.LogCaptureFixture) -> None:
    openai = FakeProvider("openai", LLMProviderError("bad key", provider="openai"))
    ollama = FakeProvider("ollama", "from llama")
    sleeps: list[float] = []

    with caplog.at_level(logging.WARNING, logger="ledgerlens.llm.router"):
        response = make_client(openai, ollama, sleeps=sleeps).complete(MESSAGES, model="smart")

    assert response.text == "from llama"
    assert len(openai.requests) == 1
    assert sleeps == []
    assert "falling back to next model" in caplog.text


def test_provider_without_an_adapter_falls_through() -> None:
    ollama = FakeProvider("ollama", "fallback")

    response = make_client(ollama).complete(MESSAGES, model="future")

    assert response.provider == "ollama"


def test_all_models_failing_lists_every_attempt() -> None:
    openai = FakeProvider("openai", LLMProviderError("bad key"))
    ollama = FakeProvider("ollama", *[transient()] * 3)

    with pytest.raises(AllProvidersFailedError) as excinfo:
        make_client(openai, ollama).complete(MESSAGES, model="smart")

    assert excinfo.value.alias == "smart"
    assert [ref for ref, _ in excinfo.value.attempts] == ["openai/big", "ollama/small"]
    assert "bad key" in str(excinfo.value)


def test_validation_error_is_not_tried_on_another_provider() -> None:
    openai = FakeProvider("openai", LLMValidationError("schema rejected", provider="openai"))
    ollama = FakeProvider("ollama")

    with pytest.raises(LLMValidationError):
        make_client(openai, ollama).complete(MESSAGES, model="smart")

    assert ollama.requests == []


def test_providers_are_built_lazily_and_reused() -> None:
    built: list[str] = []

    def factory() -> LLMProvider:
        built.append("ollama")
        return FakeProvider("ollama")

    client = LLMClient(CONFIG, factories={"ollama": factory})

    assert built == []
    client.complete(MESSAGES, model="local")
    client.complete(MESSAGES, model="local")
    assert built == ["ollama"]


# --------------------------------------------------------------------------- retry


def test_transient_error_is_retried_with_backoff() -> None:
    openai = FakeProvider("openai", transient(429), transient(503), "third time lucky")
    ollama = FakeProvider("ollama")
    sleeps: list[float] = []

    response = make_client(openai, ollama, sleeps=sleeps).complete(MESSAGES, model="smart")

    assert response.text == "third time lucky"
    assert len(openai.requests) == 3
    assert sleeps == [1, 2]
    assert ollama.requests == []


def test_retries_are_capped_then_next_model_is_tried() -> None:
    openai = FakeProvider("openai", *[transient()] * 5)
    ollama = FakeProvider("ollama", "rescued")
    sleeps: list[float] = []

    response = make_client(openai, ollama, sleeps=sleeps).complete(MESSAGES, model="smart")

    assert response.provider == "ollama"
    assert len(openai.requests) == CONFIG.retry.max_attempts
    assert sleeps == [1, 2]  # no sleep after the last attempt


def test_retry_after_is_honoured_and_capped() -> None:
    openai = FakeProvider("openai", transient(429, retry_after_s=7), transient(429, 3600), "ok")
    sleeps: list[float] = []

    make_client(openai, sleeps=sleeps).complete(MESSAGES, model="openai/big")

    assert sleeps == [7, MAX_RETRY_AFTER_S]


def test_backoff_wins_over_a_shorter_retry_after() -> None:
    openai = FakeProvider("openai", transient(429, retry_after_s=0.1), "ok")
    sleeps: list[float] = []

    make_client(openai, sleeps=sleeps).complete(MESSAGES, model="openai/big")

    assert sleeps == [1]


# --------------------------------------------------------------------------- structured


def test_structured_returns_a_validated_instance() -> None:
    openai = FakeProvider("openai", '{"vendor": "Acme", "total": 99.5}')

    invoice = make_client(openai).structured(MESSAGES, Invoice, model="smart")

    assert invoice == Invoice(vendor="Acme", total=99.5)
    sent = openai.requests[0]
    assert sent.json_schema == Invoice.model_json_schema()
    assert sent.schema_name == "Invoice"


def test_structured_tolerates_a_code_fence() -> None:
    ollama = FakeProvider("ollama", '```json\n{"vendor": "Acme", "total": 1}\n```')

    invoice = make_client(ollama).structured(MESSAGES, Invoice, model="local")

    assert invoice.total == 1


def test_structured_repairs_once_on_the_model_that_answered() -> None:
    openai = FakeProvider("openai", LLMProviderError("down"))
    ollama = FakeProvider("ollama", '{"vendor": "Acme"}', '{"vendor": "Acme", "total": 5}')

    invoice = make_client(openai, ollama).structured(MESSAGES, Invoice, model="smart")

    assert invoice.total == 5
    assert len(openai.requests) == 1  # the repair is not sent back through the alias
    repair = ollama.requests[1].messages
    assert repair[: len(MESSAGES)] == MESSAGES
    assert repair[-2] == Message.assistant('{"vendor": "Acme"}')
    assert repair[-1].role == "user"
    assert "total: Field required" in repair[-1].content


def test_structured_gives_up_after_one_repair() -> None:
    ollama = FakeProvider("ollama", "not json", '{"total": "lots"}')

    with pytest.raises(LLMValidationError, match="after one repair attempt") as excinfo:
        make_client(ollama).structured(MESSAGES, Invoice, model="local")

    assert excinfo.value.raw == '{"total": "lots"}'
    assert excinfo.value.provider == "ollama"
    assert len(ollama.requests) == 2


# --------------------------------------------------------------------------- end to end


def test_structured_through_the_openai_adapter() -> None:
    body = (FIXTURES / "openai" / "chat_structured.json").read_text(encoding="utf-8")
    http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=body)))
    openai = OpenAIProvider("sk-test", base_url="https://openai.test/v1", client=http)
    config = ModelsConfig.model_validate(
        {
            "aliases": {"smart": ["openai/gpt-4o"]},
            "pricing_usd_per_1m_tokens": {"openai/gpt-4o": {"in": 2.5, "out": 10.0}},
        }
    )
    client = LLMClient(config, providers={"openai": openai})

    invoice = client.structured(MESSAGES, Invoice, model="smart")
    response = client.complete(MESSAGES, model="smart")

    assert invoice == Invoice(vendor="Acme Supplies", total=1250.0, currency="USD")
    assert response.cost_usd == pytest.approx((112 * 2.5 + 18 * 10.0) / 1_000_000)
