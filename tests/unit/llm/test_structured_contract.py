"""The same Pydantic schema parses from every provider's structured output.

Each provider's recorded structured reply is played back through its real adapter and
``LLMClient.structured``; all five must produce the same model instance. Also checks that
``configs/models.yaml`` only names providers the router can build and prices every model.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from ledgerlens.core.config import ModelsConfig, load_config, split_model_ref
from ledgerlens.llm.providers.bedrock_provider import BedrockProvider
from ledgerlens.llm.providers.gemini_provider import GeminiProvider
from ledgerlens.llm.providers.mistral_provider import MistralProvider
from ledgerlens.llm.providers.ollama_provider import OllamaProvider
from ledgerlens.llm.providers.openai_provider import OpenAIProvider
from ledgerlens.llm.router import DEFAULT_PROVIDER_FACTORIES, LLMClient
from ledgerlens.llm.types import LLMProvider, Message

FIXTURES = Path(__file__).parents[2] / "fixtures" / "llm"
REPO_CONFIGS = Path(__file__).parents[3] / "configs"


class Invoice(BaseModel):
    vendor: str
    total: float
    currency: str = "USD"
    po_number: str | None = None


EXPECTED = Invoice(vendor="Acme Supplies", total=1250.0)


def _http(fixture: str) -> httpx.Client:
    body = (FIXTURES / fixture).read_text(encoding="utf-8")
    return httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, text=body)))


class _Converse:
    def converse(self, **kwargs: Any) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(
            (FIXTURES / "bedrock" / "converse_structured.json").read_text(encoding="utf-8")
        )
        return data

    def converse_stream(self, **kwargs: Any) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError


ADAPTERS: dict[str, Callable[[], LLMProvider]] = {
    "openai": lambda: OpenAIProvider("k", client=_http("openai/chat_structured.json")),
    "ollama": lambda: OllamaProvider("http://ollama.test", client=_http("ollama/chat_json.json")),
    "gemini": lambda: GeminiProvider("k", client=_http("gemini/generate_structured.json")),
    "mistral": lambda: MistralProvider("k", client=_http("mistral/chat_structured.json")),
    "bedrock": lambda: BedrockProvider(client=_Converse()),
}


@pytest.mark.parametrize("provider", sorted(ADAPTERS))
def test_same_schema_parses_from_every_provider(provider: str) -> None:
    client = LLMClient(ModelsConfig(), providers={provider: ADAPTERS[provider]()}, factories={})

    invoice = client.structured(
        [Message.user("Extract the invoice.")], Invoice, model=f"{provider}/any-model"
    )

    assert invoice == EXPECTED


def test_every_provider_has_an_adapter() -> None:
    assert set(DEFAULT_PROVIDER_FACTORIES) == set(ADAPTERS)


def test_models_yaml_uses_known_providers_and_prices_every_model() -> None:
    models = load_config(REPO_CONFIGS).models
    refs = {ref for alias in models.aliases for ref in models.resolve(alias)}

    assert {split_model_ref(ref)[0] for ref in refs} <= set(DEFAULT_PROVIDER_FACTORIES)
    unpriced = sorted(
        ref
        for ref in refs
        if (price := models.pricing_usd_per_1m_tokens.get(ref)) is None
        or price.input is None
        or price.output is None
    )
    assert unpriced == []
