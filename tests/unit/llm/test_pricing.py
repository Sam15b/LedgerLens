import logging

import pytest

from ledgerlens.core.config import ModelPrice, ModelsConfig, load_config
from ledgerlens.llm.pricing import PriceTable
from ledgerlens.llm.types import Usage


def price(inp: float | None, out: float | None) -> ModelPrice:
    return ModelPrice.model_validate({"in": inp, "out": out})


def test_cost_is_per_million_tokens() -> None:
    table = PriceTable({"openai/gpt-4o": price(2.5, 10.0)})

    cost = table.cost_usd("openai/gpt-4o", Usage(tokens_in=1_000, tokens_out=500))

    assert cost == pytest.approx(0.0025 + 0.005)


def test_free_local_model_costs_zero_not_none() -> None:
    table = PriceTable({"ollama/llama3:8b": price(0.0, 0.0)})

    assert table.cost_usd("ollama/llama3:8b", Usage(tokens_in=9_000, tokens_out=9_000)) == 0.0


def test_cost_is_rounded_to_avoid_float_noise() -> None:
    table = PriceTable({"openai/gpt-4o-mini": price(0.15, 0.6)})

    cost = table.cost_usd("openai/gpt-4o-mini", Usage(tokens_in=3, tokens_out=7))

    assert cost == 0.00000465


@pytest.mark.parametrize("entry", [None, price(None, None), price(1.0, None), price(None, 1.0)])
def test_unpriced_model_costs_none_and_warns_once(
    entry: ModelPrice | None, caplog: pytest.LogCaptureFixture
) -> None:
    table = PriceTable({} if entry is None else {"openai/gpt-x": entry})

    with caplog.at_level(logging.WARNING, logger="ledgerlens.llm.pricing"):
        first = table.cost_usd("openai/gpt-x", Usage(tokens_in=10, tokens_out=10))
        second = table.cost_usd("openai/gpt-x", Usage(tokens_in=10, tokens_out=10))

    assert first is None and second is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "openai/gpt-x" in warnings[0].getMessage()


def test_each_unpriced_model_warns_separately(caplog: pytest.LogCaptureFixture) -> None:
    table = PriceTable({})

    with caplog.at_level(logging.WARNING, logger="ledgerlens.llm.pricing"):
        table.cost_usd("openai/a", Usage())
        table.cost_usd("openai/b", Usage())

    assert len(caplog.records) == 2


def test_from_config_reads_models_yaml_pricing() -> None:
    table = PriceTable.from_config(load_config().models)

    assert table.cost_usd("ollama/llama3:8b", Usage(tokens_in=100, tokens_out=100)) == 0.0


def test_from_config_with_explicit_section() -> None:
    config = ModelsConfig.model_validate(
        {"pricing_usd_per_1m_tokens": {"mistral/small": {"in": 1, "out": 3}}}
    )

    cost = PriceTable.from_config(config).cost_usd("mistral/small", Usage(tokens_out=1_000_000))

    assert cost == 3.0
