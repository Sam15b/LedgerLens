"""Token usage -> USD, from ``pricing_usd_per_1m_tokens`` in ``configs/models.yaml``.

Prices are keyed by the same ``provider/model`` reference the aliases use. A model with no
price (or only half of one) costs ``None`` rather than a made-up ``0.0``, and a warning is
logged once per model so a missing entry is noticed without flooding the logs.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from ledgerlens.core.config import ModelPrice, ModelsConfig, get_config
from ledgerlens.llm.types import Usage

logger = logging.getLogger(__name__)

_PER_TOKENS = 1_000_000


class PriceTable:
    def __init__(self, prices: Mapping[str, ModelPrice]) -> None:
        self._prices = dict(prices)
        self._warned: set[str] = set()

    @classmethod
    def from_config(cls, config: ModelsConfig | None = None) -> PriceTable:
        config = config or get_config().models
        return cls(config.pricing_usd_per_1m_tokens)

    def cost_usd(self, model_ref: str, usage: Usage) -> float | None:
        """Cost of one call to ``model_ref`` (``provider/model``), or ``None`` if unpriced."""
        price = self._prices.get(model_ref)
        if price is None or price.input is None or price.output is None:
            self._warn_unpriced(model_ref, price)
            return None
        cost = (usage.tokens_in * price.input + usage.tokens_out * price.output) / _PER_TOKENS
        return round(cost, 8)

    def _warn_unpriced(self, model_ref: str, price: ModelPrice | None) -> None:
        if model_ref in self._warned:
            return
        self._warned.add(model_ref)
        reason = "no entry" if price is None else "incomplete in/out entry"
        logger.warning(
            "no price for %s in pricing_usd_per_1m_tokens (%s); cost will be None",
            model_ref,
            reason,
            extra={"model_ref": model_ref},
        )
