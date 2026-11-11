"""``LLMClient``: one interface for every model, addressed by alias.

Callers ask for an alias from ``configs/models.yaml`` (``smart``, ``fast``, ``local``, ...)
or an explicit ``provider/model``. The alias resolves to an ordered list of models; each is
tried in turn until one answers:

* :class:`LLMTransientError` (429, 5xx, timeout) -> retried on the same model with the
  ``retry`` backoff from ``models.yaml``, then the next model.
* :class:`LLMProviderError` (bad key, unknown model, provider not built yet) -> next model.
* :class:`LLMValidationError` -> raised to the caller; another provider is not tried.

:meth:`LLMClient.structured` sends a Pydantic model's JSON schema, validates the reply into
an instance and, if it does not validate, asks the *same* model once to repair it.

Every successful call is priced from ``pricing_usd_per_1m_tokens`` and logged with
provider, model, tokens, cost and latency. See docs/layers/05-model-router.md.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ledgerlens.core.config import ModelsConfig, get_config, split_model_ref
from ledgerlens.llm.pricing import PriceTable
from ledgerlens.llm.providers.gemini_provider import GeminiProvider
from ledgerlens.llm.providers.mistral_provider import MistralProvider
from ledgerlens.llm.providers.ollama_provider import OllamaProvider
from ledgerlens.llm.providers.openai_provider import OpenAIProvider
from ledgerlens.llm.types import (
    LLMError,
    LLMProvider,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMTransientError,
    LLMValidationError,
    Message,
)

logger = logging.getLogger(__name__)

ProviderFactory = Callable[[], LLMProvider]
T = TypeVar("T", bound=BaseModel)

# Honour a server's Retry-After, but never park a caller for longer than this.
MAX_RETRY_AFTER_S = 30.0

# Local models often wrap JSON in a Markdown fence even in JSON mode.
_CODE_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)
_MAX_REPORTED_ERRORS = 10

# Providers built so far; the rest of models.yaml's providers arrive with their adapters.
DEFAULT_PROVIDER_FACTORIES: Mapping[str, ProviderFactory] = {
    "gemini": GeminiProvider,
    "mistral": MistralProvider,
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
}


class AllProvidersFailedError(LLMError):
    """Every model behind an alias failed; ``attempts`` holds ``(model_ref, error)`` pairs."""

    def __init__(self, alias: str, attempts: list[tuple[str, LLMError]]) -> None:
        tried = "; ".join(f"{ref}: {exc}" for ref, exc in attempts) or "nothing to try"
        super().__init__(f"all models for {alias!r} failed ({tried})")
        self.alias = alias
        self.attempts = attempts


class LLMClient:
    def __init__(
        self,
        config: ModelsConfig | None = None,
        *,
        providers: Mapping[str, LLMProvider] | None = None,
        factories: Mapping[str, ProviderFactory] | None = None,
        prices: PriceTable | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """``providers`` pins ready-made adapters (tests, custom clients); any other provider
        is built on first use from ``factories``."""
        self.config = config or get_config().models
        self._providers: dict[str, LLMProvider] = dict(providers or {})
        self._factories = dict(DEFAULT_PROVIDER_FACTORIES if factories is None else factories)
        self._prices = prices or PriceTable.from_config(self.config)
        self._sleep = sleep

    def complete(
        self,
        messages: list[Message],
        *,
        model: str = "smart",
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "response",
    ) -> LLMResponse:
        """Answer with the first model behind ``model`` (an alias or ``provider/model``)."""
        refs = self.config.resolve(model)
        attempts: list[tuple[str, LLMError]] = []
        for index, ref in enumerate(refs):
            provider_name, model_id = split_model_ref(ref)
            request = LLMRequest(
                messages=messages,
                model=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode or json_schema is not None,
                json_schema=json_schema,
                schema_name=schema_name,
                timeout_s=self.config.timeout_s,
            )
            try:
                response = self._call(provider_name, request)
            except (LLMTransientError, LLMProviderError) as exc:
                attempts.append((ref, exc))
                self._log_failure(model, ref, exc, has_next=index + 1 < len(refs))
                continue
            return self._finish(model, ref, response)
        raise AllProvidersFailedError(model, attempts)

    def structured(
        self,
        messages: list[Message],
        schema: type[T],
        *,
        model: str = "smart",
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> T:
        """Return the reply parsed into ``schema``, with one repair attempt.

        The schema goes to the provider as native structured output. If the reply still does
        not validate, the validation errors are shown to the model that produced it (not the
        next one in the alias) and it gets one chance to fix them; after that
        :class:`LLMValidationError` is raised.
        """
        json_schema = schema.model_json_schema()
        call: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "json_schema": json_schema,
            "schema_name": schema.__name__,
        }
        response = self.complete(messages, model=model, **call)
        try:
            return parse_structured(schema, response.text)
        except ValidationError as first_error:
            ref = f"{response.provider}/{response.model}"
            logger.warning(
                "structured output from %s does not match %s; asking it to repair: %s",
                ref,
                schema.__name__,
                _describe(first_error),
                extra={"llm_model_ref": ref, "schema": schema.__name__},
            )
            repair = [
                *messages,
                Message.assistant(response.text),
                Message.user(_repair_prompt(schema, first_error)),
            ]
            repaired = self.complete(repair, model=ref, **call)
            try:
                return parse_structured(schema, repaired.text)
            except ValidationError as second_error:
                raise LLMValidationError(
                    f"{ref} returned output that does not match {schema.__name__} after one "
                    f"repair attempt: {_describe(second_error)}",
                    provider=response.provider,
                    raw=repaired.text,
                ) from second_error

    # ------------------------------------------------------------------ internals

    def _provider(self, name: str) -> LLMProvider:
        provider = self._providers.get(name)
        if provider is None:
            factory = self._factories.get(name)
            if factory is None:
                raise LLMProviderError(f"provider {name!r} is not available yet", provider=name)
            provider = self._providers[name] = factory()
        return provider

    def _call(self, provider_name: str, request: LLMRequest) -> LLMResponse:
        """Call one model, retrying transient errors with the configured backoff."""
        provider = self._provider(provider_name)
        retry = self.config.retry
        for attempt in range(1, retry.max_attempts + 1):
            try:
                return provider.complete(request)
            except LLMTransientError as exc:
                if attempt == retry.max_attempts:
                    raise
                delay = retry.delay_for(attempt)
                if exc.retry_after_s is not None:
                    delay = max(delay, min(exc.retry_after_s, MAX_RETRY_AFTER_S))
                logger.warning(
                    "llm call transient error provider=%s model=%s attempt=%d/%d: %s; "
                    "retrying in %.1fs",
                    provider_name,
                    request.model,
                    attempt,
                    retry.max_attempts,
                    exc,
                    delay,
                    extra={"llm_provider": provider_name, "llm_model": request.model},
                )
                self._sleep(delay)
        raise AssertionError("unreachable: max_attempts >= 1")  # pragma: no cover

    def _finish(self, alias: str, ref: str, response: LLMResponse) -> LLMResponse:
        response = response.model_copy(
            update={"cost_usd": self._prices.cost_usd(ref, response.usage)}
        )
        cost = "unknown" if response.cost_usd is None else f"{response.cost_usd:.6f}"
        logger.info(
            "llm call alias=%s provider=%s model=%s tokens_in=%d tokens_out=%d "
            "cost_usd=%s latency_ms=%.0f",
            alias,
            response.provider,
            response.model,
            response.usage.tokens_in,
            response.usage.tokens_out,
            cost,
            response.latency_ms,
            extra={
                "llm_alias": alias,
                "llm_provider": response.provider,
                "llm_model": response.model,
                "tokens_in": response.usage.tokens_in,
                "tokens_out": response.usage.tokens_out,
                "cost_usd": response.cost_usd,
                "latency_ms": response.latency_ms,
            },
        )
        return response

    @staticmethod
    def _log_failure(alias: str, ref: str, exc: LLMError, *, has_next: bool) -> None:
        action = "falling back to next model" if has_next else "no models left"
        logger.warning(
            "llm call failed alias=%s model=%s error=%s: %s; %s",
            alias,
            ref,
            type(exc).__name__,
            exc,
            action,
            extra={"llm_alias": alias, "llm_model_ref": ref, "error_type": type(exc).__name__},
        )


# --------------------------------------------------------------------------- structured


def parse_structured(schema: type[T], text: str) -> T:
    """Validate ``text`` as JSON for ``schema``, tolerating a surrounding code fence."""
    fenced = _CODE_FENCE.match(text)
    return schema.model_validate_json(fenced.group(1) if fenced else text)


def _describe(error: ValidationError) -> str:
    problems = [
        f"{'.'.join(str(part) for part in e['loc']) or '<root>'}: {e['msg']}"
        for e in error.errors()[:_MAX_REPORTED_ERRORS]
    ]
    more = error.error_count() - len(problems)
    return "; ".join(problems) + (f"; and {more} more" if more > 0 else "")


def _repair_prompt(schema: type[BaseModel], error: ValidationError) -> str:
    return (
        f"Your previous reply is not a valid {schema.__name__} object. Problems: "
        f"{_describe(error)}. Reply again with only the corrected JSON object - no prose, "
        "no code fences."
    )
