"""Model-router types shared by the router and every provider adapter.

Callers and the rest of the code base only see :class:`Message`, :class:`LLMRequest` and
:class:`LLMResponse`; each adapter converts them to and from its vendor's wire format.

Errors are split by what the router should do next:

* :class:`LLMTransientError` -- rate limit, 5xx, timeout, connection drop: retry with
  backoff, then fall back to the next provider.
* :class:`LLMProviderError` -- this provider cannot serve the request (bad key, unknown
  model, rejected request): no retry, fall back to the next provider.
* :class:`LLMValidationError` -- the model answered but the output does not match the
  requested schema: not retried on another provider.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Message(_Frozen):
    role: Role
    content: str

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> Message:
        return cls(role="assistant", content=content)


class Usage(_Frozen):
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)

    @property
    def total(self) -> int:
        return self.tokens_in + self.tokens_out


class LLMRequest(_Frozen):
    """One call to one provider. ``model`` is the provider's own id (e.g. ``llama3:8b``)."""

    messages: list[Message] = Field(min_length=1)
    model: str = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)
    json_mode: bool = False
    """Ask for a JSON object (no particular shape)."""
    json_schema: dict[str, Any] | None = None
    """Ask for output matching this JSON schema; implies ``json_mode``."""
    schema_name: str = "response"
    """Name for ``json_schema``; some providers require one."""
    timeout_s: float = Field(default=60, gt=0)


class LLMResponse(_Frozen):
    text: str
    provider: str
    model: str
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float | None = None
    """``None`` when the model has no price in ``configs/models.yaml``."""
    latency_ms: float = Field(default=0.0, ge=0)
    finish_reason: str | None = None


class LLMProvider(Protocol):
    """A vendor adapter. ``name`` is the prefix used in ``models.yaml`` (``ollama/...``)."""

    name: str

    def complete(self, req: LLMRequest) -> LLMResponse: ...

    def stream(self, req: LLMRequest) -> Iterator[str]: ...


# --------------------------------------------------------------------------- errors


class LLMError(RuntimeError):
    """Base class for model-router errors."""

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        super().__init__(message)
        self.provider = provider


class LLMTransientError(LLMError):
    """Worth retrying on the same provider (429, 5xx, timeout, connection error)."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status_code: int | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message, provider=provider)
        self.status_code = status_code
        self.retry_after_s = retry_after_s


class LLMProviderError(LLMError):
    """Not worth retrying on this provider; the next provider may still succeed."""

    def __init__(
        self, message: str, *, provider: str | None = None, status_code: int | None = None
    ) -> None:
        super().__init__(message, provider=provider)
        self.status_code = status_code


class LLMValidationError(LLMError):
    """The model's output does not match the requested schema."""

    def __init__(self, message: str, *, provider: str | None = None, raw: str = "") -> None:
        super().__init__(message, provider=provider)
        self.raw = raw
