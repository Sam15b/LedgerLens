"""Building blocks shared by the document schemas.

Extracted records are validated in two levels:

* **Hard validation** (Pydantic): every value must be well formed -- amounts are finite
  decimals, dates are real dates, currency codes look like ISO 4217, required text is not
  blank. Malformed output is an extraction error and is rejected (the model router then
  asks the LLM to repair it).
* **Consistency checks** (``consistency_issues()``): arithmetic and date order across
  fields -- line amounts vs quantity x unit price, line sum vs subtotal, subtotal + tax vs
  total, due date vs issue date. A real document can be inconsistent, and that is exactly
  what reconciliation and anomaly detection look for, so these are reported, not rejected.
"""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, StringConstraints

ROUNDING_TOLERANCE = Decimal("0.01")
"""Largest difference treated as rounding when comparing amounts (one cent per term)."""

Money = Annotated[Decimal, Field(allow_inf_nan=False)]
"""An amount in the document's currency. Negative values are allowed (credits, discounts)."""

NonNegativeMoney = Annotated[Decimal, Field(allow_inf_nan=False, ge=0)]


def _upper(value: object) -> object:
    return value.strip().upper() if isinstance(value, str) else value


def _blank_to_none(value: object) -> object:
    if isinstance(value, str):
        return value.strip() or None
    return value


CurrencyCode = Annotated[str, BeforeValidator(_upper), StringConstraints(pattern=r"^[A-Z]{3}$")]
"""ISO 4217 alphabetic code (``usd`` and `` GBP`` are normalised to ``USD`` / ``GBP``)."""

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
"""Required free text: surrounding whitespace removed, never empty."""

OptionalText = Annotated[str | None, BeforeValidator(_blank_to_none)]
"""Optional free text: blank strings become ``None``."""


class SchemaModel(BaseModel):
    """Base for every LedgerLens record: unknown fields are rejected, values are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _require_doc_type(schema: dict[str, Any]) -> None:
    required = schema.setdefault("required", [])
    if "doc_type" in schema.get("properties", {}) and "doc_type" not in required:
        required.insert(0, "doc_type")


class ExtractedModel(SchemaModel):
    """Base for records extracted from a document (invoice, PO, payment, contract, ...).

    Each subclass fixes ``doc_type`` to a literal. It has a default for Python callers, but
    the JSON schema lists it as required, so LLM output always carries the discriminator
    that :data:`~ledgerlens.core.schemas.ExtractedDocument` needs.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, json_schema_extra=_require_doc_type)


def amounts_differ(a: Decimal, b: Decimal, *, tolerance: Decimal = ROUNDING_TOLERANCE) -> bool:
    return abs(a - b) > tolerance


def sum_amounts(amounts: Iterable[Decimal]) -> Decimal:
    return sum(amounts, Decimal(0))
