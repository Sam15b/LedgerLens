"""Extraction with the fine-tuned Llama 3 (``ledgerlens-llama3-lora`` on Ollama).

The model was trained (``ml/finetune``) on one prompt format, and it is only reliable with
that format, so this module owns it and ``ml/finetune/prepare_dataset.py`` imports it from
here: the system rules (:data:`SYSTEM_PROMPT`), a task naming the keys to fill
(:func:`instruction`), then the document text. The reply is one JSON object with exactly
those keys. Training covered receipts and invoices only (:data:`KEYS_BY_DOC_TYPE`);
purchase orders, payments and contracts go to ``fallback`` (the generic
:class:`~ledgerlens.extraction.llm_extractor.LLMExtractor`).

The call goes through the router under the ``extraction.model_alias`` alias
(``extractor`` in ``models.yaml``): the fine-tuned model first, then API models, which
follow the same self-describing prompt. The reply's keys are those of the training
targets, so it is mapped onto the record schema (:func:`record_fields`): keys the record
does not have are dropped (a receipt's ``subtotal``), and so are line items without a
description or amount. Values that are not in the trained format (``1,250.00``,
``14/03/2026``) are normalised the same way the rules extractor normalises text.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from ledgerlens.core.config import ExtractionConfig, get_config
from ledgerlens.core.schemas import DocType, extraction_schema
from ledgerlens.extraction.base import ExtractionError, ExtractionResult, Extractor
from ledgerlens.extraction.llm_extractor import MAX_DOCUMENT_CHARS, LLMExtractor, tidy_record
from ledgerlens.extraction.normalize import parse_amount, parse_date
from ledgerlens.llm.router import LLMClient
from ledgerlens.llm.types import LLMError, Message

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- prompt contract
# Changing any of this changes what the model was trained on: retrain, or keep it as is.

SYSTEM_PROMPT = (
    "You extract structured fields from business documents. Reply with one JSON object and "
    "nothing else."
)
RECEIPT_KEYS = ("vendor_name", "vendor_address", "receipt_date", "total")
"""SROIE's labels."""
CORD_KEYS = ("line_items", "subtotal", "tax", "total")
"""CORD's labels."""
INVOICE_KEYS = (
    "invoice_number",
    "vendor_name",
    "vendor_tax_id",
    "customer_name",
    "invoice_date",
    "due_date",
    "po_number",
    "currency",
    "subtotal",
    "tax",
    "total",
    "line_items",
)
"""The synthetic invoices' labels."""
KEYS_BY_DOC_TYPE: dict[str, tuple[str, ...]] = {
    "receipt": (*RECEIPT_KEYS[:-1], *CORD_KEYS),  # vendor, address, date, lines, totals
    "invoice": INVOICE_KEYS,
}
"""What to ask for at inference: every key the training data has for the document type."""
MAX_NEW_TOKENS = 768


def instruction(doc_type: str, keys: Sequence[str]) -> str:
    """The task for one document: which document, which keys, and the value formats."""
    return (
        f"Extract the {doc_type.replace('_', ' ')} fields from the document text as one JSON "
        f"object with exactly these keys: {', '.join(keys)}. Use null for a field the text "
        "does not show. Write dates as YYYY-MM-DD and amounts as plain numbers with two "
        "decimals and no thousands separators. Line items have description, quantity, "
        "unit_price and amount."
    )


def user_message(text: str, doc_type: str, keys: Sequence[str]) -> str:
    """The user turn: the task, then the document text."""
    return f"{instruction(doc_type, keys)}\n\nDocument text:\n{text}"


def parse_json_object(text: str) -> dict[str, Any] | None:
    """The first JSON object in a model's reply (code fences and chatter around it are
    ignored), or ``None`` if there is none."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        return value if isinstance(value, dict) else None
    return None


# --------------------------------------------------------------------------- reply -> record

_AMOUNT_KEYS = frozenset({"subtotal", "tax", "total", "unit_price", "amount", "quantity"})
_LINE_KEYS = ("description", "quantity", "unit_price", "amount")


def _amount(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return Decimal(value.strip())
    except InvalidOperation:
        parsed = parse_amount(value)
        return value if parsed is None else parsed


def _date(value: Any, *, day_first: bool) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        parsed = parse_date(value, day_first=day_first)
        return value if parsed is None else parsed


def _line_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    items = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        line = {
            k: _amount(item[k]) if k != "description" else item[k] for k in _LINE_KEYS if k in item
        }
        if line.get("description") in (None, "") or line.get("amount") is None:
            continue  # a line without a name or a total cannot be booked
        items.append(line)
    return items


def record_fields(
    doc_type: DocType, reply: Mapping[str, Any], *, day_first: bool
) -> dict[str, Any]:
    """The reply's values under the record schema's field names, normalised."""
    allowed = set(extraction_schema(doc_type).model_fields) - {"doc_type"}
    fields: dict[str, Any] = {}
    for key, value in reply.items():
        if key not in allowed:
            continue
        if key == "line_items":
            fields[key] = _line_items(value)
        elif key in _AMOUNT_KEYS:
            fields[key] = _amount(value)
        elif key.endswith("_date"):
            fields[key] = _date(value, day_first=day_first)
        else:
            fields[key] = value
    return fields


# --------------------------------------------------------------------------- extractor


class FineTunedExtractor:
    """:class:`~ledgerlens.extraction.base.Extractor` backed by the fine-tuned model."""

    name = "finetuned"

    def __init__(
        self,
        client: LLMClient | None = None,
        *,
        config: ExtractionConfig | None = None,
        fallback: Extractor | None = None,
    ) -> None:
        self.config = config or get_config().ingestion.extraction
        self._client = client
        self._fallback = fallback

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            self._client = LLMClient()
        return self._client

    @property
    def fallback(self) -> Extractor:
        """The extractor for document types the model was not trained on."""
        if self._fallback is None:
            self._fallback = LLMExtractor(self.client, config=self.config)
        return self._fallback

    @property
    def model(self) -> str:
        return self.config.model_alias

    def messages(self, text: str, doc_type: DocType) -> list[Message]:
        if len(text) > MAX_DOCUMENT_CHARS:
            logger.warning("document text truncated from %d characters for extraction", len(text))
            text = text[:MAX_DOCUMENT_CHARS]
        keys = KEYS_BY_DOC_TYPE[doc_type.value]
        return [Message.system(SYSTEM_PROMPT), Message.user(user_message(text, doc_type, keys))]

    def extract(self, text: str, doc_type: DocType | str) -> ExtractionResult:
        """Extract ``doc_type`` from ``text``.

        A reply that is not a JSON object, or whose values do not validate, comes back as a
        result with ``errors``; failing to reach any model raises :class:`ExtractionError`.
        """
        kind = DocType(doc_type)
        if kind.value not in KEYS_BY_DOC_TYPE:
            return self.fallback.extract(text, kind)
        try:
            response = self.client.complete(
                self.messages(text, kind),
                model=self.model,
                temperature=0.0,
                max_tokens=MAX_NEW_TOKENS,
                json_mode=True,
            )
        except LLMError as exc:
            raise ExtractionError(f"fine-tuned extraction failed: {exc}") from exc
        reply = parse_json_object(response.text)
        if reply is None:
            ref = f"{response.provider}/{response.model}"
            logger.warning("%s replied without a JSON object for %s", ref, kind)
            return ExtractionResult(
                extractor=self.name, doc_type=kind, errors=[f"{ref}: reply is not a JSON object"]
            )
        fields = record_fields(kind, reply, day_first=self.config.day_first)
        result = ExtractionResult.from_fields(self.name, kind, fields)
        if result.record is None:
            return result
        tidy = tidy_record(result.record)
        return ExtractionResult(
            extractor=self.name,
            doc_type=kind,
            record=tidy,
            fields=tidy.model_dump(mode="json", exclude_none=True),
        )
