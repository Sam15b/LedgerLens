"""Typed records passed between every layer: documents and what is extracted from them.

* :class:`Document` -- an uploaded file and its pipeline status.
* Extracted records, one per document type: :class:`Invoice` (with :class:`LineItem`),
  :class:`Receipt`, :class:`PurchaseOrder`, :class:`Payment` and :class:`Contract`.
* :data:`ExtractedDocument` -- any of those, told apart by ``doc_type``.

For structured LLM output, hand the record class to ``LLMClient.structured()``, or get the
JSON schema with :func:`extraction_json_schema`.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import Field, TypeAdapter

from ledgerlens.core.schemas.common import ROUNDING_TOLERANCE, ExtractedModel
from ledgerlens.core.schemas.contract import Contract, RateCardEntry
from ledgerlens.core.schemas.document import DocType, Document, DocumentStatus
from ledgerlens.core.schemas.invoice import Invoice, LineItem, Receipt
from ledgerlens.core.schemas.payment import Payment, PaymentAllocation
from ledgerlens.core.schemas.purchase_order import PurchaseOrder

ExtractedRecord = Invoice | Receipt | PurchaseOrder | Payment | Contract

ExtractedDocument = Annotated[ExtractedRecord, Field(discriminator="doc_type")]
"""Any extracted record; ``doc_type`` says which (validate with :func:`parse_extracted`)."""

EXTRACTION_SCHEMAS: Mapping[DocType, type[ExtractedModel]] = {
    DocType.INVOICE: Invoice,
    DocType.RECEIPT: Receipt,
    DocType.PURCHASE_ORDER: PurchaseOrder,
    DocType.PAYMENT: Payment,
    DocType.CONTRACT: Contract,
}

_EXTRACTED: TypeAdapter[ExtractedRecord] = TypeAdapter(ExtractedDocument)


def extraction_schema(doc_type: DocType | str) -> type[ExtractedModel]:
    """The record class to extract for ``doc_type``."""
    kind = DocType(doc_type)
    try:
        return EXTRACTION_SCHEMAS[kind]
    except KeyError:
        raise ValueError(f"no extraction schema for doc_type {kind.value!r}") from None


def parse_extracted(data: str | bytes | Mapping[str, Any]) -> ExtractedRecord:
    """Validate a JSON string/bytes or a mapping into the record its ``doc_type`` names."""
    if isinstance(data, str | bytes):
        return _EXTRACTED.validate_json(data)
    return _EXTRACTED.validate_python(data)


def extraction_json_schema(doc_type: DocType | str | None = None) -> dict[str, Any]:
    """JSON schema for structured LLM output.

    With ``doc_type``: that record's schema, exactly what ``LLMClient.structured()`` sends.
    Without: every record type as one schema. Pydantic writes that union as ``oneOf`` plus
    a ``discriminator`` mapping, which several providers' structured-output modes reject, so
    it is rewritten to a plain ``anyOf``; ``doc_type`` is required in each branch, so the
    reply still says which record it is.
    """
    if doc_type is not None:
        return extraction_schema(doc_type).model_json_schema()
    schema = copy.deepcopy(_EXTRACTED.json_schema())
    schema["anyOf"] = schema.pop("oneOf")
    schema.pop("discriminator", None)
    return schema


__all__ = [
    "EXTRACTION_SCHEMAS",
    "ROUNDING_TOLERANCE",
    "Contract",
    "DocType",
    "Document",
    "DocumentStatus",
    "ExtractedDocument",
    "ExtractedModel",
    "ExtractedRecord",
    "Invoice",
    "LineItem",
    "Payment",
    "PaymentAllocation",
    "PurchaseOrder",
    "RateCardEntry",
    "Receipt",
    "extraction_json_schema",
    "extraction_schema",
    "parse_extracted",
]
