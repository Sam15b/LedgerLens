"""The ExtractedDocument union, the JSON-schema export helpers, and JSON round-trips."""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from ledgerlens.core.schemas import (
    EXTRACTION_SCHEMAS,
    Contract,
    DocType,
    ExtractedRecord,
    Invoice,
    Payment,
    PurchaseOrder,
    Receipt,
    extraction_json_schema,
    extraction_schema,
    parse_extracted,
)

RECORDS: list[ExtractedRecord] = [
    Invoice.model_validate(
        {
            "invoice_number": "INV-1",
            "vendor_name": "Acme",
            "invoice_date": "2026-10-02",
            "total": "12.50",
            "line_items": [
                {"description": "Wrap", "quantity": "2.5", "unit_price": "5", "amount": "12.50"}
            ],
        }
    ),
    Receipt.model_validate({"vendor_name": "Cafe", "receipt_date": "2026-03-14", "total": "9.00"}),
    PurchaseOrder.model_validate(
        {"po_number": "PO-1", "vendor_name": "Acme", "order_date": "2026-09-20", "total": "100"}
    ),
    Payment.model_validate(
        {
            "payment_reference": "RA-1",
            "vendor_name": "Acme",
            "paid_at": "2026-10-30",
            "amount": "12.50",
            "allocations": [{"invoice_number": "INV-1", "amount": "12.50"}],
        }
    ),
    Contract.model_validate(
        {
            "vendor_name": "Acme",
            "start_date": "2026-01-01",
            "payment_terms_days": 30,
            "rate_card": [{"item": "Wrap", "unit_price": "5.00"}],
        }
    ),
]


@pytest.mark.parametrize("record", RECORDS, ids=lambda r: r.doc_type)
def test_every_record_round_trips_through_json(record: ExtractedRecord) -> None:
    as_json = record.model_dump_json()

    assert type(record).model_validate_json(as_json) == record
    assert parse_extracted(as_json) == record
    assert parse_extracted(json.loads(as_json)) == record


def test_decimals_keep_their_exact_digits_in_json() -> None:
    data = json.loads(RECORDS[0].model_dump_json())

    assert data["total"] == "12.50"
    assert data["line_items"][0]["quantity"] == "2.5"


def test_parse_extracted_picks_the_record_type_from_doc_type() -> None:
    payload = {
        "doc_type": "receipt",
        "vendor_name": "Cafe",
        "receipt_date": "2026-03-14",
        "total": 9,
    }

    assert isinstance(parse_extracted(payload), Receipt)


@pytest.mark.parametrize(
    "payload",
    [
        {"vendor_name": "Cafe", "receipt_date": "2026-03-14", "total": 9},  # no doc_type
        {"doc_type": "memo", "vendor_name": "Cafe"},
        {"doc_type": "invoice", "vendor_name": "Cafe", "receipt_date": "2026-03-14", "total": 9},
    ],
)
def test_parse_extracted_rejects_missing_or_wrong_doc_type(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        parse_extracted(payload)


def test_every_doc_type_except_unknown_has_a_schema() -> None:
    assert set(EXTRACTION_SCHEMAS) == set(DocType) - {DocType.UNKNOWN}
    for doc_type, model in EXTRACTION_SCHEMAS.items():
        assert extraction_schema(doc_type) is model
        assert extraction_schema(doc_type.value) is model
        assert model.model_fields["doc_type"].default == doc_type.value
    with pytest.raises(ValueError, match="no extraction schema"):
        extraction_schema(DocType.UNKNOWN)


@pytest.mark.parametrize("doc_type", list(EXTRACTION_SCHEMAS))
def test_per_type_schema_requires_doc_type(doc_type: DocType) -> None:
    schema = extraction_json_schema(doc_type)

    assert schema == extraction_schema(doc_type).model_json_schema()
    assert schema["required"][0] == "doc_type"
    assert schema["properties"]["doc_type"]["const"] == doc_type.value
    assert schema["additionalProperties"] is False


def test_union_schema_is_a_plain_anyof_with_doc_type_in_every_branch() -> None:
    schema = extraction_json_schema()

    assert "oneOf" not in schema and "discriminator" not in schema
    branches = [ref["$ref"].removeprefix("#/$defs/") for ref in schema["anyOf"]]
    assert sorted(branches) == sorted(m.__name__ for m in EXTRACTION_SCHEMAS.values())
    for name in branches:
        assert "doc_type" in schema["$defs"][name]["required"]


def test_union_schema_is_a_fresh_copy() -> None:
    extraction_json_schema()["anyOf"].clear()

    assert extraction_json_schema()["anyOf"]
