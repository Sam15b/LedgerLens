import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from ledgerlens.core.schemas import DocType, Invoice, Payment, Receipt
from ledgerlens.extraction.base import ExtractionResult, Extractor
from ledgerlens.extraction.rules import RulesExtractor, first_line_vendor, receipt_address
from ledgerlens.ingestion.layout import PageContent, layout_pages

FIXTURES = Path(__file__).parents[2] / "fixtures" / "documents"
# What Tesseract reads off receipt_scanned.png ("KOPI" is lost to speckle).
RECEIPT_OCR = """\
KEDAI  SENTOSA
12 Jalan Bunga, 50450 Kuala Lumpur
Receipt: R-000732  Date: 14/03/2026
2x Kom O  440
1 x Roti bakar  5.10
TOTAL MYR  9.50
incl. SST 6%  0.54
Terima kasih!
"""
# SROIE-style receipts (layout and wording of the ICDAR 2019 training set).
SROIE_BOOKSHOP = """\
TAN WOON YANN
BOOK TA .K(TAMAN DAYA) SDN BHD
789417-W
NO.53 55,57 & 59, JALAN SAGU 18,
TAMAN DAYA,
81100 JOHOR BAHRU,
JOHOR.
DOCUMENT NO : TD01167104
DATE: 25/12/2018 8:13:39 PM
CASHIER: MANIS
MEMBER:
CASH BILL
CODE/DESC PRICE DISC AMOUNT
QTY RM RM
9556939040116 KF MODELLING CLAY KIDDY FISH
1 PC * 9.000 0.00 9.00
TOTAL: 9.00
ROUND D TOTAL (RM): 9.00
CASH 10.00
CHANGE 1.00
"""
SROIE_HARDWARE = """\
SYARIKAT PERNIAGAAN GIN KEE
(81109-A)
NO 290, JALAN AIR PANAS,
SETAPAK,
53200, KUALA LUMPUR.
TEL : 03-40210276
GST ID : 000750673920
SIMPLIFIED TAX INVOICE
CASH
DOC NO. : CS00012993
DATE : 04/01/2018
SALES PERSON : FATIN
Item Qty Price Amount
1072 1 7.95 7.95 SR
Total Sales (Excluding GST) : 7.50
Total GST : 0.45
Total Amount : 7.95
ROUNDING : 0.00
TOTAL : 7.95
CASH : 10.00
CHANGE : 2.05
"""


def fixture_text(name: str) -> str:
    pages = layout_pages(FIXTURES / f"{name}.pdf", "application/pdf")
    return "\n\n".join(p.text for p in pages if isinstance(p, PageContent))


def expected(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    return data


def scalars(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items() if not isinstance(v, list)}


def test_rules_extractor_is_an_extractor() -> None:
    assert isinstance(RulesExtractor(), Extractor)


@pytest.mark.parametrize("name", ["invoice_digital", "purchase_order", "contract"])
def test_fixture_scalar_fields_match_the_expected_record(name: str) -> None:
    want = expected(name)

    result = RulesExtractor().extract(fixture_text(name), want["doc_type"])

    assert result.ok and result.missing == [] and result.errors == []
    assert result.record is not None
    assert scalars(json.loads(result.record.model_dump_json())) == scalars(want)


def test_payment_advice_including_allocations() -> None:
    result = RulesExtractor().extract(fixture_text("payment_advice"), "payment")

    assert isinstance(result.record, Payment)
    assert json.loads(result.record.model_dump_json()) == expected("payment_advice")


def test_scanned_receipt_from_ocr_text() -> None:
    result = RulesExtractor().extract(RECEIPT_OCR, "receipt")

    assert isinstance(result.record, Receipt)
    got = result.record
    assert (got.receipt_number, got.receipt_date, got.currency) == (
        "R-000732",
        date(2026, 3, 14),
        "MYR",
    )
    assert (got.total, got.tax) == (Decimal("9.50"), Decimal("0.54"))
    assert got.vendor_name == "KEDAI SENTOSA"  # as read; OCR dropped "KOPI"
    assert got.vendor_address == "12 Jalan Bunga, 50450 Kuala Lumpur"


@pytest.mark.parametrize(
    "text,vendor,number,day,total,currency",
    [
        (
            SROIE_BOOKSHOP,
            "BOOK TA .K(TAMAN DAYA) SDN BHD",
            "TD01167104",
            date(2018, 12, 25),
            "9.00",
            "MYR",
        ),
        (
            SROIE_HARDWARE,
            "SYARIKAT PERNIAGAAN GIN KEE",
            "CS00012993",
            date(2018, 1, 4),
            "7.95",
            "USD",
        ),
    ],
)
def test_sroie_style_receipts(
    text: str, vendor: str, number: str, day: date, total: str, currency: str
) -> None:
    # The hardware receipt never prints "RM": with no evidence the schema default (USD) stays.
    result = RulesExtractor().extract(text, DocType.RECEIPT)

    assert isinstance(result.record, Receipt)
    record = result.record
    assert (record.vendor_name, record.receipt_number) == (vendor, number)
    assert (record.receipt_date, record.total, record.currency) == (day, Decimal(total), currency)


def test_sroie_tax_line_is_not_confused_with_the_gst_id() -> None:
    result = RulesExtractor().extract(SROIE_HARDWARE, DocType.RECEIPT)

    assert isinstance(result.record, Receipt) and result.record.tax == Decimal("0.45")


def test_missing_required_fields_are_reported_with_what_was_found() -> None:
    result = RulesExtractor().extract("INVOICE\nAcme Supplies Ltd\nInvoice no: A-17", "invoice")

    assert not result.ok
    assert result.missing == ["invoice_date", "total"]
    assert result.fields["invoice_number"] == "A-17"
    assert result.fields["vendor_name"] == "Acme Supplies Ltd"


def test_found_values_that_fail_validation_are_errors_not_crashes() -> None:
    result = ExtractionResult.from_fields(
        "rules",
        "invoice",
        {
            "invoice_number": "A-1",
            "vendor_name": "Acme",
            "invoice_date": "2026-01-02",
            "total": "x",
        },
    )

    assert not result.ok and result.errors and result.errors[0].startswith("invoice.total")


def test_unknown_documents_have_no_rules() -> None:
    result = RulesExtractor().extract("anything", DocType.UNKNOWN)

    assert not result.ok and result.errors == ["no rules for unknown"]


def test_every_fixture_gives_a_valid_record() -> None:
    texts = {
        name: fixture_text(name)
        for name in ("invoice_digital", "purchase_order", "contract", "payment_advice")
    }
    texts["receipt_scanned"] = RECEIPT_OCR

    results = {
        name: RulesExtractor().extract(text, expected(name)["doc_type"])
        for name, text in texts.items()
    }

    assert all(r.ok for r in results.values()), {n: r.missing for n, r in results.items()}
    assert isinstance(results["invoice_digital"].record, Invoice)


def test_vendor_heuristics() -> None:
    assert (
        first_line_vendor(["INVOICE", "Acme Supplies Ltd, 4 Canal Wharf, Leeds"])
        == "Acme Supplies Ltd"
    )
    assert first_line_vendor(["TAX INVOICE", "Globex  Tel: 555-0100"]) == "Globex"
    assert first_line_vendor(["Northwind Traders, 80 Market Street"]) == "Northwind Traders"
    assert first_line_vendor(["Supplier: Acme Ltd"]) is None  # labelled: someone else's name
    # A title printed beside the name on the same line is not part of it.
    assert first_line_vendor(["Northwind Traders  PURCHASE ORDER"]) == "Northwind Traders"
    assert receipt_address(["SHOP", "Order: 12"]) is None
