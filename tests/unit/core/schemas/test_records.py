"""Hard validation (amounts, dates, text) and consistency checks for every record type."""

from datetime import date
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from ledgerlens.core.schemas import (
    Contract,
    Invoice,
    LineItem,
    Payment,
    PurchaseOrder,
    Receipt,
)


def invoice(**overrides: Any) -> Invoice:
    fields: dict[str, Any] = {
        "invoice_number": "INV-2026-0417",
        "vendor_name": "Acme Supplies Ltd",
        "invoice_date": "2026-10-02",
        "due_date": "2026-11-01",
        "po_number": "PO-88231",
        "currency": "GBP",
        "subtotal": "415.00",
        "tax": "83.00",
        "total": "498.00",
        "line_items": [
            {
                "description": "Pallet wrap 500m",
                "quantity": 2,
                "unit_price": "45.00",
                "amount": "90.00",
            },
            {
                "description": "Shipping labels",
                "quantity": 10,
                "unit_price": "32.50",
                "amount": "325.00",
            },
        ],
    }
    return Invoice.model_validate(fields | overrides)


# --------------------------------------------------------------------------- amounts


def test_amounts_are_exact_decimals_from_strings_or_numbers() -> None:
    inv = invoice(total=498.0, tax=83)

    assert inv.total == Decimal("498") and inv.tax == Decimal("83")
    assert inv.subtotal == Decimal("415.00")
    assert inv.line_items[1].unit_price == Decimal("32.50")


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-inf", "twelve", "1,250.00", None])
def test_total_must_be_a_finite_amount(bad: object) -> None:
    with pytest.raises(ValidationError, match="total"):
        invoice(total=bad)


def test_negative_amounts_are_allowed_for_credits_but_not_quantities() -> None:
    credit = LineItem(description="Early payment discount", amount=Decimal("-10.00"))

    assert credit.amount == Decimal("-10.00")
    with pytest.raises(ValidationError, match="quantity"):
        LineItem(description="Returned", quantity=Decimal(-1), amount=Decimal(0))


def test_receipts_and_payments_reject_negative_totals() -> None:
    with pytest.raises(ValidationError):
        Receipt.model_validate({"vendor_name": "Cafe", "receipt_date": "2026-01-01", "total": "-1"})
    with pytest.raises(ValidationError):
        Payment.model_validate(
            {"payment_reference": "R", "vendor_name": "V", "paid_at": "2026-01-01", "amount": "-5"}
        )


# --------------------------------------------------------------------------- dates and text


@pytest.mark.parametrize("bad", ["2026-02-30", "02/10/2026", "next Tuesday", ""])
def test_dates_must_be_real_iso_dates(bad: str) -> None:
    with pytest.raises(ValidationError, match="invoice_date"):
        invoice(invoice_date=bad)


def test_currency_is_normalised_and_checked() -> None:
    assert invoice(currency=" gbp ").currency == "GBP"
    for bad in ("£", "POUNDS", "G1P"):
        with pytest.raises(ValidationError, match="currency"):
            invoice(currency=bad)


def test_required_text_cannot_be_blank_and_optional_blank_becomes_none() -> None:
    with pytest.raises(ValidationError, match="vendor_name"):
        invoice(vendor_name="  ")

    inv = invoice(po_number="   ", vendor_name="  Acme Supplies Ltd ")

    assert inv.po_number is None
    assert inv.vendor_name == "Acme Supplies Ltd"


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="discount_code"):
        invoice(discount_code="SAVE10")


# --------------------------------------------------------------------------- invoice totals


def test_a_consistent_invoice_has_no_issues() -> None:
    assert invoice().consistency_issues() == []


def test_line_quantity_times_price_must_match_amount() -> None:
    inv = invoice(
        line_items=[
            {"description": "Wrap", "quantity": 2, "unit_price": "45.00", "amount": "95.00"}
        ],
        subtotal="95.00",
        total="178.00",
    )

    assert inv.consistency_issues() == ["line 'Wrap': 2 x 45.00 = 90.00, but amount is 95.00"]


def test_line_items_must_add_up_to_the_subtotal() -> None:
    issues = invoice(subtotal="420.00", total="503.00").consistency_issues()

    assert issues == ["line items add up to 415.00, but subtotal is 420.00"]


def test_subtotal_plus_tax_must_equal_total() -> None:
    issues = invoice(total="500.00").consistency_issues()

    assert issues == ["subtotal + tax = 498.00, but total is 500.00"]


def test_without_a_subtotal_lines_are_checked_against_total_minus_tax() -> None:
    assert invoice(subtotal=None).consistency_issues() == []
    assert invoice(subtotal=None, total="600.00").consistency_issues() == [
        "line items add up to 415.00, but total - tax is 517.00"
    ]


def test_rounding_within_a_cent_per_line_is_not_an_issue() -> None:
    lines = [
        {"description": "Bolt", "quantity": 3, "unit_price": "0.3333", "amount": "1.00"},
        {"description": "Nut", "quantity": 3, "unit_price": "0.3333", "amount": "1.00"},
    ]

    assert (
        invoice(line_items=lines, subtotal="2.01", tax="0", total="2.01").consistency_issues() == []
    )
    assert invoice(line_items=lines, subtotal="2.03", tax="0", total="2.03").consistency_issues()


def test_due_date_before_invoice_date_is_reported() -> None:
    issues = invoice(due_date="2026-09-01").consistency_issues()

    assert issues == ["due date 2026-09-01 is before invoice date 2026-10-02"]


def test_an_invoice_without_lines_only_checks_the_header() -> None:
    inv = invoice(line_items=[], subtotal=None, tax=None, total="12.00")

    assert inv.consistency_issues() == []


def test_receipt_lines_are_checked_against_the_total() -> None:
    receipt = Receipt.model_validate(
        {
            "vendor_name": "Kedai Kopi",
            "receipt_date": "2026-03-14",
            "currency": "MYR",
            "total": "9.00",
            "line_items": [
                {"description": "Kopi O", "quantity": 3, "unit_price": "2.50", "amount": "7.50"}
            ],
        }
    )

    assert receipt.consistency_issues() == ["line items add up to 7.50, but total - tax is 9.00"]


# --------------------------------------------------------------------------- purchase orders


def test_purchase_order_checks_totals_and_delivery_date() -> None:
    po = PurchaseOrder.model_validate(
        {
            "po_number": "PO-88231",
            "vendor_name": "Acme Supplies Ltd",
            "order_date": "2026-09-20",
            "delivery_date": "2026-09-01",
            "currency": "GBP",
            "total": "100.00",
            "line_items": [
                {"description": "Wrap", "quantity": 2, "unit_price": "45", "amount": "90"}
            ],
        }
    )

    assert po.consistency_issues() == [
        "line items add up to 90, but total - tax is 100.00",
        "delivery date 2026-09-01 is before order date 2026-09-20",
    ]


# --------------------------------------------------------------------------- payments


def payment(**overrides: Any) -> Payment:
    fields: dict[str, Any] = {
        "payment_reference": "RA-5531",
        "vendor_name": "Acme Supplies Ltd",
        "paid_at": "2026-10-30",
        "method": "bank_transfer",
        "currency": "GBP",
        "amount": "748.00",
        "allocations": [
            {"invoice_number": "INV-2026-0417", "amount": "498.00"},
            {"invoice_number": "INV-2026-0420", "amount": "250.00"},
        ],
    }
    return Payment.model_validate(fields | overrides)


def test_payment_allocations_must_add_up_to_the_amount() -> None:
    assert payment().consistency_issues() == []
    assert payment().invoice_numbers == ["INV-2026-0417", "INV-2026-0420"]
    assert payment(amount="800.00").consistency_issues() == [
        "allocations add up to 748.00, but amount is 800.00"
    ]


def test_an_invoice_allocated_twice_is_reported() -> None:
    twice = [{"invoice_number": "INV-1", "amount": "1"}, {"invoice_number": "INV-1", "amount": "1"}]

    assert payment(amount="2", allocations=twice).consistency_issues() == [
        "invoice INV-1 is allocated more than once"
    ]


def test_payment_without_allocations_is_consistent() -> None:
    assert payment(allocations=[]).consistency_issues() == []


def test_payment_method_must_be_known() -> None:
    with pytest.raises(ValidationError, match="method"):
        payment(method="bitcoin")


# --------------------------------------------------------------------------- contracts


def contract(**overrides: Any) -> Contract:
    fields: dict[str, Any] = {
        "contract_number": "MSA-2026-07",
        "vendor_name": "Acme Supplies Ltd",
        "start_date": "2026-01-01",
        "end_date": "2026-12-31",
        "payment_terms_days": 30,
        "currency": "GBP",
        "rate_card": [
            {"item": "Pallet wrap 500m", "unit": "roll", "unit_price": "45.00"},
            {"item": "Courier", "unit": "hour", "unit_price": "38.00"},
            {"item": "Courier", "unit": "day", "unit_price": "260.00"},
        ],
    }
    return Contract.model_validate(fields | overrides)


def test_contract_is_consistent_and_active_within_its_dates() -> None:
    c = contract()

    assert c.consistency_issues() == []
    assert c.is_active(date(2026, 6, 1))
    assert not c.is_active(date(2027, 1, 1))
    assert contract(end_date=None).is_active(date(2030, 1, 1))


@pytest.mark.parametrize("days", [-1, 366])
def test_payment_terms_must_be_plausible(days: int) -> None:
    with pytest.raises(ValidationError, match="payment_terms_days"):
        contract(payment_terms_days=days)


def test_contract_reports_reversed_dates_and_duplicate_rates() -> None:
    rates = [
        {"item": "Courier", "unit": "hour", "unit_price": "38.00"},
        {"item": "courier", "unit": "Hour", "unit_price": "40.00"},
    ]

    assert contract(end_date="2025-12-31", rate_card=rates).consistency_issues() == [
        "end date 2025-12-31 is before start date 2026-01-01",
        "rate card prices 'courier' per hour more than once",
    ]
