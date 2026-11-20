"""The 3-way match, one test per mismatch type, plus the reminder-draft helpers."""

from datetime import date
from decimal import Decimal

import pytest

from ledgerlens.core.config import AgentConfig, AgentsConfig
from ledgerlens.core.schemas import Contract, Invoice, LineItem, PurchaseOrder, RateCardEntry
from ledgerlens.mcp_server.tools.reconcile import (
    Check,
    MatchInputs,
    Tolerances,
    build_report,
    check_line_qty,
    check_payment,
    check_po_exists,
    check_terms,
    check_total,
    check_unit_price,
    check_vendor_match,
    run_checks,
)
from ledgerlens.mcp_server.tools.reminders import missing_facts, tone_for

TOL = Tolerances()
TODAY = date(2026, 3, 20)


def line(description: str, qty: str, price: str) -> LineItem:
    return LineItem(
        description=description,
        quantity=Decimal(qty),
        unit_price=Decimal(price),
        amount=Decimal(qty) * Decimal(price),
    )


WRAP, TAPE = "Pallet wrap 500m", "Packing tape 48mm"


def make_invoice(**changes: object) -> Invoice:
    values: dict[str, object] = {
        "invoice_number": "INV-889",
        "vendor_name": "Acme Supplies Ltd",
        "invoice_date": date(2026, 2, 1),
        "due_date": date(2026, 3, 3),
        "po_number": "PO-1042",
        "currency": "GBP",
        "subtotal": Decimal("143.97"),
        "tax": Decimal("28.79"),
        "total": Decimal("172.76"),
        "line_items": [line(WRAP, "2", "42.41"), line(TAPE, "1", "59.15")],
    }
    values.update(changes)
    return Invoice.model_validate(values)


PO = PurchaseOrder(
    po_number="PO-1042",
    vendor_name="Acme Supplies Ltd",
    order_date=date(2026, 1, 20),
    currency="GBP",
    total=Decimal("400.00"),
    line_items=[line(WRAP, "4", "42.41"), line(TAPE, "2", "59.15")],
)
CONTRACT = Contract(
    contract_number="CT-V001",
    vendor_name="Acme Supplies Ltd",
    start_date=date(2025, 1, 1),
    payment_terms_days=30,
    currency="GBP",
    rate_card=[
        RateCardEntry(item=WRAP, unit="roll", unit_price=Decimal("42.41")),
        RateCardEntry(item=TAPE, unit="case", unit_price=Decimal("59.15")),
    ],
)


def inputs(**changes: object) -> MatchInputs:
    values: dict[str, object] = {
        "invoice": make_invoice(),
        "invoice_id": "I-889",
        "vendor_id": "V001",
        "po": PO,
        "po_vendor_id": "V001",
        "contract": CONTRACT,
        "paid": Decimal("172.76"),
        "payments": 1,
        "today": TODAY,
    }
    values.update(changes)
    return MatchInputs(**values)  # type: ignore[arg-type]


def test_clean_invoice_matches() -> None:
    checks = run_checks(inputs(), TOL)
    assert [c.check for c in checks] == [
        "po_exists",
        "vendor_match",
        "line_qty",
        "unit_price",
        "total",
        "payment",
        "terms",
    ]
    assert {c.check: c.status for c in checks} == dict.fromkeys([c.check for c in checks], "pass")
    report = build_report(inputs(), checks, {"doc:INV-889": "d1"})
    assert (report.status, report.mismatches, report.citation) == ("match", [], "doc:INV-889")


# --------------------------------------------------------------------------- one per type


def test_po_missing_or_not_cited() -> None:
    missing = check_po_exists(inputs(po=None))
    assert (missing.status, missing.message) == ("fail", "PO-1042 is not a stored purchase order")
    uncited = check_po_exists(inputs(invoice=make_invoice(po_number=None), po=None))
    assert uncited.message == "the invoice cites no purchase order"


def test_vendor_mismatch() -> None:
    result = check_vendor_match(inputs(po_vendor_id="V009"))
    assert result.status == "fail"
    assert result.details == {"invoice_vendor": "V001", "po_vendor": "V009"}
    # unresolved vendors: compare the printed names
    by_name = check_vendor_match(inputs(vendor_id=None, po_vendor_id=None))
    assert by_name.status == "pass"


def test_quantity_over_the_po_across_its_invoices() -> None:
    earlier = make_invoice(invoice_number="INV-880", line_items=[line(WRAP, "3", "42.41")])
    result = check_line_qty(inputs(other_invoices_on_po=[earlier]), TOL)
    assert result.status == "fail"
    assert result.message == "Pallet wrap 500m: 5 invoiced vs 4 ordered"
    alone = check_line_qty(inputs(), TOL)
    assert alone.status == "pass"


def test_line_not_on_the_po() -> None:
    invoice = make_invoice(line_items=[line(WRAP, "1", "42.41"), line("Bubble wrap", "1", "9")])
    result = check_line_qty(inputs(invoice=invoice), TOL)
    assert result.status == "fail" and result.details["not_on_po"] == ["Bubble wrap"]


def test_unit_price_off_the_contract_rate() -> None:
    invoice = make_invoice(line_items=[line(WRAP, "2", "43.50"), line(TAPE, "1", "59.15")])
    result = check_unit_price(inputs(invoice=invoice), TOL)
    assert result.status == "fail"
    assert "Pallet wrap 500m at 43.50 vs 42.41 (+2.6%)" in result.message
    within = make_invoice(line_items=[line(WRAP, "2", "43.00")])  # +1.4 %: inside 2 %
    assert check_unit_price(inputs(invoice=within), TOL).status == "pass"
    assert check_unit_price(inputs(contract=None), TOL).status == "skipped"


def test_total_arithmetic() -> None:
    result = check_total(inputs(invoice=make_invoice(total=Decimal("180.00"))), TOL)
    assert result.status == "fail"
    assert "subtotal + tax = 172.76, but total is 180.00" in result.message


def test_payment_states() -> None:
    def status(paid: str, today: date = TODAY) -> tuple[str, str]:
        check = check_payment(inputs(paid=Decimal(paid), today=today), TOL)
        return check.status, check.details["payment_status"]

    assert status("172.76") == ("pass", "paid")
    assert status("200.00") == ("fail", "overpaid")
    assert status("0") == ("fail", "unpaid")  # 17 days past due
    assert status("0", date(2026, 3, 1)) == ("pass", "unpaid")  # not due yet
    assert status("100.00", date(2026, 3, 1)) == ("warn", "partial")
    overdue = check_payment(inputs(paid=Decimal("100.00")), TOL)
    assert (
        overdue.message
        == "partly paid (100.00 GBP), 72.76 GBP outstanding, 17 day(s) past due 2026-03-03"
    )


def test_terms_shorter_than_the_contract() -> None:
    rush = make_invoice(due_date=date(2026, 2, 2))
    result = check_terms(inputs(invoice=rush), TOL)
    assert result.status == "fail"
    assert result.message == "due 1 day(s) after the invoice date, but CT-V001 says Net 30"
    assert check_terms(inputs(invoice=make_invoice(due_date=None)), TOL).status == "warn"
    assert check_terms(inputs(contract=None), TOL).status == "skipped"


def test_report_lists_every_mismatch() -> None:
    m = inputs(po_vendor_id="V009", paid=Decimal(0))
    report = build_report(m, run_checks(m, TOL), {})
    assert report.status == "mismatch"
    assert report.mismatches == ["vendor_match", "payment"]
    assert report.payment_status == "unpaid" and report.citation is None


# --------------------------------------------------------------------------- tolerances


def test_tolerances_from_config() -> None:
    config = AgentsConfig(
        agents={
            "reconciler": AgentConfig.model_validate(
                {"model": "smart", "price_tolerance_pct": 5, "terms_tolerance_days": 2}
            )
        }
    )
    tol = Tolerances.from_config(config)
    assert (tol.price_pct, tol.terms_days, tol.total) == (Decimal(5), 2, Decimal("0.01"))
    invoice = make_invoice(line_items=[line(WRAP, "2", "43.50")])  # +2.6 %: inside 5 %
    assert check_unit_price(inputs(invoice=invoice), tol).status == "pass"


def test_checks_are_serializable() -> None:
    check: Check = check_line_qty(inputs(), TOL)
    assert check.model_dump(mode="json")["check"] == "line_qty"


# --------------------------------------------------------------------------- reminders


@pytest.mark.parametrize(
    ("days", "tone"),
    [
        (0, None),
        (1, "friendly"),
        (15, "friendly"),
        (16, "firm"),
        (45, "firm"),
        (46, "final_notice"),
    ],
)
def test_tone_by_days_overdue(days: int, tone: str | None) -> None:
    assert tone_for(days) == tone


def test_missing_facts() -> None:
    facts = {
        "invoice_number": "INV-889",
        "amounts": ("4200.00", "1200.00"),
        "due_date": date(2026, 4, 13),
        "po_number": "PO-1042",
    }
    complete = "Invoice INV-889 for GBP 4,200.00 (PO-1042) was due on 13 April 2026."
    assert missing_facts(complete, **facts) == []  # type: ignore[arg-type]
    assert missing_facts("Please pay INV-889.", **facts) == [  # type: ignore[arg-type]
        "amount",
        "due_date",
        "po_number",
    ]
    outstanding = "INV-889 (PO-1042), due 2026-04-13: 1,200.00 is still open."
    assert missing_facts(outstanding, **facts) == []  # type: ignore[arg-type]
