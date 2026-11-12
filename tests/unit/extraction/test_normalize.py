from datetime import date
from decimal import Decimal

import pytest

from ledgerlens.extraction.normalize import (
    VendorMatcher,
    canonical_vendor,
    find_amounts,
    find_dates,
    parse_amount,
    parse_currency,
    parse_date,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("14/03/26", date(2026, 3, 14)),
        ("14/03/2026", date(2026, 3, 14)),
        ("14-03-2026", date(2026, 3, 14)),
        ("14.03.2026", date(2026, 3, 14)),
        ("2026-03-14", date(2026, 3, 14)),
        ("2026/3/4", date(2026, 3, 4)),
        ("Mar 14 2026", date(2026, 3, 14)),
        ("March 14, 2026", date(2026, 3, 14)),
        ("14 Mar 2026", date(2026, 3, 14)),
        ("14th March 2026", date(2026, 3, 14)),
        ("14-Mar-26", date(2026, 3, 14)),
        ("Sept 3 2026", date(2026, 9, 3)),
        ("2 October 2026", date(2026, 10, 2)),
        ("DATE: 25/12/2018 8:13:39 PM", date(2018, 12, 25)),  # SROIE
        ("Date 01/01/99", date(1999, 1, 1)),
    ],
)
def test_dates(text: str, expected: date) -> None:
    assert parse_date(text, day_first=True) == expected


def test_ambiguous_numeric_dates_follow_day_first() -> None:
    assert parse_date("03/04/2026", day_first=True) == date(2026, 4, 3)
    assert parse_date("03/04/2026", day_first=False) == date(2026, 3, 4)
    assert parse_date("03/04/2026") == date(2026, 4, 3)  # extraction.day_first: true
    # A part above 12 can only be the day, whatever the setting.
    assert parse_date("12/25/2026", day_first=True) == date(2026, 12, 25)
    assert parse_date("25/12/2026", day_first=False) == date(2026, 12, 25)


def test_impossible_and_missing_dates() -> None:
    assert parse_date("31/02/2026") is None
    assert parse_date("Total 9.50") is None
    assert parse_date("") is None


def test_find_dates_in_order_without_overlaps() -> None:
    text = "This agreement runs from 1 January 2026 to 31 December 2026 (signed 2025-12-20)."

    assert find_dates(text) == [date(2026, 1, 1), date(2026, 12, 31), date(2025, 12, 20)]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("$1,234.50", "1234.50"),
        ("1.234,50", "1234.50"),
        ("1 234,50", "1234.50"),
        ("1,234,567.89", "1234567.89"),
        ("RM 9.50", "9.50"),
        ("RM9.50", "9.50"),
        ("£498.00", "498.00"),
        ("EUR 12,5", "12.5"),
        ("(12.00)", "-12.00"),
        ("-12.00", "-12.00"),
        ("1,234", "1234"),
        ("415", "415"),
        ("TOTAL: 9.00, INCL. GST", "9.00"),  # SROIE
    ],
)
def test_amounts(text: str, expected: str) -> None:
    assert parse_amount(text) == Decimal(expected)


def test_percentages_and_missing_amounts() -> None:
    assert find_amounts("VAT 20%  83.00") == [Decimal("83.00")]
    assert find_amounts("incl. SST 6%  0.54") == [Decimal("0.54")]
    assert parse_amount("no amount here") is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Total (GBP)  498.00", "GBP"),
        ("TOTAL MYR 9.50", "MYR"),
        ("ROUND D TOTAL (RM): 9.00", "MYR"),
        ("RM9.50", "MYR"),
        ("£3.00", "GBP"),
        ("€4", "EUR"),
        ("S$12", "SGD"),
        ("$5", "USD"),
        ("Kuala Lumpur 50450", None),
    ],
)
def test_currency(text: str, expected: str | None) -> None:
    assert parse_currency(text) == expected


def test_bare_dollar_uses_the_given_default() -> None:
    assert parse_currency("$5", default_dollar="AUD") == "AUD"


@pytest.mark.parametrize(
    "name,key",
    [
        ("Acme Supplies Ltd", "acme supplies"),
        ("ACME SUPPLIES LIMITED.", "acme supplies"),
        ("Acme Supplies, Inc.", "acme supplies"),
        ("BOOK TA .K(TAMAN DAYA) SDN BHD", "book ta k taman daya"),
        ("Smith & Sons Co.", "smith and sons"),
        ("Siemens GmbH", "siemens"),
        ("Pinnacle Pte. Ltd.", "pinnacle"),
        ("  ", ""),
    ],
)
def test_canonical_vendor(name: str, key: str) -> None:
    assert canonical_vendor(name) == key


def test_vendor_matcher() -> None:
    matcher = VendorMatcher(
        {
            "Acme Supplies Ltd": ["Acme"],
            "Book Ta .K (Taman Daya) Sdn Bhd": ["Book TA K"],
            "Northwind Traders": [],
        }
    )

    assert matcher.match("ACME SUPPLIES LIMITED") == "Acme Supplies Ltd"  # exact key
    assert matcher.match("acme") == "Acme Supplies Ltd"  # alias
    assert matcher.match("ACME SUPPLES LTD") == "Acme Supplies Ltd"  # OCR slip
    assert matcher.match("Traders Northwind") == "Northwind Traders"  # word order
    assert matcher.match("BOOK TA .K(TAMAN DAYA) SDN BHD") == "Book Ta .K (Taman Daya) Sdn Bhd"
    assert matcher.match("Globex Corporation") is None
    assert matcher.match("") is None
