"""Turn the way documents write dates, amounts and currencies into typed values.

Every extractor runs its raw strings through these functions, and evaluation compares
values *after* normalization, so ``14/03/26``, ``Mar 14 2026`` and ``2026-03-14`` all count
as the same date.

Dates
    ISO (``2026-03-14``), numeric with ``/ - .`` (``14/03/26``, ``14.03.2026``), and
    month names (``14 Mar 2026``, ``March 14, 2026``, ``14-Mar-26``). Numeric dates are
    day-first unless ``extraction.day_first`` is false; a part above 12 settles it either
    way. Two-digit years are 2000-2069 / 1970-1999.
Amounts
    ``$1,234.50``, ``1.234,50``, ``1 234,50``, ``RM 9.50``, ``(12.00)`` and ``-12.00``.
    With both ``,`` and ``.`` the last one is the decimal point; with one of them, it is a
    decimal point when one or two digits follow it, otherwise a thousands separator.
Currency
    ISO 4217 codes and common symbols (``RM`` -> MYR, ``£`` -> GBP, ...). A bare ``$`` is
    ambiguous and maps to ``default_dollar`` (USD).
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation

from ledgerlens.core.config import get_config

_MONTHS = {
    name: number
    for number, names in enumerate(
        [
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ],
        start=1,
    )
    for name in names
}
_MONTH = (
    r"(?P<mon>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
)
_DAY = r"(?P<day>\d{1,2})(?:st|nd|rd|th)?"
_YEAR = r"(?P<year>\d{4}|\d{2})"

DATE_PATTERNS = [
    re.compile(r"\b(?P<year>\d{4})[-/.](?P<a>\d{1,2})[-/.](?P<b>\d{1,2})\b"),  # ISO-ish
    re.compile(r"\b(?P<a>\d{1,2})[-/.](?P<b>\d{1,2})[-/.]" + _YEAR + r"\b"),
    re.compile(r"\b" + _DAY + r"[\s\-/]+" + _MONTH + r"[\s\-/,]+" + _YEAR + r"\b", re.I),
    re.compile(r"\b" + _MONTH + r"\s+" + _DAY + r",?\s+" + _YEAR + r"\b", re.I),
]


def _year(text: str) -> int:
    value = int(text)
    if len(text) == 2:
        return 2000 + value if value < 70 else 1900 + value
    return value


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _match_to_date(match: re.Match[str], day_first: bool) -> date | None:
    groups = match.groupdict()
    year = _year(groups["year"])
    if groups.get("mon"):
        return _safe_date(year, _MONTHS[groups["mon"].lower().rstrip(".")], int(groups["day"]))
    a, b = int(groups["a"]), int(groups["b"])
    if len(groups["year"]) == 4 and match.group(0).startswith(groups["year"]):
        return _safe_date(year, a, b)  # year first: always Y-M-D
    if a > 12 >= b:
        day_first = True
    elif b > 12 >= a:
        day_first = False
    return _safe_date(year, b, a) if day_first else _safe_date(year, a, b)


def find_dates(text: str, *, day_first: bool | None = None) -> list[date]:
    """Every date written in ``text``, in the order they appear."""
    if day_first is None:
        day_first = get_config().ingestion.extraction.day_first
    found: list[tuple[int, date]] = []
    taken: list[tuple[int, int]] = []
    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(text):
            span = match.span()
            if any(start < span[1] and span[0] < end for start, end in taken):
                continue
            value = _match_to_date(match, day_first)
            if value is not None:
                found.append((span[0], value))
                taken.append(span)
    return [value for _, value in sorted(found)]


def parse_date(text: str, *, day_first: bool | None = None) -> date | None:
    """The first date in ``text``, or ``None``."""
    dates = find_dates(text, day_first=day_first)
    return dates[0] if dates else None


# --------------------------------------------------------------------------- amounts

# A number with optional thousands groups / decimals, optionally signed or in parentheses.
AMOUNT = re.compile(
    r"(?P<neg>[-(])?\s?(?P<num>\d{1,3}(?:[,.\s]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)"
    r"(?P<close>\))?(?!\s?%)(?![\d/])"
)


def parse_amount(text: str) -> Decimal | None:
    """The amount written in ``text`` (currency symbols and codes are ignored)."""
    match = AMOUNT.search(text)
    if match is None:
        return None
    return _amount_from_match(match)


def find_amounts(text: str) -> list[Decimal]:
    """Every amount in ``text``, in order (percentages are skipped)."""
    amounts = []
    for match in AMOUNT.finditer(text):
        value = _amount_from_match(match)
        if value is not None:
            amounts.append(value)
    return amounts


def _amount_from_match(match: re.Match[str]) -> Decimal | None:
    number = _canonical_number(match["num"])
    try:
        value = Decimal(number)
    except InvalidOperation:
        return None
    negative = match["neg"] == "-" or (match["neg"] == "(" and match["close"] == ")")
    return -value if negative else value


def _canonical_number(raw: str) -> str:
    digits = raw.replace(" ", "")
    last_dot, last_comma = digits.rfind("."), digits.rfind(",")
    if last_dot >= 0 and last_comma >= 0:
        decimal_sep = "." if last_dot > last_comma else ","
    elif last_dot >= 0 or last_comma >= 0:
        sep = "." if last_dot >= 0 else ","
        after = len(digits) - max(last_dot, last_comma) - 1
        decimal_sep = sep if digits.count(sep) == 1 and after in (1, 2) else ""
    else:
        decimal_sep = ""
    thousands = {".", ","} - {decimal_sep}
    for sep in thousands:
        digits = digits.replace(sep, "")
    return digits.replace(decimal_sep, ".") if decimal_sep else digits


# --------------------------------------------------------------------------- currency

ISO_CURRENCIES = frozenset(
    [
        "USD",
        "EUR",
        "GBP",
        "JPY",
        "CNY",
        "INR",
        "MYR",
        "SGD",
        "AUD",
        "CAD",
        "CHF",
        "HKD",
        "NZD",
        "SEK",
        "NOK",
        "DKK",
        "ZAR",
        "AED",
        "SAR",
        "THB",
        "IDR",
        "PHP",
        "VND",
        "KRW",
        "BRL",
        "MXN",
        "PLN",
        "CZK",
        "HUF",
        "TRY",
    ]
)
_SYMBOLS = [
    (re.compile(r"\bRM(?=\s?\d|\b)"), "MYR"),
    (re.compile(r"S\$"), "SGD"),
    (re.compile(r"A\$"), "AUD"),
    (re.compile(r"C\$"), "CAD"),
    (re.compile(r"HK\$"), "HKD"),
    (re.compile(r"£"), "GBP"),
    (re.compile(r"€"), "EUR"),
    (re.compile(r"¥"), "JPY"),
    (re.compile(r"₹"), "INR"),
    (re.compile(r"\$"), "$"),
]
_ISO = re.compile(r"\b(" + "|".join(sorted(ISO_CURRENCIES)) + r")\b")


def parse_currency(text: str, *, default_dollar: str = "USD") -> str | None:
    """The currency ``text`` is in: the first ISO code, else the first known symbol."""
    iso = _ISO.search(text)
    if iso:
        return iso.group(1)
    for pattern, code in _SYMBOLS:
        if pattern.search(text):
            return default_dollar if code == "$" else code
    return None
