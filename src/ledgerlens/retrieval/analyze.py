"""Query analysis: pull entities, filters and intent out of a question.

:class:`QueryAnalyzer` reads a question the way the retrieval layer needs it
(docs/layers/03-retrieval.md, "Analyze"):

* **document numbers** -- ``INV-889``, ``PO-1042``, ``PAY-2025-000227``, ``CT-V001``, ...
  (upper-cased; the prefix also says which document type is meant);
* **vendors** -- names matched against the vendor master: exact canonical name or alias
  ("ACME Supplies, Ltd." = "Acme Supplies Limited"), a distinctive first word ("Acme"),
  or a fuzzy match on a word n-gram (``rapidfuzz``, for typos);
* **date range** -- ``Q1`` / ``Q1 2026``, ``FY2026``, ``last month``, ``this quarter``,
  ``in March``, ``March 2026``, ``last 30 days``, ``since``/``before``/``after`` a date,
  ``between`` two dates, or one explicit date -- resolved against ``today``.

Everything above is rules: fast, free and predictable.
"""

from __future__ import annotations

import calendar
import logging
import re
from collections.abc import Iterable
from datetime import date, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict
from rapidfuzz import fuzz
from sqlalchemy.orm import Session, sessionmaker

from ledgerlens.core.schemas import DocType
from ledgerlens.extraction.normalize import canonical_vendor, find_dates
from ledgerlens.retrieval import QueryFilters
from ledgerlens.storage.neo4j.linker import VendorCandidate
from ledgerlens.storage.postgres.db import get_session_factory
from ledgerlens.storage.postgres.repositories import VendorRepository

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- results


class QueryAnalysis(BaseModel):
    """What a question asks for. ``method`` says whether the rules or the LLM decided."""

    model_config = ConfigDict(frozen=True)

    query: str
    doc_numbers: tuple[str, ...] = ()
    doc_types: tuple[DocType, ...] = ()
    vendor_ids: tuple[str, ...] = ()
    vendor_names: tuple[str, ...] = ()
    """The master names of ``vendor_ids``, in the same order."""
    date_from: date | None = None
    date_to: date | None = None
    needs_graph: bool = False
    aggregation: bool = False
    method: Literal["rules", "llm"] = "rules"

    @property
    def found_anything(self) -> bool:
        return bool(
            self.doc_numbers
            or self.doc_types
            or self.vendor_ids
            or self.date_from
            or self.date_to
            or self.needs_graph
            or self.aggregation
        )

    def filters(self) -> QueryFilters:
        """Store filters for the question.

        Vendor and date range always apply. Document numbers narrow the search only for a
        plain lookup: a question about what is *linked* to ``INV-889``, or a total over
        several documents, needs documents whose own number is not ``INV-889``. For the
        same reason a relationship question is not narrowed to the types it names.
        """
        lookup = not (self.needs_graph or self.aggregation)
        return QueryFilters(
            doc_types=() if self.needs_graph else self.doc_types,
            vendor_ids=self.vendor_ids,
            doc_numbers=self.doc_numbers if lookup else (),
            date_from=self.date_from,
            date_to=self.date_to,
        )


# --------------------------------------------------------------------------- document numbers

_DOC_NUMBER = re.compile(
    r"(?<![A-Za-z0-9])(?P<prefix>SINV|INV|PO|PAY|RA|CT|MSA|RCPT|REC)"
    r"[-/#]?(?P<body>\d[A-Za-z0-9]*(?:[-/][A-Za-z0-9]+)*|[A-Za-z]\d[A-Za-z0-9]*(?:[-/][A-Za-z0-9]+)*)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
PREFIX_TYPES: dict[str, DocType] = {
    "INV": DocType.INVOICE,
    "SINV": DocType.INVOICE,
    "PO": DocType.PURCHASE_ORDER,
    "PAY": DocType.PAYMENT,
    "RA": DocType.PAYMENT,
    "CT": DocType.CONTRACT,
    "MSA": DocType.CONTRACT,
    "RCPT": DocType.RECEIPT,
    "REC": DocType.RECEIPT,
}


def find_doc_numbers(text: str) -> list[tuple[str, DocType]]:
    """Document numbers in ``text`` with the type their prefix names, in order, once each.

    Numbers are upper-cased (``inv-889`` -> ``INV-889``) but otherwise kept as written,
    separator included: they are matched exactly against stored numbers.
    """
    found: dict[str, DocType] = {}
    for match in _DOC_NUMBER.finditer(text):
        number = match.group(0).upper()
        found.setdefault(number, PREFIX_TYPES[match["prefix"].upper()])
    return list(found.items())


# --------------------------------------------------------------------------- vendors

# First words too generic to name a vendor on their own ("Global" Solutions, "The" ...).
_GENERIC = frozenset(
    [
        "the",
        "global",
        "general",
        "national",
        "international",
        "united",
        "first",
        "new",
        "best",
        "prime",
        "premier",
        "royal",
        "city",
        "metro",
        "central",
        "north",
        "south",
        "east",
        "west",
        "asia",
        "pacific",
        "euro",
    ]
)
_WORD = re.compile(r"[A-Za-z0-9&']+")


class VendorFinder:
    """Finds vendors of the master in free text.

    A vendor is found when the question contains its canonical name or an alias (as
    whole words), its first word when that word is distinctive (unique among vendors, at
    least four letters, not generic -- "Acme" for "Acme Supplies Ltd"), or a word n-gram
    within ``threshold`` (rapidfuzz ratio) of a name -- for typos ("Acme Suplies").
    """

    def __init__(self, vendors: Iterable[VendorCandidate], *, threshold: float = 88.0) -> None:
        self.vendors = list(vendors)
        self.threshold = threshold
        self.keys: dict[str, str] = {}  # canonical name/alias -> vendor id
        for vendor in self.vendors:
            for name in vendor.names():
                key = canonical_vendor(name)
                if key:
                    self.keys.setdefault(key, vendor.id)
        firsts: dict[str, set[str]] = {}
        for key, vendor_id in self.keys.items():
            firsts.setdefault(key.split()[0], set()).add(vendor_id)
        self.short = {
            word: ids.pop()
            for word, ids in firsts.items()
            if len(ids) == 1 and len(word) >= 4 and word not in _GENERIC and not word.isdigit()
        }
        self.names = {vendor.id: vendor.name for vendor in self.vendors}

    def find(self, text: str) -> list[str]:
        """Vendor ids mentioned in ``text``, in order of first mention."""
        words = [w.lower() for w in _WORD.findall(text)]
        hits: dict[str, int] = {}  # vendor id -> first word position
        longest = max((len(k.split()) for k in self.keys), default=0)
        for size in range(min(longest, len(words)), 0, -1):
            for start in range(len(words) - size + 1):
                gram = canonical_vendor(" ".join(words[start : start + size]))
                if not gram:
                    continue
                vendor_id = self.keys.get(gram)
                if vendor_id is None and size == 1:
                    vendor_id = self.short.get(gram)
                if vendor_id is None and size > 1 and len(gram) >= 8:
                    vendor_id = self._fuzzy(gram)
                if vendor_id is not None and vendor_id not in hits:
                    hits[vendor_id] = start
        return sorted(hits, key=lambda vendor_id: hits[vendor_id])

    def _fuzzy(self, gram: str) -> str | None:
        best: tuple[float, str] | None = None
        for key, vendor_id in self.keys.items():
            if abs(len(key) - len(gram)) > max(3, len(key) // 4):
                continue
            score = fuzz.ratio(gram, key)
            if score >= self.threshold and (best is None or score > best[0]):
                best = (score, vendor_id)
        return best[1] if best else None


# --------------------------------------------------------------------------- dates

DateRange = tuple[date | None, date | None]

_MONTH_NAMES = {
    name.lower(): number
    for number in range(1, 13)
    for name in (calendar.month_name[number], calendar.month_abbr[number])
}
_MONTH_NAMES["sept"] = 9
_MONTH = "|".join(sorted(_MONTH_NAMES, key=len, reverse=True))
_YEAR = r"(?:19|20)\d\d"

_QUARTER = re.compile(
    rf"\b(?:q([1-4])|([1-4])(?:st|nd|rd|th)\s+quarter)(?:\s*(?:of\s+)?'?({_YEAR}))?\b", re.I
)
_FISCAL_YEAR = re.compile(rf"\b(?:fy|fiscal(?:\s+year)?)\s*'?({_YEAR}|\d\d)\b", re.I)
_MONTH_YEAR = re.compile(
    rf"\b(?:in\s+|during\s+|for\s+)?({_MONTH})\.?(?:\s*,?\s*({_YEAR}))?\b", re.I
)
_RELATIVE = re.compile(
    r"\b(this|last|previous|past|current|next)\s+(week|month|quarter|year)\b", re.I
)
_LAST_N = re.compile(r"\b(?:last|past|previous)\s+(\d{1,3})\s+(day|week|month)s?\b", re.I)
_YEAR_ONLY = re.compile(rf"\b(?:in|during|for|of|year)\s+({_YEAR})\b", re.I)
_TODAY = re.compile(
    r"\b(today|yesterday|ytd|year[\s-]to[\s-]date|mtd|month[\s-]to[\s-]date)\b", re.I
)
_SINCE = re.compile(r"\b(since|after|from|starting)\b", re.I)
_UNTIL = re.compile(r"\b(before|until|till|up\s+to|through|by)\b", re.I)


def month_range(year: int, month: int) -> tuple[date, date]:
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def quarter_range(year: int, quarter: int, *, fiscal_start: int = 1) -> tuple[date, date]:
    """Quarter ``quarter`` of fiscal year ``year`` (a fiscal year starting in month
    ``fiscal_start`` is named by the calendar year it ends in, as FY2026 = Apr 2025 to
    Mar 2026 when it starts in April)."""
    first = (fiscal_start - 1) + (quarter - 1) * 3  # months after January of the start year
    start_year = year - (1 if fiscal_start > 1 else 0) + first // 12
    start = date(start_year, first % 12 + 1, 1)
    end_month = start.month + 2
    end_year = start.year + (end_month - 1) // 12
    end_month = (end_month - 1) % 12 + 1
    return start, month_range(end_year, end_month)[1]


def fiscal_quarter_of(day: date, fiscal_start: int = 1) -> tuple[int, int]:
    """(fiscal year, quarter) that ``day`` falls in."""
    offset = (day.month - fiscal_start) % 12
    fiscal_year = day.year + (1 if fiscal_start > 1 and day.month >= fiscal_start else 0)
    return fiscal_year, offset // 3 + 1


def _shift_months(year: int, month: int, delta: int) -> tuple[int, int]:
    index = year * 12 + (month - 1) + delta
    return index // 12, index % 12 + 1


def parse_date_range(text: str, today: date, *, fiscal_start: int = 1) -> DateRange:
    """The date range a question talks about, or ``(None, None)``.

    The first matching form wins, most specific first: explicit dates (one = that day,
    two = between them; ``since``/``after`` or ``before``/``until`` make one date an open
    range), quarters, fiscal years, relative periods, months, then a bare year. A month or
    quarter without a year is the latest one that has started by ``today``.
    """
    dates = sorted(find_dates(text))
    if len(dates) >= 2:
        return dates[0], dates[-1]
    if len(dates) == 1:
        (day,) = dates
        if _SINCE.search(text):
            return (day + timedelta(days=1) if re.search(r"\bafter\b", text, re.I) else day), None
        if _UNTIL.search(text):
            return None, (day - timedelta(days=1) if re.search(r"\bbefore\b", text, re.I) else day)
        return day, day

    if match := _QUARTER.search(text):
        quarter = int(match[1] or match[2])
        if match[3]:
            return quarter_range(int(match[3]), quarter, fiscal_start=fiscal_start)
        fiscal_year, _ = fiscal_quarter_of(today, fiscal_start)
        start, end = quarter_range(fiscal_year, quarter, fiscal_start=fiscal_start)
        if start > today:  # "Q4" asked in February is last year's
            start, end = quarter_range(fiscal_year - 1, quarter, fiscal_start=fiscal_start)
        return start, end

    if match := _FISCAL_YEAR.search(text):
        year = int(match[1])
        year += 2000 if year < 100 else 0
        start, _ = quarter_range(year, 1, fiscal_start=fiscal_start)
        _, end = quarter_range(year, 4, fiscal_start=fiscal_start)
        return start, end

    if match := _TODAY.search(text):
        word = match[1].lower().replace("-", " ")
        if word == "today":
            return today, today
        if word == "yesterday":
            return today - timedelta(days=1), today - timedelta(days=1)
        if word in ("ytd", "year to date"):
            return date(today.year, 1, 1), today
        return date(today.year, today.month, 1), today  # month to date

    if match := _LAST_N.search(text):
        n, unit = int(match[1]), match[2].lower()
        if unit == "month":
            year, month = _shift_months(today.year, today.month, -n)
            last = calendar.monthrange(year, month)[1]
            return date(year, month, min(today.day, last)), today
        return today - timedelta(days=n * (7 if unit == "week" else 1)), today

    if match := _RELATIVE.search(text):
        which, unit = match[1].lower(), match[2].lower()
        step = {"this": 0, "current": 0, "last": -1, "previous": -1, "past": -1, "next": 1}[which]
        if unit == "week":
            monday = today - timedelta(days=today.weekday()) + timedelta(weeks=step)
            return monday, monday + timedelta(days=6)
        if unit == "month":
            return month_range(*_shift_months(today.year, today.month, step))
        if unit == "quarter":
            fiscal_year, quarter = fiscal_quarter_of(today, fiscal_start)
            index = fiscal_year * 4 + (quarter - 1) + step
            return quarter_range(index // 4, index % 4 + 1, fiscal_start=fiscal_start)
        return date(today.year + step, 1, 1), date(today.year + step, 12, 31)

    if match := _MONTH_YEAR.search(text):
        month = _MONTH_NAMES[match[1].lower()]
        if match[2]:
            return month_range(int(match[2]), month)
        if match[0][:2].lower() in ("in", "du", "fo") or len(match[1]) > 3:
            year = today.year if month <= today.month else today.year - 1
            return month_range(year, month)

    if match := _YEAR_ONLY.search(text):
        year = int(match[1])
        return date(year, 1, 1), date(year, 12, 31)
    return None, None


# --------------------------------------------------------------------------- the analyzer


class QueryAnalyzer:
    """Rules over the question text; nothing is sent anywhere."""

    def __init__(
        self,
        vendors: Iterable[VendorCandidate] = (),
        *,
        today: date | None = None,
        fiscal_start: int = 1,
        vendor_threshold: float = 88.0,
    ) -> None:
        self.finder = VendorFinder(vendors, threshold=vendor_threshold)
        self._today = today
        self.fiscal_start = fiscal_start

    @classmethod
    def from_postgres(
        cls, sessions: sessionmaker[Session] | None = None, **kwargs: Any
    ) -> QueryAnalyzer:
        """An analyzer that knows the vendors in the Postgres ``vendors`` table."""
        with (sessions or get_session_factory())() as session:
            vendors = [
                VendorCandidate(row.id, row.canonical_name, row.tax_id, tuple(row.aliases))
                for row in VendorRepository(session).all()
            ]
        return cls(vendors, **kwargs)

    @property
    def today(self) -> date:
        return self._today or date.today()

    def analyze(self, question: str) -> QueryAnalysis:
        return self.rules(question)

    def rules(self, question: str) -> QueryAnalysis:
        numbers = find_doc_numbers(question)
        doc_numbers = [number for number, _ in numbers]
        rest = _DOC_NUMBER.sub(" ", question)  # numbers must not read as dates or vendors
        vendor_ids = self.finder.find(rest)
        start, end = parse_date_range(rest, self.today, fiscal_start=self.fiscal_start)
        all_types = list(dict.fromkeys(t for _, t in numbers))
        return QueryAnalysis(
            query=question,
            doc_numbers=tuple(doc_numbers),
            doc_types=tuple(all_types),
            vendor_ids=tuple(vendor_ids),
            vendor_names=tuple(self.finder.names[v] for v in vendor_ids),
            date_from=start,
            date_to=end,
        )
