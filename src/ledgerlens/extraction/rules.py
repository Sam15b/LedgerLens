"""Regex baseline extractor: labelled fields, totals and dates, no model.

The number every learned extractor has to beat. It reads the document line by line and
looks for the labels documents use ("Invoice no:", "Your PO number", "TOTAL MYR",
"Payment date"), normalizing what follows with :mod:`ledgerlens.extraction.normalize`.
Fields without a label fall back to heuristics: the vendor is the first line that is not
the document's title (cut after its legal suffix, or before the address), and the
document date is the first date anywhere. Line items, rate cards and other tables are left
to the LLM extractor.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from datetime import date
from decimal import Decimal
from typing import Any

from ledgerlens.core.schemas import DocType
from ledgerlens.extraction.base import ExtractionResult
from ledgerlens.extraction.normalize import (
    LEGAL_SUFFIXES,
    find_amounts,
    find_dates,
    parse_currency,
)

_I = re.IGNORECASE
_ID = r"(?P<id>[A-Z0-9][A-Z0-9\-/_.]*\d[A-Z0-9\-/_]*)"  # an identifier with a digit in it
_SEP = r"\s*(?:no\.?|number|num|#|ref(?:erence)?)?\s*[:.#]?\s*"

INVOICE_NUMBER = [
    re.compile(r"\b(?:tax\s+)?invoice" + _SEP + _ID, _I),
    re.compile(r"\b(?:bill|inv)\s*(?:no\.?|number|#)\s*[:.#]?\s*" + _ID, _I),
    re.compile(r"\bdocument\s*(?:no\.?|number)\s*[:.]?\s*" + _ID, _I),
]
RECEIPT_NUMBER = [
    re.compile(r"\breceipt" + _SEP + _ID, _I),
    re.compile(
        r"\b(?:bill|invoice|inv|trans(?:action)?|doc(?:ument)?)\s*(?:no\.?|number|#)\s*[:.#]?\s*"
        + _ID,
        _I,
    ),
]
PO_NUMBER = [
    re.compile(r"\b(?:p\.?o\.?|purchase\s+order)" + _SEP + _ID, _I),
    re.compile(r"\b(?P<id>PO[-\s]?\d{3,}[\w\-/]*)\b", _I),
]
CONTRACT_NUMBER = [
    re.compile(r"\b(?:contract|agreement)" + _SEP + _ID, _I),
    re.compile(r"\bunder\s+contract\s*[:.]?\s*" + _ID, _I),
]
PAYMENT_REFERENCE = [
    re.compile(
        r"\b(?:payment|remittance|transfer)\s+(?:ref(?:erence)?|no\.?|number|id)\s*[:.#]?\s*" + _ID,
        _I,
    ),
    re.compile(r"\bref(?:erence)?\s*[:.#]\s*" + _ID, _I),
]
TAX_ID = re.compile(
    r"\b(?:vat|gst|sst|tax|abn|ein|tin|uen|co\.?)\s*(?:reg(?:istration)?\.?|id)?\s*"
    r"(?:no\.?|number|#|id)?\s*[:.]?\s*(?P<id>[A-Z]{0,3}\s?\d[\d\s-]{5,}\d[A-Z]?)\b",
    _I,
)
_TITLE_WORDS = re.compile(
    r"^\s*(?:tax\s+)?(?:invoice|receipt|purchase\s+order|remittance\s+advice|payment\s+advice|"
    r"official\s+receipt|cash\s+(?:bill|sale)|statement|agreement|contract)\b.*$",
    _I,
)
_LEGAL_END = re.compile(
    r"^(?P<name>.*?\b(?:"
    + "|".join(s.replace(" ", r"\s+") for s in LEGAL_SUFFIXES if len(s) > 2)
    + r")\b\.?)",
    _I,
)

# Total-like labels, best first. Each is (label pattern, lines to skip).
_TOTAL_LABELS = [
    re.compile(
        r"\b(?:grand\s+total|total\s+(?:amount\s+)?(?:due|payable)|amount\s+due|balance\s+due)\b",
        _I,
    ),
    re.compile(r"\btotal\s*(?:amount|rm|incl|\(|[A-Z]{3}\b|:)", _I),
    re.compile(r"\btotal\b", _I),
]
_NOT_TOTAL = re.compile(
    r"\b(?:sub[\s-]?total|total\s+(?:qty|quantity|items?|discount|tax|vat|gst|savings?|paid|"
    r"excl)|rounding)\b",
    _I,
)
_SUBTOTAL = re.compile(r"\bsub[\s-]?total\b", _I)
_TAX_LINE = re.compile(r"\b(?:vat|gst|sst|tax)\b", _I)
_NOT_TAX = re.compile(r"\b(?:reg(?:istration)?|id|no\.?|number|invoice|excl(?:uding)?)\b", _I)


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _first_id(lines: Sequence[str], patterns: Sequence[re.Pattern[str]]) -> str | None:
    for pattern in patterns:
        for line in lines:
            match = pattern.search(line)
            if match:
                return match["id"].strip(".,:;")
    return None


def _labelled(lines: Sequence[str], label: str) -> str | None:
    """Text after ``label:`` on the first line that has it (up to a double space)."""
    pattern = re.compile(r"\b(?:" + label + r")\s*[:\-]\s*(?P<v>.+?)(?:\s{2,}|$)", _I)
    for line in lines:
        match = pattern.search(line)
        if match:
            return match["v"].strip()
    return None


def _labelled_date(lines: Sequence[str], label: str) -> date | None:
    pattern = re.compile(r"\b(?:" + label + r")\b", _I)
    for line in lines:
        match = pattern.search(line)
        if match:
            dates = find_dates(line[match.end() :])
            if dates:
                return dates[0]
    return None


def _first_date(lines: Sequence[str]) -> date | None:
    for line in lines:
        dates = find_dates(line)
        if dates:
            return dates[0]
    return None


def _amount_on(
    lines: Sequence[str], label: re.Pattern[str], skip: re.Pattern[str] | None
) -> Decimal | None:
    """The last amount on the last line matching ``label`` (and not ``skip``)."""
    for line in reversed(lines):
        match = label.search(line)
        if match and not (skip and skip.search(line)):
            amounts = find_amounts(line[match.end() :])
            if amounts:
                return amounts[-1]
    return None


def find_total(lines: Sequence[str], label: re.Pattern[str] | None = None) -> Decimal | None:
    """The amount on the best total-like line; with ``label``, on the last line matching it."""
    if label is not None:
        return _amount_on(lines, label, None)
    for pattern in _TOTAL_LABELS:
        value = _amount_on(lines, pattern, _NOT_TOTAL)
        if value is not None:
            return value
    return None


def find_tax(lines: Sequence[str]) -> Decimal | None:
    return _amount_on(lines, _TAX_LINE, _NOT_TAX)


def find_subtotal(lines: Sequence[str]) -> Decimal | None:
    return _amount_on(lines, _SUBTOTAL, None)


# A line that starts with a label ("Supplier: ...", "TEL : ...", "GST ID : ..."): up to four
# single-spaced words, then a colon. "Globex  Tel: ..." is a name followed by a label.
_LABEL_START = re.compile(r"^[A-Za-z][\w.]*(?: [\w.]+){0,3} ?:")


def _vendor_line(lines: Sequence[str]) -> int | None:
    """Index of the line most likely to be the issuer's name, among the first six.

    A line with a legal suffix (``SDN BHD``, ``Ltd``) wins -- SROIE receipts often print
    the outlet or owner's name above the company -- otherwise the first line that is not
    the document's title. Labelled lines ("Supplier: ...") are someone else's name.
    """
    candidates = [
        i
        for i, line in enumerate(lines[:6])
        if not _TITLE_WORDS.match(line)
        and not _LABEL_START.match(line)
        and re.search(r"[A-Za-z]{2}", line)
    ]
    for i in candidates:
        if _LEGAL_END.match(lines[i]):
            return i
    return candidates[0] if candidates else None


def first_line_vendor(lines: Sequence[str]) -> str | None:
    """The first line that is not a title, cut after its legal suffix or before a comma.

    Wide gaps inside the name (OCR often leaves them) are closed up; the name stops at the
    first gap-separated part with a digit or a colon ("Tel: ...", "GST No ...").
    """
    index = _vendor_line(lines)
    if index is None:
        return None
    line = lines[index]
    legal = _LEGAL_END.match(line)
    name = legal["name"] if legal else line.split(",")[0]
    parts: list[str] = []
    for part in re.split(r"\s{2,}", name):
        if parts and re.search(r"[\d:]", part):
            break
        parts.append(part)
    return " ".join(parts).strip(" ,.-") or None


_ADDRESS_HINT = re.compile(r"\d.*[A-Za-z]{3}|[A-Za-z]{3}.*\d")


def receipt_address(lines: Sequence[str]) -> str | None:
    """The line under the vendor's name if it reads like an address (words and numbers,
    no label)."""
    index = _vendor_line(lines)
    if index is None or index + 1 >= len(lines):
        return None
    line = lines[index + 1]
    if ":" in line or not _ADDRESS_HINT.search(line) or find_dates(line):
        return None
    return re.sub(r"\s{2,}", " ", line).strip()


def _party(lines: Sequence[str], labels: str) -> str | None:
    value = _labelled(lines, labels)
    if value is None:
        return None
    legal = _LEGAL_END.match(value)
    return (legal["name"] if legal else value.split(",")[0]).strip(" ,.")


# --------------------------------------------------------------------------- per doc type


def _invoice(lines: list[str], text: str) -> dict[str, Any]:
    return {
        "invoice_number": _first_id(lines, INVOICE_NUMBER),
        "vendor_name": _party(lines, r"vendor|supplier|seller|from") or first_line_vendor(lines),
        "vendor_tax_id": _tax_id(lines),
        "customer_name": _party(lines, r"bill\s+to|sold\s+to|customer|invoice\s+to"),
        "invoice_date": _labelled_date(lines, r"invoice\s+date|date\s+of\s+issue|issue\s+date|date")
        or _first_date(lines),
        "due_date": _labelled_date(lines, r"due\s+date|payment\s+due|due\s+by|pay\s+by"),
        "po_number": _first_id(lines, PO_NUMBER),
        "currency": parse_currency(text),
        "subtotal": find_subtotal(lines),
        "tax": find_tax(lines),
        "total": find_total(lines),
    }


def _receipt(lines: list[str], text: str) -> dict[str, Any]:
    return {
        "vendor_name": first_line_vendor(lines),
        "vendor_address": receipt_address(lines),
        "receipt_number": _first_id(lines, RECEIPT_NUMBER),
        "receipt_date": _labelled_date(lines, r"date") or _first_date(lines),
        "currency": parse_currency(text),
        "tax": find_tax(lines),
        "total": find_total(lines),
    }


def _purchase_order(lines: list[str], text: str) -> dict[str, Any]:
    return {
        "po_number": _first_id(lines, PO_NUMBER),
        "vendor_name": _party(lines, r"supplier|vendor|to"),
        "vendor_tax_id": _tax_id(lines),
        "buyer_name": _party(lines, r"buyer|from|bill\s+to") or first_line_vendor(lines),
        "order_date": _labelled_date(lines, r"order\s+date|po\s+date|date") or _first_date(lines),
        "delivery_date": _labelled_date(
            lines, r"deliver(?:y)?\s+(?:by|date|on)|ship\s+by|required\s+by"
        ),
        "contract_number": _first_id(lines, CONTRACT_NUMBER),
        "currency": parse_currency(text),
        "subtotal": find_subtotal(lines),
        "tax": find_tax(lines),
        "total": find_total(lines),
    }


_ALLOCATION = re.compile(r"^(?P<inv>[A-Z]{2,}[\w\-/]*\d[\w\-/]*)\s{2,}.*?(?P<amt>\d[\d,.\s]*\d)$")


def _payment(lines: list[str], text: str) -> dict[str, Any]:
    allocations = []
    for line in lines:
        match = _ALLOCATION.match(line)
        if match and find_amounts(match["amt"]):
            allocations.append(
                {"invoice_number": match["inv"], "amount": find_amounts(match["amt"])[-1]}
            )
    amount = find_total(
        lines, re.compile(r"\b(?:total\s+paid|amount\s+paid|payment\s+amount)\b", _I)
    )
    return {
        "payment_reference": _first_id(lines, PAYMENT_REFERENCE),
        "payer_name": _party(lines, r"from|payer|remitter"),
        "vendor_name": _party(lines, r"to|payee|beneficiary|supplier|vendor"),
        "paid_at": _labelled_date(lines, r"payment\s+date|paid\s+on|value\s+date|date")
        or _first_date(lines),
        "method": _payment_method(text),
        "currency": parse_currency(text),
        "amount": amount if amount is not None else find_total(lines),
        "allocations": allocations,
    }


_BETWEEN = re.compile(
    r"\bbetween\s+(?P<a>.+?)\s*(?:\((?:the\s+)?(?P<ra>[^)]+)\))?\s*,?\s+and\s+(?P<b>.+?)\s*"
    r"(?:\((?:the\s+)?(?P<rb>[^)]+)\))?\s*[.,;]?$",
    _I,
)
_AGREEMENT_TITLE = re.compile(r"\b(?:agreement|contract)\b", _I)
_TERM = re.compile(r"\bfrom\s+(?P<start>.+?)\s+(?:to|until|through)\s+(?P<end>.+?)[.;]?$", _I)
_PAYMENT_TERMS = re.compile(r"\b(?:within\s+(?P<a>\d{1,3})\s+days|net\s*(?P<b>\d{1,3}))\b", _I)


def _contract(lines: list[str], text: str) -> dict[str, Any]:
    vendor = customer = None
    for line in lines:
        match = _BETWEEN.search(line)
        if match:
            a, b = match["a"].strip(), match["b"].strip()
            if (match["ra"] or "").lower().startswith("customer") or (
                match["rb"] or ""
            ).lower().startswith(("supplier", "vendor")):
                a, b = b, a
            vendor, customer = a, b
            break
    start = end = None
    for line in lines:
        match = _TERM.search(line)
        if match and find_dates(match["start"]) and find_dates(match["end"]):
            start, end = find_dates(match["start"])[0], find_dates(match["end"])[0]
            break
    terms = _PAYMENT_TERMS.search(text)
    return {
        "contract_number": _first_id(lines, CONTRACT_NUMBER),
        "title": lines[0].title() if lines and _AGREEMENT_TITLE.search(lines[0]) else None,
        "vendor_name": vendor or _party(lines, r"supplier|vendor|contractor"),
        "customer_name": customer or _party(lines, r"customer|client|buyer"),
        "start_date": start
        or _labelled_date(lines, r"effective\s+date|commencement\s+date|start\s+date"),
        "end_date": end or _labelled_date(lines, r"end\s+date|expiry\s+date|termination\s+date"),
        "payment_terms_days": int(terms["a"] or terms["b"]) if terms else None,
        "currency": parse_currency(text),
    }


def _tax_id(lines: Sequence[str]) -> str | None:
    for line in lines:
        match = TAX_ID.search(line)
        if match:
            return re.sub(r"\s", "", match["id"])
    return None


_METHODS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:bank\s+transfer|bacs|ach|wire|swift|sepa|giro)\b", _I), "bank_transfer"),
    (re.compile(r"\bcheque|check\s+no\b", _I), "cheque"),
    (re.compile(r"\b(?:credit|debit)\s+card|visa|mastercard\b", _I), "card"),
    (re.compile(r"\bcash\b", _I), "cash"),
]


def _payment_method(text: str) -> str | None:
    for pattern, method in _METHODS:
        if pattern.search(text):
            return method
    return None


_BY_TYPE: dict[DocType, Callable[[list[str], str], dict[str, Any]]] = {
    DocType.INVOICE: _invoice,
    DocType.RECEIPT: _receipt,
    DocType.PURCHASE_ORDER: _purchase_order,
    DocType.PAYMENT: _payment,
    DocType.CONTRACT: _contract,
}


class RulesExtractor:
    """:class:`~ledgerlens.extraction.base.Extractor` built from regexes and heuristics."""

    name = "rules"

    def extract(self, text: str, doc_type: DocType | str) -> ExtractionResult:
        kind = DocType(doc_type)
        if kind not in _BY_TYPE:
            return ExtractionResult(
                extractor=self.name, doc_type=kind, errors=[f"no rules for {kind.value}"]
            )
        fields = _BY_TYPE[kind](_lines(text), text)
        return ExtractionResult.from_fields(self.name, kind, fields)
