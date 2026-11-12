"""Decide what kind of document a page of text is: invoice, receipt, PO, contract or payment.

Keyword rules first -- they are free, instant and right on almost every real document,
because financial documents announce themselves ("TAX INVOICE", "PURCHASE ORDER",
"REMITTANCE ADVICE"). Each rule has a weight; a match in the document's opening lines (its
title) counts double. The rules decide when the best type scores at least
:data:`MIN_SCORE` and clearly beats the runner-up (:data:`MIN_MARGIN`); otherwise the
text goes to an LLM through the ``fast`` alias. With no LLM, or if it fails, the result
is the rules' best guess, or ``unknown``.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ledgerlens.core.schemas import DocType
from ledgerlens.llm.router import LLMClient
from ledgerlens.llm.types import LLMError, Message

logger = logging.getLogger(__name__)

MIN_SCORE = 4.0
MIN_MARGIN = 1.5  # best score must be at least this many times the runner-up's
TITLE_LINES = 3
TITLE_WEIGHT = 2.0
LLM_ALIAS = "fast"
LLM_MAX_CHARS = 4000

_Rules = list[tuple[re.Pattern[str], float]]


def _rules(*pairs: tuple[str, float]) -> _Rules:
    return [(re.compile(pattern, re.IGNORECASE), weight) for pattern, weight in pairs]


RULES: dict[DocType, _Rules] = {
    DocType.INVOICE: _rules(
        (r"\b(tax\s+)?invoice\b", 3),
        (r"\binvoice\s*(no\.?|number|#|date)", 3),
        (r"\bbill(ed)?\s+to\b", 1),
        (r"\b(due\s+date|payment\s+due|amount\s+due)\b", 2),
        (r"\b(vat|gst|tax)\s+(reg|id|no)", 1),
    ),
    DocType.RECEIPT: _rules(
        (r"\breceipt\b", 3),
        (r"\b(cash|change|tendered|card\s+payment)\b", 2),
        (r"\b(thank\s+you|terima\s+kasih)\b", 2),
        (r"\b(cashier|counter|table\s*no)\b", 1),
    ),
    DocType.PURCHASE_ORDER: _rules(
        (r"\bpurchase\s+order\b", 4),
        (r"\b(po|p\.o\.)\s*(no\.?|number|#)", 1),
        (r"\b(deliver(y)?\s+(by|to|date)|ship\s+to)\b", 1),
        (r"\b(buyer|supplier)\b", 1),
    ),
    DocType.CONTRACT: _rules(
        (r"\b(agreement|contract)\b", 3),
        (r"\b(term|termination|governing\s+law|liability|indemnif)", 1),
        (r"\bbetween\b.+\band\b", 2),
        (r"^\s*(\d+\.|section\s+\d+|article\s+[ivx\d]+)\s+[A-Z]", 1),
        (r"\b(schedule|exhibit|annex)\s+\w+", 1),
    ),
    DocType.PAYMENT: _rules(
        (r"\bremittance(\s+advice)?\b", 5),
        (r"\bpayment\s+(reference|advice|date|confirmation)\b", 3),
        (r"\bamount\s+paid\b|\btotal\s+paid\b", 3),
        (r"\b(bacs|ach|swift|wire\s+transfer)\b", 1),
    ),
}


class Classification(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_type: DocType
    method: Literal["rules", "llm", "fallback"]
    scores: dict[DocType, float] = Field(default_factory=dict)


class DocTypeAnswer(BaseModel):
    """What the LLM is asked to return."""

    model_config = ConfigDict(extra="forbid")

    doc_type: Literal["invoice", "receipt", "purchase_order", "payment", "contract", "unknown"]


def score_text(text: str) -> dict[DocType, float]:
    """Weighted keyword score per type; matches in the first lines count double."""
    lines = [line for line in text.splitlines() if line.strip()]
    title = "\n".join(lines[:TITLE_LINES])
    body = "\n".join(lines[TITLE_LINES:])
    scores: dict[DocType, float] = {}
    for doc_type, rules in RULES.items():
        total = 0.0
        for pattern, weight in rules:
            if pattern.search(title):
                total += weight * TITLE_WEIGHT
            elif any(pattern.search(line) for line in body.splitlines()):
                total += weight
        scores[doc_type] = total
    return scores


def classify_by_rules(text: str) -> Classification | None:
    """The rules' answer, or ``None`` when they are not confident enough."""
    scores = score_text(text)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    (best, top), (_, second) = ranked[0], ranked[1]
    if top >= MIN_SCORE and top >= MIN_MARGIN * second:
        return Classification(doc_type=best, method="rules", scores=scores)
    return None


def classify(text: str, *, llm: LLMClient | None = None, model: str = LLM_ALIAS) -> Classification:
    """Classify ``text`` (a whole document's text): rules, then LLM, then best guess."""
    decided = classify_by_rules(text)
    if decided is not None:
        return decided
    scores = score_text(text)
    if llm is not None:
        try:
            answer = llm.structured(_prompt(text), DocTypeAnswer, model=model)
            return Classification(doc_type=DocType(answer.doc_type), method="llm", scores=scores)
        except LLMError as exc:
            logger.warning("LLM doc-type classification failed, using rules guess: %s", exc)
    best, top = max(scores.items(), key=lambda item: item[1])
    guess = best if top >= MIN_SCORE else DocType.UNKNOWN
    return Classification(doc_type=guess, method="fallback", scores=scores)


def _prompt(text: str) -> list[Message]:
    return [
        Message.system(
            "You classify business documents. Answer with the single best doc_type: "
            "invoice (a bill requesting payment), receipt (proof of a completed retail "
            "purchase), purchase_order (a buyer's order to a supplier), payment (remittance "
            "advice or payment confirmation), contract (an agreement between parties), or "
            "unknown."
        ),
        Message.user(f"Document text:\n\n{text[:LLM_MAX_CHARS]}"),
    ]
