"""The router node: what does the user want done? (docs/layers/04-agents-mcp.md)

Two layers, cheapest first:

1. **Rules** (:func:`rule_intent`): patterns for each intent. A request that clearly asks
   for one action -- a reminder, a 3-way match, reading a document -- or clearly asks a
   question about the ledger, or is clearly small talk or off topic, is routed without a
   model call. Action intents win over ``question`` ("Does INV-889 match its PO?" is a
   match, not a lookup); a status question about an action ("Was a reminder sent?") is a
   question.
2. **The** ``fast`` **model** with structured output (:class:`RouterDecision`, prompt
   ``router.md``), when the rules are ambiguous (two actions) or find nothing. If no model
   answers, the rules' best guess stands: the first action by priority, else ``question``
   when the request mentions the ledger, else ``other``.

Document numbers are always found by the same patterns retrieval uses
(:func:`ledgerlens.retrieval.analyze.find_doc_numbers`), and vendor names by the vendor
master when the router has it, whatever decided the intent.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.messages import AnyMessage, HumanMessage
from pydantic import BaseModel, Field

from ledgerlens.agents.state import Intent, LedgerState
from ledgerlens.core.prompts import load_prompt
from ledgerlens.core.schemas import DocType
from ledgerlens.llm.types import LLMError, Message
from ledgerlens.retrieval.analyze import find_doc_numbers

if TYPE_CHECKING:
    from ledgerlens.llm.router import LLMClient
    from ledgerlens.retrieval.analyze import VendorFinder

logger = logging.getLogger(__name__)

PROMPT = "router"
Method = Literal["rules", "llm", "fallback"]


class RouterDecision(BaseModel):
    """The router's structured output."""

    intent: Intent
    invoice_numbers: list[str] = Field(default_factory=list)
    po_numbers: list[str] = Field(default_factory=list)
    vendor: str | None = None
    reason: str = ""


def _any(*patterns: str) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


REMIND = _any(
    r"\bremind(?:er|ers|ed)?\b",
    r"\bchas(?:e|ing)\b",
    r"\bfollow[- ]?up\b",
    r"\bnudge\b",
    r"\bdunning\b",
    r"\bfinal notice\b",
    r"\b(?:overdue|payment|late[- ]payment) (?:notice|letter)\b",
    r"\b(?:draft|write|prepare|send|compose)\b.{0,40}\b(?:email|letter|message|note)\b",
)
RECONCILE = _any(
    r"\breconcil",
    r"\b(?:3|three)[- ]way\b",
    r"\bmatch(?:es|ed|ing)?\b",
    r"\bmismatch",
    r"\bdiscrepanc",
    r"\bover[- ]?(?:bill|charg)",
    r"\bconsistent with\b",
    r"\bagree(?:s)? with\b",
    r"\b(?:billed|charged|invoiced|priced)\b.{0,40}\b(?:contract|agreed) (?:rate|price)",
    r"\b(?:check|verify|validate|compare|audit)\b.{0,60}\b(?:against|with|versus|vs)\b",
    r"\b(?:right|correct|wrong)\b.{0,30}\b(?:po|purchase order|contract)\b",
)
EXTRACT = _any(
    r"\bextract",
    r"\bupload(?:ed|s)?\b",
    r"\bparse\b",
    r"\bingest",
    r"\bocr\b",
    r"\battach(?:ed|ment)\b",
    r"\bcapture\b",
    r"\bkey (?:it |this |them )?in\b",
    r"\b(?:here'?s|here is) (?:a|an|the)? ?new\b",
    r"\b(?:read|process|scan|digiti[sz]e)\b.{0,30}\b(?:pdf|scan|file|document|attachment|"
    r"receipt|image|photo|payment advice|remittance)\b",
    r"\b(?:process|read)\b (?:this|these|the new)\b",
)
QUESTION = _any(
    r"\?\s*$",
    r"^\W*(?:what|what's|which|who|whom|whose|when|where|why|how|is|are|was|were|do|does|"
    r"did|has|have|had|show|list|give|tell|find|summari[sz]e|total|count)\b",
)
"""Interrogative form: a question mark, or a question / lookup word first. A question is
only a ledger question if it also names something in the ledger (:data:`LEDGER`)."""
STATUS_OF_ACTION = re.compile(
    r"^\s*(?:was|were|has|have|had|did|is|are)\b.{0,60}\b(?:drafted|sent|reconciled|"
    r"matched|chased|reminded|extracted|processed|uploaded)\b",
    re.IGNORECASE,
)
SMALL_TALK = _any(
    r"^\W*(?:hi|hello|hey|hiya|thanks|thank you|cheers|good (?:morning|afternoon|evening)|"
    r"bye|goodbye|ok(?:ay)?|cool|great|nice)\b",
    r"\bwhat can you (?:do|help)",
    r"\b(?:who|what) are you\b",
)
OFF_TOPIC = _any(
    r"\b(?:weather|joke|poem|recipe|football|soccer|movie|film|song|music|translate|"
    r"bitcoin|crypto|stock price|python|javascript|code)\b",
)
LEDGER = _any(
    r"\binvoices?\b",
    r"\binv-?\d",
    r"\bpo\b",
    r"\bpo-?\d",
    r"\bpurchase orders?\b",
    r"\bcontracts?\b",
    r"\bpayments?\b",
    r"\b(?:paid|pay|owe|owed|spend|spent|bill|billed)\b",
    r"\b(?:vendors?|suppliers?|receipts?|remittance|ledger|accounts payable)\b",
    r"\b(?:overdue|outstanding|balance|aging|ageing|tax|total|amount)\b",
    r"[$£€]\s?\d",
)
ACTIONS: tuple[tuple[Intent, re.Pattern[str]], ...] = (
    ("remind", REMIND),
    ("reconcile", RECONCILE),
    ("extract", EXTRACT),
)
"""In priority order, for the fallback when two actions match and no model decides."""


def matched_actions(text: str, *, attachments: bool = False) -> list[Intent]:
    found = [intent for intent, pattern in ACTIONS if pattern.search(text)]
    if attachments and "extract" not in found and not found:
        found.append("extract")  # an upload with no other ask is there to be read
    return found


def rule_intent(text: str, *, attachments: bool = False) -> Intent | None:
    """The intent the rules are sure of, or ``None`` (ambiguous, or nothing matched)."""
    ledger = bool(LEDGER.search(text))
    if not ledger and not attachments and (OFF_TOPIC.search(text) or SMALL_TALK.search(text)):
        return "other"
    if STATUS_OF_ACTION.search(text):
        return "question"
    actions = matched_actions(text, attachments=attachments)
    if len(actions) == 1:
        return actions[0]
    if len(actions) > 1:
        return None
    if QUESTION.search(text) and ledger:
        return "question"
    return None


def fallback_intent(text: str, *, attachments: bool = False) -> Intent:
    """The rules' best guess when no model answers."""
    actions = matched_actions(text, attachments=attachments)
    if actions:
        return actions[0]
    return "question" if LEDGER.search(text) else "other"


def find_entities(text: str, vendors: VendorFinder | None = None) -> dict[str, Any]:
    """Document numbers by type, and the vendor master ids the text names."""
    entities: dict[str, Any] = {}
    keys = {
        DocType.INVOICE: "invoice_numbers",
        DocType.PURCHASE_ORDER: "po_numbers",
        DocType.CONTRACT: "contract_numbers",
        DocType.PAYMENT: "payment_references",
    }
    for number, doc_type in find_doc_numbers(text):
        key = keys.get(doc_type, "document_numbers")
        entities.setdefault(key, [])
        if number not in entities[key]:
            entities[key].append(number)
    if vendors is not None and (found := vendors.find(text)):
        entities["vendor_ids"] = found
        entities["vendor"] = vendors.names.get(found[0])
    return entities


def latest_user_text(messages: Sequence[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return (
                message.text() if callable(getattr(message, "text", None)) else str(message.content)
            )
    return ""


class Router:
    """The router node: ``state`` in, ``intent`` / ``entities`` / ``document_ids`` out."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        *,
        model: str = "fast",
        vendors: VendorFinder | None = None,
    ) -> None:
        self.llm = llm
        self.model = model
        self.vendors = vendors
        self.prompt = load_prompt(PROMPT)

    def decide(
        self, text: str, *, attachments: Sequence[str] = ()
    ) -> tuple[RouterDecision, Method]:
        has_files = bool(attachments)
        intent = rule_intent(text, attachments=has_files)
        if intent is not None:
            return RouterDecision(intent=intent, reason="rules"), "rules"
        if self.llm is not None:
            listed = ", ".join(attachments) if attachments else "none"
            prompt = self.prompt.render(message=text, attachments=listed)
            try:
                decision = self.llm.structured(
                    [Message.user(prompt)], RouterDecision, model=self.model
                )
                return decision, "llm"
            except LLMError as exc:
                logger.warning("router model unavailable (%s); using the rules' best guess", exc)
        guess = fallback_intent(text, attachments=has_files)
        return RouterDecision(intent=guess, reason="rules fallback"), "fallback"

    def __call__(self, state: LedgerState) -> dict[str, Any]:
        text = latest_user_text(state.get("messages", []))
        attachments = list(state.get("document_ids") or [])
        decision, method = self.decide(text, attachments=attachments)
        entities = find_entities(text, self.vendors)
        for key, values in (
            ("invoice_numbers", decision.invoice_numbers),
            ("po_numbers", decision.po_numbers),
        ):
            merged = list(dict.fromkeys([*entities.get(key, []), *values]))
            if merged:
                entities[key] = merged
        if decision.vendor and "vendor" not in entities:
            entities["vendor"] = decision.vendor
        entities["routed_by"] = method
        logger.info("routed to %s by %s", decision.intent, method)
        return {
            "intent": decision.intent,
            "entities": entities,
            "document_ids": attachments,
            "step_count": 1,
        }
