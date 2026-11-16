"""Query analysis: pull entities, filters and intent out of a question.

:class:`QueryAnalyzer` reads a question the way the retrieval layer needs it
(docs/layers/03-retrieval.md, "Analyze"):

* **document numbers** -- ``INV-889``, ``PO-1042``, ``PAY-2025-000227``, ``CT-V001``, ...
  (upper-cased; the prefix also says which document type is meant);
* **vendors** -- names matched against the vendor master: exact canonical name or alias
  ("ACME Supplies, Ltd." = "Acme Supplies Limited"), a distinctive first word ("Acme"),
  or a fuzzy match on a word n-gram (``rapidfuzz``, for typos).

Everything above is rules: fast, free and predictable.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict
from rapidfuzz import fuzz
from sqlalchemy.orm import Session, sessionmaker

from ledgerlens.core.schemas import DocType
from ledgerlens.extraction.normalize import canonical_vendor
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


# --------------------------------------------------------------------------- the analyzer


class QueryAnalyzer:
    """Rules over the question text; nothing is sent anywhere."""

    def __init__(
        self,
        vendors: Iterable[VendorCandidate] = (),
        *,
        vendor_threshold: float = 88.0,
    ) -> None:
        self.finder = VendorFinder(vendors, threshold=vendor_threshold)

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

    def analyze(self, question: str) -> QueryAnalysis:
        return self.rules(question)

    def rules(self, question: str) -> QueryAnalysis:
        numbers = find_doc_numbers(question)
        doc_numbers = [number for number, _ in numbers]
        rest = _DOC_NUMBER.sub(" ", question)  # a number is not a vendor
        vendor_ids = self.finder.find(rest)
        all_types = list(dict.fromkeys(t for _, t in numbers))
        return QueryAnalysis(
            query=question,
            doc_numbers=tuple(doc_numbers),
            doc_types=tuple(all_types),
            vendor_ids=tuple(vendor_ids),
            vendor_names=tuple(self.finder.names[v] for v in vendor_ids),
        )
