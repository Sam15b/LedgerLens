"""Build the knowledge graph: Vendor -> Invoice -> PurchaseOrder <- Contract, Payment -> Invoice.

Vendor resolution (:class:`VendorResolver`) decides which ``Vendor`` node a document's
printed vendor is. In order:

1. **tax id** -- the same VAT / tax registration number (spaces, dashes and case ignored);
2. **name** -- the same :func:`~ledgerlens.extraction.normalize.canonical_vendor` key as the
   vendor's name or one of its aliases ("ACME Supplies, Ltd." = "Acme Supplies Limited");
3. **fuzzy** -- the closest name or alias by ``rapidfuzz`` token-sort ratio, if it scores at
   least ``threshold`` (OCR slips, word order).

A fuzzy or alias match remembers the printed spelling as a new alias, so the next document
spelled that way matches exactly. An unknown vendor becomes a new ``Vendor`` whose id is
derived from its canonical name, so linking the same document twice finds it again.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from neo4j import Driver
from rapidfuzz import fuzz, process

from ledgerlens.extraction.normalize import canonical_vendor
from ledgerlens.storage.neo4j.driver import run_read, run_write

DEFAULT_MATCH_THRESHOLD = 88.0  # same cut-off as extraction.normalize.VendorMatcher
MatchMethod = Literal["tax_id", "name", "fuzzy", "new"]

_TAX_ID_NOISE = re.compile(r"[\s\-./]")
_SLUG = re.compile(r"[^a-z0-9]+")


def normalize_tax_id(tax_id: str | None) -> str | None:
    """``"gb 123-456 789"`` -> ``"GB123456789"``; blank -> ``None``."""
    if tax_id is None:
        return None
    cleaned = _TAX_ID_NOISE.sub("", tax_id).upper()
    return cleaned or None


def new_vendor_id(name: str) -> str:
    """A stable id for a vendor first seen on a document: ``v_`` + its canonical name.

    The same printed name (or any spelling with the same canonical key) always gives the
    same id, which is what makes re-linking idempotent.
    """
    key = canonical_vendor(name) or name.strip().lower()
    slug = _SLUG.sub("_", key).strip("_")[:40]
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    return f"v_{slug}_{digest}" if slug else f"v_{digest}"


@dataclass(frozen=True)
class VendorCandidate:
    """A ``Vendor`` node as the resolver sees it."""

    id: str
    name: str
    tax_id: str | None = None
    aliases: tuple[str, ...] = ()

    def names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


@dataclass(frozen=True)
class VendorMatch:
    vendor_id: str
    method: MatchMethod
    score: float
    """100 for tax id and exact name matches, the rapidfuzz ratio for fuzzy ones."""


def resolve_vendor(
    name: str,
    tax_id: str | None,
    candidates: Iterable[VendorCandidate],
    *,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> VendorMatch | None:
    """The known vendor a printed name / tax id refers to, or ``None``."""
    pool = list(candidates)
    wanted_tax = normalize_tax_id(tax_id)
    if wanted_tax:
        for vendor in pool:
            if normalize_tax_id(vendor.tax_id) == wanted_tax:
                return VendorMatch(vendor.id, "tax_id", 100.0)
    key = canonical_vendor(name)
    if not key:
        return None
    keys: dict[str, str] = {}  # canonical name/alias -> vendor id (first vendor wins)
    for vendor in pool:
        for known in vendor.names():
            known_key = canonical_vendor(known)
            if known_key:
                keys.setdefault(known_key, vendor.id)
    if key in keys:
        return VendorMatch(keys[key], "name", 100.0)
    best = process.extractOne(key, list(keys), scorer=fuzz.token_sort_ratio, score_cutoff=threshold)
    if best is None:
        return None
    return VendorMatch(keys[best[0]], "fuzzy", float(best[1]))


class VendorResolver:
    """Resolves printed vendors against the ``Vendor`` nodes in Neo4j, creating new ones."""

    def __init__(
        self,
        driver: Driver,
        *,
        threshold: float = DEFAULT_MATCH_THRESHOLD,
        database: str | None = None,
    ) -> None:
        self.driver = driver
        self.threshold = threshold
        self.database = database

    def candidates(self) -> list[VendorCandidate]:
        rows = run_read(
            self.driver,
            "MATCH (v:Vendor) RETURN v.id AS id, v.name AS name, v.tax_id AS tax_id, "
            "coalesce(v.aliases, []) AS aliases ORDER BY v.id",
            database=self.database,
        )
        return [
            VendorCandidate(
                id=row["id"],
                name=row["name"] or row["id"],
                tax_id=row["tax_id"],
                aliases=tuple(row["aliases"]),
            )
            for row in rows
        ]

    def upsert(
        self, vendor_id: str, name: str, *, tax_id: str | None = None, aliases: Sequence[str] = ()
    ) -> None:
        """Create or update a vendor from master data (e.g. the Postgres ``vendors`` table).

        Aliases already on the node are kept; ``aliases`` are added to them.
        """
        run_write(
            self.driver,
            """
            MERGE (v:Vendor {id: $id})
            SET v.name = $name,
                v.tax_id = coalesce($tax_id, v.tax_id),
                v.aliases = reduce(
                    acc = coalesce(v.aliases, []), a IN $aliases |
                    CASE WHEN a IN acc OR a = $name THEN acc ELSE acc + a END
                )
            """,
            {"id": vendor_id, "name": name, "tax_id": tax_id, "aliases": list(aliases)},
            database=self.database,
        )

    def resolve(self, name: str, tax_id: str | None = None) -> VendorMatch:
        """Match, or the id a new vendor with this name would get (``method="new"``)."""
        match = resolve_vendor(name, tax_id, self.candidates(), threshold=self.threshold)
        return match or VendorMatch(new_vendor_id(name), "new", 0.0)

    def ensure(self, name: str, tax_id: str | None = None) -> VendorMatch:
        """Resolve the vendor and make sure its node exists.

        A new vendor is created with the printed name; a name or fuzzy match whose printed
        spelling differs from the vendor's name records it as an alias; a tax id is filled
        in if the vendor had none.
        """
        match = self.resolve(name, tax_id)
        if match.method == "new":
            run_write(
                self.driver,
                "MERGE (v:Vendor {id: $id}) "
                "ON CREATE SET v.name = $name, v.aliases = [], v.tax_id = $tax_id",
                {"id": match.vendor_id, "name": name, "tax_id": tax_id},
                database=self.database,
            )
            return match
        run_write(
            self.driver,
            """
            MATCH (v:Vendor {id: $id})
            SET v.tax_id = coalesce(v.tax_id, $tax_id),
                v.aliases = CASE
                    WHEN $name = v.name OR $name IN coalesce(v.aliases, [])
                    THEN coalesce(v.aliases, [])
                    ELSE coalesce(v.aliases, []) + $name
                END
            """,
            {"id": match.vendor_id, "name": name, "tax_id": tax_id},
            database=self.database,
        )
        return match
