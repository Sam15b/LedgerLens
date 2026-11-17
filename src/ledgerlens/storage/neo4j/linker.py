"""Build the knowledge graph: Vendor -> Invoice -> PurchaseOrder <- Contract, Payment -> Invoice.

:class:`GraphLinker` MERGEs one extracted record at a time (docs/layers/02-storage.md)::

    (Vendor)-[:ISSUED]->(Invoice)-[:REFERENCES]->(PurchaseOrder)<-[:GOVERNS]-(Contract)
    (Payment)-[:SETTLES {invoice_number, amount}]->(Invoice)      (Vendor)-[:PARTY_TO]->(Contract)
    (Invoice|PurchaseOrder)-[:HAS_LINE]->(LineItem)

Node ids are the record ids used in Postgres, and nodes from a document carry its
``document_id``, so graph results map back to rows and Pinecone chunks. A ``Contract``
keeps its rate card as three parallel lists, ``rate_items`` / ``rate_units`` /
``rate_prices`` (Neo4j properties cannot hold maps; a missing unit is ``""``).

A reference that finds no target -- an invoice's PO number with no PO, a remittance line
naming an unknown invoice, a PO citing a contract that is not in the graph -- is recorded
as an ``(:UnresolvedLink)`` node for the reconciler, and removed again once the target is
linked. :meth:`GraphLinker.unresolved` lists them.

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
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, Literal

from neo4j import Driver, ManagedTransaction
from rapidfuzz import fuzz, process

from ledgerlens.core.schemas import (
    Contract,
    ExtractedRecord,
    Invoice,
    LineItem,
    Payment,
    PurchaseOrder,
    Receipt,
)
from ledgerlens.extraction.normalize import canonical_vendor
from ledgerlens.storage.neo4j.driver import run_read, run_write, write_session

DEFAULT_MATCH_THRESHOLD = 88.0  # same cut-off as extraction.normalize.VendorMatcher
MatchMethod = Literal["given", "tax_id", "name", "fuzzy", "new"]
"""``given``: the caller supplied the vendor id (master data), nothing was resolved."""

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
    """100 for given ids, tax id and exact name matches; the rapidfuzz ratio for fuzzy ones."""


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
        """Resolve the vendor and make sure its node exists (:meth:`resolve` +
        :meth:`remember`)."""
        match = self.resolve(name, tax_id)
        self.remember(match, name, tax_id)
        return match

    def remember(self, match: VendorMatch, name: str, tax_id: str | None = None) -> None:
        """Write what a resolution learned: a new vendor is created with the printed name;
        a match whose printed spelling differs from the vendor's name records it as an
        alias; a tax id is filled in if the vendor had none."""
        if match.method == "new":
            run_write(
                self.driver,
                "MERGE (v:Vendor {id: $id}) "
                "ON CREATE SET v.name = $name, v.aliases = [], v.tax_id = $tax_id",
                {"id": match.vendor_id, "name": name, "tax_id": tax_id},
                database=self.database,
            )
            return
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


# --------------------------------------------------------------------------- linking
#
# Every query below is safe to repeat: nodes are MERGEd on their id, and the edges a
# record owns are deleted and MERGEd again from its current values, so a re-link after a
# correction (a new PO number, a removed allocation) leaves no stale edge behind.
# A reference whose target is not in the graph yet is kept as a number on the node
# (Invoice.po_number, Payment.invoice_numbers, PurchaseOrder.contract_number); when the
# target is linked later, it picks those up -- linking order does not matter.

_VENDOR = """
MERGE (v:Vendor {id: $vendor_id})
  ON CREATE SET v.name = $vendor_name, v.aliases = []
"""

_INVOICE = (
    _VENDOR
    + """
MERGE (i:Invoice {id: $id})
SET i += $props, i.invoice_date = date($invoice_date), i.due_date = date($due_date)
WITH v, i
OPTIONAL MATCH (:Vendor)-[old:ISSUED]->(i)
DELETE old
WITH DISTINCT v, i
MERGE (v)-[:ISSUED]->(i)
WITH i
OPTIONAL MATCH (i)-[ref:REFERENCES]->(:PurchaseOrder)
DELETE ref
WITH DISTINCT i
CALL (i) {
  OPTIONAL MATCH (p:PurchaseOrder {po_number: i.po_number})
  WITH i, p ORDER BY CASE WHEN p.vendor_id = i.vendor_id THEN 0 ELSE 1 END, p.id LIMIT 1
  FOREACH (_ IN CASE WHEN p IS NULL THEN [] ELSE [1] END | MERGE (i)-[:REFERENCES]->(p))
  RETURN p.id AS po_id
}
CALL (i) {
  OPTIONAL MATCH (pay:Payment)
  WHERE i.invoice_number IN pay.invoice_numbers
    AND (pay.vendor_id IS NULL OR pay.vendor_id = i.vendor_id)
    AND NOT EXISTS { (pay)-[:SETTLES {invoice_number: i.invoice_number}]->(:Invoice) }
  WITH i, pay,
       [k IN range(0, size(coalesce(pay.invoice_numbers, [])) - 1)
        WHERE pay.invoice_numbers[k] = i.invoice_number | pay.allocated_amounts[k]][0] AS amount
  FOREACH (_ IN CASE WHEN pay IS NULL THEN [] ELSE [1] END |
    MERGE (pay)-[s:SETTLES {invoice_number: i.invoice_number}]->(i) SET s.amount = amount)
  RETURN collect(pay.id) AS settled_by
}
RETURN po_id, settled_by
"""
)

_LINES = """
MATCH (parent:{label} {{id: $id}})
OPTIONAL MATCH (parent)-[:HAS_LINE]->(stale:LineItem) WHERE stale.position >= size($lines)
DETACH DELETE stale
WITH DISTINCT parent
UNWIND $lines AS line
MERGE (l:LineItem {{id: line.id}})
SET l += line.props
MERGE (parent)-[:HAS_LINE]->(l)
"""

_PURCHASE_ORDER = (
    _VENDOR
    + """
MERGE (p:PurchaseOrder {id: $id})
SET p += $props, p.order_date = date($order_date), p.delivery_date = date($delivery_date)
WITH p
OPTIONAL MATCH (:Contract)-[old:GOVERNS]->(p)
DELETE old
WITH DISTINCT p
CALL (p) {
  OPTIONAL MATCH (c:Contract {contract_number: p.contract_number, vendor_id: p.vendor_id})
  WITH p, c ORDER BY c.id LIMIT 1
  FOREACH (_ IN CASE WHEN c IS NULL THEN [] ELSE [1] END | MERGE (c)-[:GOVERNS]->(p))
  RETURN c.id AS contract_id
}
CALL (p) {
  OPTIONAL MATCH (i:Invoice {po_number: p.po_number})
  WHERE NOT EXISTS { (i)-[:REFERENCES]->(:PurchaseOrder) }
  FOREACH (_ IN CASE WHEN i IS NULL THEN [] ELSE [1] END | MERGE (i)-[:REFERENCES]->(p))
  RETURN collect(i.id) AS referenced_by
}
RETURN contract_id, referenced_by
"""
)

_PAYMENT = (
    _VENDOR
    + """
MERGE (pay:Payment {id: $id})
SET pay += $props, pay.paid_at = date($paid_at)
WITH pay
OPTIONAL MATCH (pay)-[old:SETTLES]->(:Invoice)
DELETE old
WITH DISTINCT pay
UNWIND range(0, size(pay.invoice_numbers) - 1) AS k
CALL (pay, k) {
  OPTIONAL MATCH (i:Invoice {invoice_number: pay.invoice_numbers[k]})
  WHERE pay.vendor_id IS NULL OR i.vendor_id = pay.vendor_id
  WITH pay, k, i ORDER BY i.id LIMIT 1
  FOREACH (_ IN CASE WHEN i IS NULL THEN [] ELSE [1] END |
    MERGE (pay)-[s:SETTLES {invoice_number: pay.invoice_numbers[k]}]->(i)
    SET s.amount = pay.allocated_amounts[k])
  RETURN i.id AS invoice_id
}
RETURN pay.invoice_numbers[k] AS invoice_number, invoice_id
"""
)

_CONTRACT = (
    _VENDOR
    + """
MERGE (c:Contract {id: $id})
SET c += $props, c.start_date = date($start_date), c.end_date = date($end_date)
WITH v, c
OPTIONAL MATCH (:Vendor)-[old:PARTY_TO]->(c)
DELETE old
WITH DISTINCT v, c
MERGE (v)-[:PARTY_TO]->(c)
WITH c
OPTIONAL MATCH (c)-[gov:GOVERNS]->(:PurchaseOrder)
DELETE gov
WITH DISTINCT c
CALL (c) {
  OPTIONAL MATCH (p:PurchaseOrder {contract_number: c.contract_number})
  WHERE p.vendor_id = c.vendor_id
  FOREACH (_ IN CASE WHEN p IS NULL THEN [] ELSE [1] END | MERGE (c)-[:GOVERNS]->(p))
  RETURN collect(p.id) AS governs
}
RETURN governs
"""
)


# An UnresolvedLink is deleted once the edge it was waiting for exists.
_CLEAR_RESOLVED = {
    "REFERENCES": """
        MATCH (u:UnresolvedLink {rel: 'REFERENCES', value: $value})
        WHERE EXISTS { (:Invoice {id: u.source_id})-[:REFERENCES]->(:PurchaseOrder) }
        DELETE u
    """,
    "SETTLES": """
        MATCH (u:UnresolvedLink {rel: 'SETTLES', value: $value})
        WHERE EXISTS {
          (:Payment {id: u.source_id})-[:SETTLES {invoice_number: $value}]->(:Invoice)
        }
        DELETE u
    """,
    "GOVERNS": """
        MATCH (u:UnresolvedLink {rel: 'GOVERNS', value: $value})
        WHERE EXISTS { (:Contract)-[:GOVERNS]->(:PurchaseOrder {id: u.source_id}) }
        DELETE u
    """,
}

_RECORD_UNRESOLVED = """
OPTIONAL MATCH (stale:UnresolvedLink {source_id: $source_id})
WHERE NOT stale.id IN [link IN $links | link.id]
DELETE stale
WITH count(*) AS _
UNWIND $links AS link
MERGE (u:UnresolvedLink {id: link.id})
  ON CREATE SET u.recorded_at = datetime()
SET u += link
"""


@dataclass(frozen=True)
class UnresolvedLink:
    """A reference on ``source`` whose target (``target_label`` with ``key = value``) is
    not in the graph. ``rel`` is the edge that will be made once it is."""

    source_label: str
    source_id: str
    rel: Literal["REFERENCES", "SETTLES", "GOVERNS"]
    target_label: str
    key: str
    value: str
    vendor_id: str | None = None

    @property
    def id(self) -> str:
        return f"{self.source_id}|{self.rel}|{self.value}"

    def to_props(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_label": self.source_label,
            "source_id": self.source_id,
            "rel": self.rel,
            "target_label": self.target_label,
            "key": self.key,
            "value": self.value,
            "vendor_id": self.vendor_id,
        }


def _iso(day: date | None) -> str | None:
    return None if day is None else day.isoformat()


def _money(amount: Decimal | None) -> float | None:
    """Neo4j has no decimal type; Postgres keeps the exact amounts."""
    return None if amount is None else float(amount)


def _line_params(record_id: str, lines: Sequence[LineItem]) -> dict[str, Any]:
    return {"id": record_id, "lines": _lines(record_id, lines)}


def _lines(record_id: str, lines: Sequence[LineItem]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"{record_id}#l{n}",
            "props": {
                "position": n,
                "description": line.description,
                "quantity": _money(line.quantity),
                "unit_price": _money(line.unit_price),
                "amount": _money(line.amount),
            },
        }
        for n, line in enumerate(lines)
    ]


RECORD_LABELS = ("Invoice", "PurchaseOrder", "Payment", "Contract")


def record_label(record: ExtractedRecord) -> str:
    """The node label a record is linked as (receipts are invoices)."""
    if isinstance(record, Invoice | Receipt):
        return "Invoice"
    if isinstance(record, PurchaseOrder):
        return "PurchaseOrder"
    if isinstance(record, Payment):
        return "Payment"
    return "Contract"


@dataclass
class LinkResult:
    """What linking one record did."""

    id: str
    label: str
    vendor: VendorMatch
    links: dict[str, list[str]] = field(default_factory=dict)
    """Edges made, by type: ``{"REFERENCES": ["po-1"], "SETTLES": [...]}``."""
    unresolved: list[UnresolvedLink] = field(default_factory=list)
    """References this record makes that found no target (also stored in the graph)."""


class GraphLinker:
    """Writes extracted records into the graph. Every ``link`` is idempotent."""

    def __init__(
        self,
        driver: Driver,
        *,
        resolver: VendorResolver | None = None,
        database: str | None = None,
    ) -> None:
        self.driver = driver
        self.database = database
        self.resolver = resolver or VendorResolver(driver, database=database)

    def link(
        self,
        record: ExtractedRecord,
        *,
        id: str,
        document_id: str | None = None,
        vendor_id: str | None = None,
    ) -> LinkResult:
        """Link any extracted record. ``vendor_id`` skips vendor resolution (master data)."""
        if isinstance(record, Receipt):
            record = record.as_invoice(id)
            doc_type = "receipt"
        else:
            doc_type = record.doc_type
        tax_id = record.vendor_tax_id if isinstance(record, Invoice | PurchaseOrder) else None
        vendor = (
            VendorMatch(vendor_id, "given", 100.0)
            if vendor_id is not None
            else self.resolver.ensure(record.vendor_name, tax_id)
        )
        base = {
            "id": id,
            "vendor_id": vendor.vendor_id,
            "vendor_name": record.vendor_name,
        }
        common = {"document_id": document_id, "vendor_id": vendor.vendor_id, "doc_type": doc_type}
        if isinstance(record, Invoice):
            return self._invoice(record, base, common, vendor)
        if isinstance(record, PurchaseOrder):
            return self._purchase_order(record, base, common, vendor)
        if isinstance(record, Payment):
            return self._payment(record, base, common, vendor)
        return self._contract(record, base, common, vendor)

    # ------------------------------------------------------------------ per type

    def _invoice(
        self, record: Invoice, base: dict[str, Any], common: dict[str, Any], vendor: VendorMatch
    ) -> LinkResult:
        props = common | {
            "invoice_number": record.invoice_number,
            "vendor_name": record.vendor_name,
            "po_number": record.po_number,
            "currency": record.currency,
            "subtotal": _money(record.subtotal),
            "tax": _money(record.tax),
            "total": _money(record.total),
        }
        params = base | {
            "props": props,
            "invoice_date": _iso(record.invoice_date),
            "due_date": _iso(record.due_date),
        }

        def missing(rows: list[dict[str, Any]]) -> list[UnresolvedLink]:
            if record.po_number is None or rows[0]["po_id"]:
                return []
            return [
                UnresolvedLink(
                    "Invoice",
                    base["id"],
                    "REFERENCES",
                    "PurchaseOrder",
                    "po_number",
                    record.po_number,
                    vendor.vendor_id,
                )
            ]

        rows, unresolved = self._write(
            (_INVOICE, params),
            (_LINES.format(label="Invoice"), _line_params(base["id"], record.line_items)),
            source_id=base["id"],
            missing=missing,
            resolves=[("SETTLES", record.invoice_number)],
        )
        links = {"ISSUED": [vendor.vendor_id], "HAS_LINE": _line_ids(base["id"], record.line_items)}
        if rows[0]["po_id"]:
            links["REFERENCES"] = [rows[0]["po_id"]]
        if rows[0]["settled_by"]:
            links["SETTLED_BY"] = sorted(rows[0]["settled_by"])
        return LinkResult(base["id"], "Invoice", vendor, links, unresolved)

    def _purchase_order(
        self,
        record: PurchaseOrder,
        base: dict[str, Any],
        common: dict[str, Any],
        vendor: VendorMatch,
    ) -> LinkResult:
        props = common | {
            "po_number": record.po_number,
            "vendor_name": record.vendor_name,
            "buyer_name": record.buyer_name,
            "contract_number": record.contract_number,
            "currency": record.currency,
            "subtotal": _money(record.subtotal),
            "tax": _money(record.tax),
            "total": _money(record.total),
        }
        params = base | {
            "props": props,
            "order_date": _iso(record.order_date),
            "delivery_date": _iso(record.delivery_date),
        }

        def missing(rows: list[dict[str, Any]]) -> list[UnresolvedLink]:
            if record.contract_number is None or rows[0]["contract_id"]:
                return []
            return [
                UnresolvedLink(
                    "PurchaseOrder",
                    base["id"],
                    "GOVERNS",
                    "Contract",
                    "contract_number",
                    record.contract_number,
                    vendor.vendor_id,
                )
            ]

        rows, unresolved = self._write(
            (_PURCHASE_ORDER, params),
            (_LINES.format(label="PurchaseOrder"), _line_params(base["id"], record.line_items)),
            source_id=base["id"],
            missing=missing,
            resolves=[("REFERENCES", record.po_number)],
        )
        links = {"HAS_LINE": _line_ids(base["id"], record.line_items)}
        if rows[0]["contract_id"]:
            links["GOVERNED_BY"] = [rows[0]["contract_id"]]
        if rows[0]["referenced_by"]:
            links["REFERENCED_BY"] = sorted(rows[0]["referenced_by"])
        return LinkResult(base["id"], "PurchaseOrder", vendor, links, unresolved)

    def _payment(
        self, record: Payment, base: dict[str, Any], common: dict[str, Any], vendor: VendorMatch
    ) -> LinkResult:
        props = common | {
            "payment_reference": record.payment_reference,
            "vendor_name": record.vendor_name,
            "payer_name": record.payer_name,
            "method": record.method,
            "currency": record.currency,
            "amount": _money(record.amount),
            "invoice_numbers": [a.invoice_number for a in record.allocations],
            "allocated_amounts": [_money(a.amount) for a in record.allocations],
        }

        def missing(rows: list[dict[str, Any]]) -> list[UnresolvedLink]:
            return [
                UnresolvedLink(
                    "Payment",
                    base["id"],
                    "SETTLES",
                    "Invoice",
                    "invoice_number",
                    row["invoice_number"],
                    vendor.vendor_id,
                )
                for row in rows
                if not row["invoice_id"]
            ]

        rows, unresolved = self._write(
            (_PAYMENT, base | {"props": props, "paid_at": _iso(record.paid_at)}),
            source_id=base["id"],
            missing=missing,
        )
        settled = sorted({row["invoice_id"] for row in rows if row["invoice_id"]})
        links = {"SETTLES": settled} if settled else {}
        return LinkResult(base["id"], "Payment", vendor, links, unresolved)

    def _contract(
        self, record: Contract, base: dict[str, Any], common: dict[str, Any], vendor: VendorMatch
    ) -> LinkResult:
        props = common | {
            "contract_number": record.contract_number,
            "title": record.title,
            "vendor_name": record.vendor_name,
            "customer_name": record.customer_name,
            "payment_terms_days": record.payment_terms_days,
            "currency": record.currency,
            # The rate card as parallel lists (Neo4j properties cannot hold maps), so
            # graph retrieval can put an invoice's prices next to the agreed ones.
            "rate_items": [entry.item for entry in record.rate_card],
            "rate_units": [entry.unit or "" for entry in record.rate_card],
            "rate_prices": [_money(entry.unit_price) for entry in record.rate_card],
        }
        params = base | {
            "props": props,
            "start_date": _iso(record.start_date),
            "end_date": _iso(record.end_date),
        }
        rows, _ = self._write(
            (_CONTRACT, params),
            source_id=base["id"],
            resolves=[("GOVERNS", record.contract_number)] if record.contract_number else [],
        )
        links = {"PARTY_TO": [vendor.vendor_id]}
        if rows[0]["governs"]:
            links["GOVERNS"] = sorted(rows[0]["governs"])
        return LinkResult(base["id"], "Contract", vendor, links)

    def unlink_document(self, document_id: str, *, keep: str | None = None) -> int:
        """Remove the record nodes (and their line items and unresolved links) that a
        document produced, except those labelled ``keep`` -- for a re-ingest whose
        extraction changed the document type. Vendors are shared and stay.
        Returns how many record nodes were removed."""
        removed = 0
        for label in RECORD_LABELS:
            if label == keep:
                continue
            rows = run_write(
                self.driver,
                f"""
                MATCH (n:{label} {{document_id: $document_id}})
                OPTIONAL MATCH (n)-[:HAS_LINE]->(line:LineItem)
                OPTIONAL MATCH (u:UnresolvedLink {{source_id: n.id}})
                DETACH DELETE line, u
                WITH DISTINCT n
                DETACH DELETE n
                RETURN count(*) AS removed
                """,
                {"document_id": document_id},
                database=self.database,
            )
            removed += rows[0]["removed"] if rows else 0
        return removed

    def unresolved(self, rel: str | None = None) -> list[UnresolvedLink]:
        """Recorded references that still have no target, optionally of one edge type."""
        rows = run_read(
            self.driver,
            "MATCH (u:UnresolvedLink) WHERE $rel IS NULL OR u.rel = $rel "
            "RETURN u ORDER BY u.source_label, u.source_id, u.value",
            {"rel": rel},
            database=self.database,
        )
        return [
            UnresolvedLink(
                source_label=u["source_label"],
                source_id=u["source_id"],
                rel=u["rel"],
                target_label=u["target_label"],
                key=u["key"],
                value=u["value"],
                vendor_id=u.get("vendor_id"),
            )
            for u in (row["u"] for row in rows)
        ]

    # ------------------------------------------------------------------ plumbing

    def _write(
        self,
        *statements: tuple[str, dict[str, Any]],
        source_id: str,
        missing: Callable[[list[dict[str, Any]]], list[UnresolvedLink]] | None = None,
        resolves: Sequence[tuple[str, str]] = (),
    ) -> tuple[list[dict[str, Any]], list[UnresolvedLink]]:
        """One write transaction: the statements, then this record's unresolved links
        (``missing`` reads them from the first statement's rows), then clearing the ones
        satisfied by this record being linked (``resolves``: ``(rel, value)`` pairs).
        Returns the first statement's rows and this record's unresolved links."""

        def work(tx: ManagedTransaction) -> tuple[list[dict[str, Any]], list[UnresolvedLink]]:
            first: list[dict[str, Any]] | None = None
            for query, params in statements:
                rows = [record.data() for record in tx.run(query, params)]
                if first is None:
                    first = rows
            result = first or []
            unresolved = missing(result) if missing is not None else []
            tx.run(
                _RECORD_UNRESOLVED,
                {"source_id": source_id, "links": [u.to_props() for u in unresolved]},
            ).consume()
            for rel, value in resolves:
                tx.run(_CLEAR_RESOLVED[rel], {"value": value}).consume()
            return result, unresolved

        with write_session(self.driver, database=self.database) as session:
            return session.execute_write(work)


def _line_ids(record_id: str, lines: Sequence[LineItem]) -> list[str]:
    return [f"{record_id}#l{n}" for n in range(len(lines))]
