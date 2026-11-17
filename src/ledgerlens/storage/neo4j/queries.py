"""Parameterized Cypher templates for graph retrieval (docs/layers/03-retrieval.md).

The retrieval layer reads the graph through these templates rather than generated Cypher
whenever a question has a known shape. Each one is read-only, takes every value as a
parameter (never string-formatted into the query), and caps its rows with ``$limit``:

* :data:`INVOICE_NEIGHBORHOOD` -- invoices reached from seed documents or document numbers
  within two hops (an invoice itself; the invoices a payment settles or that reference a
  PO; the invoices under a contract's POs), each with its vendor, PO, contract and
  payments.
* :data:`UNPAID_WITH_PO` -- invoices that reference a PO but are not fully paid, oldest
  due date first, optionally for some vendors and an invoice-date range.
* :data:`VENDOR_LEDGER` -- per vendor and currency: invoices, amount invoiced, paid,
  outstanding and overdue.
* :data:`CONTRACT_FOR_PO` -- the contract governing a PO (found by PO or invoice, number or
  document id), with its terms and rate card next to the PO's and invoices' line prices.

Invoice rows (the first two templates) share one shape, see :data:`INVOICE_COLUMNS`.
Dates come back as ``neo4j.time.Date``; :func:`ledgerlens.retrieval.graph.native` turns
them into ``datetime.date``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

PAID_TOLERANCE = 0.01
"""An invoice counts as paid when payments leave less than this outstanding."""


@dataclass(frozen=True)
class CypherTemplate:
    """A named, read-only Cypher query and the parameters it takes."""

    name: str
    cypher: str
    params: tuple[str, ...]
    """Every ``$parameter`` the query uses."""
    description: str
    defaults: Mapping[str, Any] = field(default_factory=dict)

    def bind(self, **values: Any) -> dict[str, Any]:
        """The parameters for one run: ``defaults`` overridden by ``values``. A missing or
        unknown parameter is an error, so a typo cannot silently match nothing."""
        unknown = sorted(set(values) - set(self.params))
        if unknown:
            raise ValueError(f"{self.name} takes no parameter(s) {', '.join(unknown)}")
        bound = {**self.defaults, **values}
        missing = [name for name in self.params if name not in bound]
        if missing:
            raise ValueError(f"{self.name} needs {', '.join(missing)}")
        return bound


# --------------------------------------------------------------------------- shared parts

_PAYMENTS = """
OPTIONAL MATCH (pay:Payment)-[s:SETTLES]->(i)
WITH i, rank, hops, v, po, c,
     collect(pay {.id, .payment_reference, .document_id, .paid_at, amount: s.amount}) AS payments,
     coalesce(sum(s.amount), 0.0) AS paid
"""

INVOICE_COLUMNS = """
RETURN i.id AS invoice_id, i.invoice_number AS invoice_number, i.document_id AS document_id,
       i.doc_type AS doc_type, i.invoice_date AS invoice_date, i.due_date AS due_date,
       i.total AS total, i.currency AS currency, i.vendor_id AS vendor_id,
       coalesce(v.name, i.vendor_name) AS vendor,
       po {.id, .po_number, .document_id, .total, .order_date} AS po,
       c {.id, .contract_number, .document_id, .title, .payment_terms_days} AS contract,
       payments, paid, i.total - paid AS outstanding, hops
"""
"""Columns of an invoice row: the invoice, its vendor's name, its PO and contract (maps,
``null`` when missing), the payments settling it, ``paid`` / ``outstanding``, and ``hops``
from the seed it was reached from (0 for the seed itself)."""

_DATE_RANGE = """
  AND ($date_from IS NULL OR i.invoice_date >= date($date_from))
  AND ($date_to IS NULL OR i.invoice_date <= date($date_to))
"""

# --------------------------------------------------------------------------- templates

INVOICE_NEIGHBORHOOD = CypherTemplate(
    name="invoice_neighborhood",
    description="Invoices within two hops of seed documents, with vendor, PO, contract and "
    "payments.",
    params=("seeds", "per_seed", "limit"),
    defaults={"per_seed": 10, "limit": 50},
    cypher="""
UNWIND range(0, size($seeds) - 1) AS rank
WITH rank, $seeds[rank] AS seed
CALL (seed) {
  CALL (seed) {
    MATCH (i:Invoice) WHERE i.document_id = seed OR i.invoice_number = seed
    RETURN i, 0 AS hops
    UNION
    MATCH (p:Payment)-[:SETTLES]->(i:Invoice)
    WHERE p.document_id = seed OR p.payment_reference = seed
    RETURN i, 1 AS hops
    UNION
    MATCH (po:PurchaseOrder)<-[:REFERENCES]-(i:Invoice)
    WHERE po.document_id = seed OR po.po_number = seed
    RETURN i, 1 AS hops
    UNION
    MATCH (c:Contract)-[:GOVERNS]->(:PurchaseOrder)<-[:REFERENCES]-(i:Invoice)
    WHERE c.document_id = seed OR c.contract_number = seed
    RETURN i, 2 AS hops
  }
  WITH i, hops ORDER BY hops, i.invoice_date DESC, i.id LIMIT $per_seed
  RETURN i, hops
}
WITH i, min(rank) AS rank, min(hops) AS hops
ORDER BY rank, hops, i.invoice_date DESC, i.id
LIMIT $limit
OPTIONAL MATCH (v:Vendor)-[:ISSUED]->(i)
OPTIONAL MATCH (i)-[:REFERENCES]->(po:PurchaseOrder)
OPTIONAL MATCH (c:Contract)-[:GOVERNS]->(po)
"""
    + _PAYMENTS
    + INVOICE_COLUMNS
    + "ORDER BY rank, hops, invoice_date DESC, invoice_id",
)
"""``seeds``: document ids and/or document numbers (invoice, payment, PO or contract), best
first. At most ``per_seed`` invoices per seed (nearest first, then newest), ``limit`` in
all, in seed order."""


UNPAID_WITH_PO = CypherTemplate(
    name="unpaid_with_po",
    description="Invoices that reference a PO but are not fully paid, oldest due first.",
    params=("vendor_ids", "date_from", "date_to", "tolerance", "limit"),
    defaults={
        "vendor_ids": [],
        "date_from": None,
        "date_to": None,
        "tolerance": PAID_TOLERANCE,
        "limit": 50,
    },
    cypher="""
MATCH (v:Vendor)-[:ISSUED]->(i:Invoice)-[:REFERENCES]->(po:PurchaseOrder)
WHERE (size($vendor_ids) = 0 OR v.id IN $vendor_ids)
"""
    + _DATE_RANGE
    + """
OPTIONAL MATCH (c:Contract)-[:GOVERNS]->(po)
WITH i, 0 AS rank, 0 AS hops, v, po, c
"""
    + _PAYMENTS
    + """
WHERE i.total - paid >= $tolerance
"""
    + INVOICE_COLUMNS
    + "ORDER BY due_date, invoice_date, invoice_id LIMIT $limit",
)
"""``vendor_ids`` (empty: every vendor) and ``date_from`` / ``date_to`` (ISO dates or
``null``) filter on the invoice; partly paid invoices are included, with ``outstanding``."""


VENDOR_LEDGER = CypherTemplate(
    name="vendor_ledger",
    description="Per vendor and currency: invoices, invoiced, paid, outstanding, overdue.",
    params=("vendor_ids", "date_from", "date_to", "as_of", "tolerance", "limit"),
    defaults={
        "vendor_ids": [],
        "date_from": None,
        "date_to": None,
        "tolerance": PAID_TOLERANCE,
        "limit": 20,
    },
    cypher="""
MATCH (v:Vendor)-[:ISSUED]->(i:Invoice)
WHERE (size($vendor_ids) = 0 OR v.id IN $vendor_ids)
"""
    + _DATE_RANGE
    + """
OPTIONAL MATCH (:Payment)-[s:SETTLES]->(i)
WITH v, i, coalesce(sum(s.amount), 0.0) AS paid
WITH v, i, paid,
     CASE WHEN i.total - paid >= $tolerance THEN i.total - paid ELSE 0.0 END AS open_amount
WITH v, i.currency AS currency,
     count(i) AS invoices,
     sum(i.total) AS invoiced,
     sum(paid) AS paid,
     sum(CASE WHEN open_amount > 0 THEN 1 ELSE 0 END) AS open_invoices,
     sum(open_amount) AS outstanding,
     sum(CASE WHEN open_amount > 0 AND i.due_date < date($as_of) THEN open_amount ELSE 0.0 END)
       AS overdue,
     min(i.invoice_date) AS first_invoice,
     max(i.invoice_date) AS last_invoice
OPTIONAL MATCH (v)-[:PARTY_TO]->(c:Contract)
WITH v, currency, invoices, invoiced, paid, open_invoices, outstanding, overdue,
     first_invoice, last_invoice,
     collect(c {.id, .contract_number, .document_id}) AS contracts
RETURN v.id AS vendor_id, v.name AS vendor, currency, invoices, invoiced, paid,
       open_invoices, outstanding, overdue, first_invoice, last_invoice, contracts
ORDER BY outstanding DESC, invoiced DESC, vendor_id, currency
LIMIT $limit
""",
)
"""``as_of`` (ISO date) decides what is overdue: open invoices due before it."""


CONTRACT_FOR_PO = CypherTemplate(
    name="contract_for_po",
    description="The contract governing a PO, its terms and rate card, with the PO's and "
    "its invoices' line prices.",
    params=("seeds", "limit"),
    defaults={"limit": 10},
    cypher="""
UNWIND range(0, size($seeds) - 1) AS rank
WITH rank, $seeds[rank] AS seed
CALL (seed) {
  MATCH (po:PurchaseOrder) WHERE po.document_id = seed OR po.po_number = seed
  RETURN po
  UNION
  MATCH (i:Invoice)-[:REFERENCES]->(po:PurchaseOrder)
  WHERE i.document_id = seed OR i.invoice_number = seed
  RETURN po
}
WITH po, min(rank) AS rank
ORDER BY rank, po.id
LIMIT $limit
OPTIONAL MATCH (c:Contract)-[:GOVERNS]->(po)
CALL (po) {
  OPTIONAL MATCH (po)-[:HAS_LINE]->(l:LineItem)
  WITH l ORDER BY l.position
  RETURN collect(l {.description, .quantity, .unit_price, .amount}) AS po_lines
}
CALL (po) {
  OPTIONAL MATCH (i:Invoice)-[:REFERENCES]->(po)
  OPTIONAL MATCH (i)-[:HAS_LINE]->(l:LineItem)
  WITH i, l ORDER BY i.id, l.position
  WITH i, collect(l {.description, .quantity, .unit_price, .amount}) AS lines
  RETURN collect(CASE WHEN i IS NULL THEN null ELSE
           i {.id, .invoice_number, .document_id, .invoice_date, .total, lines: lines} END)
         AS invoices
}
RETURN po.id AS po_id, po.po_number AS po_number, po.document_id AS document_id,
       po.order_date AS order_date, po.total AS total, po.currency AS currency,
       po.vendor_name AS vendor, po.vendor_id AS vendor_id, po.contract_number AS cited_contract,
       c {.id, .contract_number, .document_id, .title, .start_date, .end_date,
          .payment_terms_days, .currency, .rate_items, .rate_units, .rate_prices} AS contract,
       po_lines, invoices
ORDER BY rank, po_id
""",
)
"""``seeds``: PO or invoice document ids / numbers. ``contract`` is ``null`` when no contract
governs the PO (``cited_contract`` is the number the PO prints, if any)."""


TEMPLATES: dict[str, CypherTemplate] = {
    template.name: template
    for template in (INVOICE_NEIGHBORHOOD, UNPAID_WITH_PO, VENDOR_LEDGER, CONTRACT_FOR_PO)
}
