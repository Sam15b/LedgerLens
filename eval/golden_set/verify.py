"""Derive a generated golden question's answer a second time, from the graph.

The builder (``build_golden_set.py``) computes answers from Postgres. Every question it
generates carries a ``check``: its kind and record ids. Here each kind is answered again
**independently**, with Cypher over Neo4j (the graph links and SETTLES amounts, not the
builder's Python), and with SQL for the one thing the graph does not hold, a contract's
text. The derived facts are compared with the question's ``expected``; a question passes
when every fact the check derives agrees.

    python -m eval.golden_set.verify data/golden/golden_set.jsonl

A question without a ``check`` (golden set v0, hand-written questions) is skipped.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from eval.golden_set.schema import DEFAULT_PATH, GoldenQuestion, load_golden_set

Rows = list[dict[str, Any]]
Cypher = Callable[[str, Mapping[str, Any]], Rows]
"""Runs a read query and returns its rows as dicts."""
ContractText = Callable[[str], str]
"""A contract's full text by its record id (from Postgres chunks)."""

TOLERANCE = Decimal("0.01")


def money(value: Any) -> str:
    return f"{Decimal(str(value)).quantize(Decimal('0.01')):,.2f}"


def total(values: Sequence[Any]) -> Decimal:
    return sum((Decimal(str(v)) for v in values), Decimal(0))


def iso(value: Any) -> str:
    return str(value)[:10]


@dataclass(frozen=True)
class Verdict:
    question_id: str
    kind: str
    ok: bool
    detail: str = ""


class CheckError(ValueError):
    """The check could not run (a record is missing, a path does not exist)."""


def one(rows: Rows, what: str) -> dict[str, Any]:
    if len(rows) != 1:
        raise CheckError(f"{what}: expected one row, got {len(rows)}")
    return rows[0]


# --------------------------------------------------------------------------- kinds

Derive = Callable[[Cypher, ContractText, Mapping[str, Any]], dict[str, Any]]
KINDS: dict[str, Derive] = {}


def kind(name: str) -> Callable[[Derive], Derive]:
    def register(fn: Derive) -> Derive:
        KINDS[name] = fn
        return fn

    return register


def _invoice(cypher: Cypher, invoice_id: str) -> dict[str, Any]:
    rows = cypher("MATCH (i:Invoice {id: $id}) RETURN i", {"id": invoice_id})
    return dict(one(rows, "invoice")["i"])


@kind("invoice_total")
def _invoice_total(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    i = _invoice(cypher, c["invoice_id"])
    return {"amount": money(i["total"]), "currency": i["currency"]}


@kind("invoice_due")
def _invoice_due(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    return {"date": iso(_invoice(cypher, c["invoice_id"])["due_date"])}


@kind("invoice_date")
def _invoice_date(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    return {"date": iso(_invoice(cypher, c["invoice_id"])["invoice_date"])}


@kind("invoice_po")
def _invoice_po(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    return {"number": _invoice(cypher, c["invoice_id"])["po_number"]}


_PO_VENDOR = """
MATCH (p:PurchaseOrder {id: $id}) OPTIONAL MATCH (v:Vendor {id: p.vendor_id})
RETURN p.total AS total, p.currency AS currency, v.name AS vendor
"""


@kind("po_total")
def _po_total(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    row = one(cypher(_PO_VENDOR, {"id": c["po_id"]}), "purchase order")
    return {"amount": money(row["total"]), "currency": row["currency"]}


@kind("po_vendor")
def _po_vendor(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    return {"vendor": one(cypher(_PO_VENDOR, {"id": c["po_id"]}), "purchase order")["vendor"]}


_PAYMENT = """
MATCH (p:Payment {id: $id}) OPTIONAL MATCH (v:Vendor {id: p.vendor_id})
RETURN p.amount AS amount, p.currency AS currency, p.paid_at AS paid_at, v.name AS vendor
"""


@kind("payment_amount")
def _payment_amount(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    row = one(cypher(_PAYMENT, {"id": c["payment_id"]}), "payment")
    return {"amount": money(row["amount"]), "currency": row["currency"], "vendor": row["vendor"]}


@kind("payment_date")
def _payment_date(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    return {"date": iso(one(cypher(_PAYMENT, {"id": c["payment_id"]}), "payment")["paid_at"])}


def _contract(cypher: Cypher, contract_id: str) -> dict[str, Any]:
    rows = cypher("MATCH (c:Contract {id: $id}) RETURN c", {"id": contract_id})
    return dict(one(rows, "contract")["c"])


def _rate_entry(contract: Mapping[str, Any], item: str) -> tuple[Any, Any]:
    items = list(contract.get("rate_items") or [])
    if item not in items:
        raise CheckError(f"{item!r} is not on the rate card")
    k = items.index(item)
    return contract["rate_prices"][k], (contract.get("rate_units") or [None] * len(items))[k]


@kind("contract_terms")
def _contract_terms(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    return {"days": _contract(cypher, c["contract_id"])["payment_terms_days"]}


@kind("contract_rate")
def _contract_rate(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    contract = _contract(cypher, c["contract_id"])
    price, _unit = _rate_entry(contract, c["item"])
    return {"amount": money(price), "currency": contract["currency"]}


@kind("contract_unit")
def _contract_unit(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    _price, unit = _rate_entry(_contract(cypher, c["contract_id"]), c["item"])
    return {"items": [unit]}


@kind("contract_end")
def _contract_end(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    return {"date": iso(_contract(cypher, c["contract_id"])["end_date"])}


@kind("contract_start")
def _contract_start(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    return {"date": iso(_contract(cypher, c["contract_id"])["start_date"])}


# multi-hop -----------------------------------------------------------------------------

_SETTLES = """
MATCH (i:Invoice {id: $id})
OPTIONAL MATCH (p:Payment)-[s:SETTLES]->(i)
RETURN i.total AS total, i.currency AS currency,
       collect({reference: p.payment_reference, paid_at: p.paid_at, amount: s.amount}) AS paid
"""


def _settles(cypher: Cypher, invoice_id: str) -> tuple[Decimal, list[dict[str, Any]], str]:
    row = one(cypher(_SETTLES, {"id": invoice_id}), "invoice")
    paid = [p for p in row["paid"] if p["reference"] is not None]
    return Decimal(str(row["total"])), paid, row["currency"]


def _outstanding(cypher: Cypher, invoice_id: str) -> Decimal:
    invoice_total, paid, _ = _settles(cypher, invoice_id)
    return invoice_total - total([p["amount"] for p in paid])


@kind("settled_by")
def _settled_by(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    _total, paid, _currency = _settles(cypher, c["invoice_id"])
    p = one(paid, "settling payment")
    return {"number": p["reference"], "date": iso(p["paid_at"])}


@kind("owed")
def _owed(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    _total, _paid, currency = _settles(cypher, c["invoice_id"])
    return {"amount": money(_outstanding(cypher, c["invoice_id"])), "currency": currency}


_GOVERNS = """
MATCH (i:Invoice {id: $id})-[:REFERENCES]->(po:PurchaseOrder)<-[:GOVERNS]-(c:Contract)
RETURN po.po_number AS po_number, c.contract_number AS contract,
       c.payment_terms_days AS days, c.rate_items AS items, c.rate_prices AS prices,
       i.invoice_date AS invoice_date, i.due_date AS due_date
"""


@kind("governing_contract")
def _governing(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    row = one(cypher(_GOVERNS, {"id": c["invoice_id"]}), "invoice -> PO -> contract")
    return {"number": row["contract"], "po_number": row["po_number"]}


@kind("terms_via_contract")
def _terms_via(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    row = one(cypher(_GOVERNS, {"id": c["invoice_id"]}), "invoice -> PO -> contract")
    return {"days": row["days"], "number": row["contract"]}


@kind("open_with_po")
def _open_with_po(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    rows = cypher(
        """
        MATCH (v:Vendor {id: $id})-[:ISSUED]->(i:Invoice)
        WHERE i.po_number IS NOT NULL AND NOT ()-[:SETTLES]->(i)
        RETURN i.invoice_number AS number
        """,
        {"id": c["vendor_id"]},
    )
    return {"numbers": sorted(r["number"] for r in rows)}


_BILLED = """
MATCH (i:Invoice)-[:REFERENCES]->(p:PurchaseOrder {id: $id})
RETURN i.invoice_number AS number, i.total AS total
"""


@kind("po_invoices")
def _po_invoices(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    return {"numbers": sorted(r["number"] for r in cypher(_BILLED, {"id": c["po_id"]}))}


@kind("po_invoiced")
def _po_invoiced(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    rows = cypher(_BILLED, {"id": c["po_id"]})
    return {"amount": money(total([r["total"] for r in rows])), "count": len(rows)}


# aggregation ---------------------------------------------------------------------------

_VENDOR_INVOICES = """
MATCH (v:Vendor {id: $id})-[:ISSUED]->(i:Invoice)
WHERE date(i.invoice_date) >= date($start) AND date(i.invoice_date) <= date($end)
RETURN i.total AS total, i.currency AS currency
"""


def _year(c: Mapping[str, Any]) -> dict[str, Any]:
    return {"start": f"{c['year']}-01-01", "end": f"{c['year']}-12-31"}


@kind("count_year")
def _count_year(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    return {"count": len(cypher(_VENDOR_INVOICES, {"id": c["vendor_id"], **_year(c)}))}


@kind("count_period")
def _count_period(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    params = {"id": c["vendor_id"], "start": c["start"], "end": c["end"]}
    return {"count": len(cypher(_VENDOR_INVOICES, params))}


@kind("total_period")
def _total_period(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    params = {"id": c["vendor_id"], "start": c["start"], "end": c["end"]}
    rows = cypher(_VENDOR_INVOICES, params)
    return {"amount": money(total([r["total"] for r in rows])), "count": len(rows)}


_OPEN = """
MATCH (v:Vendor {id: $id})-[:ISSUED]->(i:Invoice)
OPTIONAL MATCH ()-[s:SETTLES]->(i)
WITH i, coalesce(sum(s.amount), 0) AS paid
RETURN i.id AS id, i.total AS total, paid
"""


def _open_amounts(cypher: Cypher, vendor_id: str) -> list[Decimal]:
    rows = cypher(_OPEN, {"id": vendor_id})
    owed = [Decimal(str(r["total"])) - Decimal(str(r["paid"])) for r in rows]
    return [amount for amount in owed if amount >= TOLERANCE]


@kind("balance")
def _balance(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    owed = _open_amounts(cypher, c["vendor_id"])
    return {"amount": money(sum(owed, Decimal(0))), "count": len(owed)}


@kind("unpaid_count")
def _unpaid_count(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    return {"count": len(_open_amounts(cypher, c["vendor_id"]))}


@kind("paid_year")
def _paid_year(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    rows = cypher(
        """
        MATCH (p:Payment {vendor_id: $id})
        WHERE date(p.paid_at) >= date($start) AND date(p.paid_at) <= date($end)
        RETURN p.amount AS amount
        """,
        {"id": c["vendor_id"], **_year(c)},
    )
    return {"amount": money(total([r["amount"] for r in rows])), "count": len(rows)}


@kind("po_count_year")
def _po_count_year(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    rows = cypher(
        """
        MATCH (p:PurchaseOrder {vendor_id: $id})
        WHERE date(p.order_date) >= date($start) AND date(p.order_date) <= date($end)
        RETURN p.id AS id
        """,
        {"id": c["vendor_id"], **_year(c)},
    )
    return {"count": len(rows)}


@kind("top_vendor")
def _top_vendor(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    rows = cypher(
        """
        MATCH (v:Vendor)-[:ISSUED]->(i:Invoice {currency: $currency})
        WHERE date(i.invoice_date).year = $year
        RETURN v.id AS id, v.name AS vendor, collect(i.total) AS totals
        """,
        {"currency": c["currency"], "year": c["year"]},
    )
    ranked = sorted(((total(r["totals"]), r["vendor"]) for r in rows), reverse=True)
    if len(ranked) < 2 or ranked[0][0] == ranked[1][0]:
        raise CheckError("no single top vendor")
    return {"vendor": ranked[0][1], "amount": money(ranked[0][0])}


# reconciliation ------------------------------------------------------------------------


@kind("rate_match")
def _rate_match(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    contract = one(cypher(_GOVERNS, {"id": c["invoice_id"]}), "invoice -> PO -> contract")
    rates = dict(zip(contract["items"] or [], contract["prices"] or [], strict=True))
    lines = cypher(
        """
        MATCH (:Invoice {id: $id})-[:HAS_LINE]->(l:LineItem)
        RETURN l.description AS item, l.unit_price AS price
        """,
        {"id": c["invoice_id"]},
    )
    priced = [ln for ln in lines if ln["item"] in rates and ln["price"] is not None]
    if not priced:
        raise CheckError("no line is on the rate card")
    off = sorted(
        ln["item"]
        for ln in priced
        if abs(Decimal(str(ln["price"])) - Decimal(str(rates[ln["item"]]))) >= TOLERANCE
    )
    return {"match": not off} | ({"items": off} if off else {})


_PAIR = """
MATCH (a:Invoice {id: $id}), (b:Invoice {id: $original})
RETURN a.vendor_id = b.vendor_id AS same_vendor, a.total = b.total AS same_total,
       a.invoice_number = b.invoice_number AS same_number,
       a.invoice_date = b.invoice_date AS same_date, a.invoice_number AS number,
       b.invoice_number AS original_number
"""


@kind("duplicate")
def _duplicate(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    row = one(cypher(_PAIR, {"id": c["invoice_id"], "original": c["original_id"]}), "pair")
    same = row["same_vendor"] and row["same_total"] and row["same_number"] and row["same_date"]
    return {"duplicate": bool(same), "number": row["number"]}


@kind("near_duplicate")
def _near_duplicate(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    row = one(cypher(_PAIR, {"id": c["invoice_id"], "original": c["original_id"]}), "pair")
    near = row["same_vendor"] and row["same_total"] and not row["same_number"]
    return {"duplicate": bool(near), "number": row["original_number"]}


@kind("terms_match")
def _terms_match(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    row = one(cypher(_GOVERNS, {"id": c["invoice_id"]}), "invoice -> PO -> contract")
    gap = date.fromisoformat(iso(row["due_date"])) - date.fromisoformat(iso(row["invoice_date"]))
    return {"match": gap.days == row["days"], "days": row["days"]}


@kind("paid_in_full")
def _paid_in_full(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    owed = _outstanding(cypher, c["invoice_id"])
    paid = owed < TOLERANCE
    return {"match": paid} | ({} if paid else {"amount": money(owed)})


# unanswerable --------------------------------------------------------------------------


def _absent(cypher: Cypher, query: str, params: Mapping[str, Any]) -> dict[str, Any]:
    found = one(cypher(query, params), "count")["n"]
    return {"answerable": found > 0}


@kind("missing_invoice")
def _missing_invoice(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    query = "MATCH (i:Invoice {invoice_number: $n}) RETURN count(i) AS n"
    return _absent(cypher, query, {"n": c["number"]})


@kind("missing_po")
def _missing_po(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    query = "MATCH (p:PurchaseOrder {po_number: $n}) RETURN count(p) AS n"
    return _absent(cypher, query, {"n": c["number"]})


@kind("missing_payment")
def _missing_payment(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    query = "MATCH (p:Payment {payment_reference: $n}) RETURN count(p) AS n"
    return _absent(cypher, query, {"n": c["reference"]})


@kind("missing_vendor")
def _missing_vendor(cypher: Cypher, _: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    query = """
    MATCH (v:Vendor)
    WHERE toLower(v.name) = toLower($n)
       OR any(a IN coalesce(v.aliases, []) WHERE toLower(a) = toLower($n))
    RETURN count(v) AS n
    """
    return _absent(cypher, query, {"n": c["name"]})


@kind("missing_clause")
def _missing_clause(_: Cypher, text: ContractText, c: Mapping[str, Any]) -> dict[str, Any]:
    body = text(c["contract_id"]).lower()
    if not body:
        raise CheckError("the contract has no text")
    return {"answerable": any(word in body for word in c["keywords"])}


# --------------------------------------------------------------------------- compare


def _same(key: str, derived: Any, expected: Any) -> bool:
    if key in ("numbers", "items"):
        return sorted(map(str, derived)) == sorted(map(str, expected))
    if key in ("count", "days"):
        return int(derived) == int(expected)
    return bool(derived == expected)


def verify_question(question: GoldenQuestion, cypher: Cypher, text: ContractText) -> Verdict:
    """Pass when every fact the check derives equals the question's ``expected``."""
    check = question.check or {}
    name = str(check.get("kind", ""))
    derive = KINDS.get(name)
    if derive is None:
        return Verdict(question.id, name or "-", False, f"no check kind {name!r}")
    try:
        derived = derive(cypher, text, check)
    except CheckError as exc:
        return Verdict(question.id, name, False, str(exc))
    expected = dict(question.expected)
    if question.category == "unanswerable":
        expected.setdefault("answerable", False)
    wrong = [
        f"{key}: graph says {value!r}, question says {expected.get(key)!r}"
        for key, value in derived.items()
        if key in expected and not _same(key, value, expected[key])
    ]
    missing = [key for key in derived if key not in expected and key != "items"]
    if missing:
        wrong.append(f"not in expected: {', '.join(missing)}")
    return Verdict(question.id, name, not wrong, "; ".join(wrong))


def verify(
    questions: Sequence[GoldenQuestion], cypher: Cypher, text: ContractText
) -> list[Verdict]:
    return [verify_question(q, cypher, text) for q in questions if q.check]


# --------------------------------------------------------------------------- stores


def neo4j_cypher() -> Cypher:
    from ledgerlens.storage.neo4j.driver import get_driver

    driver = get_driver()

    def run(query: str, params: Mapping[str, Any]) -> Rows:
        with driver.session() as session:
            return [record.data() for record in session.run(query, dict(params))]

    return run


def postgres_contract_text() -> ContractText:
    from sqlalchemy import select

    from ledgerlens.storage.postgres.db import get_session_factory
    from ledgerlens.storage.postgres.models import ChunkRow, ContractRow

    sessions = get_session_factory()

    def text(contract_id: str) -> str:
        with sessions() as session:
            contract = session.get(ContractRow, contract_id)
            if contract is None or contract.document_id is None:
                return ""
            chunks = session.scalars(
                select(ChunkRow.text)
                .where(ChunkRow.document_id == contract.document_id)
                .order_by(ChunkRow.chunk_index)
            )
            return "\n".join(chunks)

    return text


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check generated answers against the graph.")
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args(argv)
    questions = load_golden_set(args.path)
    verdicts = verify(questions, neo4j_cypher(), postgres_contract_text())
    failed = [v for v in verdicts if not v.ok]
    for v in failed:
        print(f"FAIL {v.question_id} ({v.kind}): {v.detail}", file=sys.stderr)
    skipped = len(questions) - len(verdicts)
    print(f"{len(verdicts) - len(failed)}/{len(verdicts)} checks agree; {skipped} without a check")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
