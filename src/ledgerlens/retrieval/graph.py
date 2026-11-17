"""Graph expansion: from seed documents to the facts around them in Neo4j.

Vector search finds the documents that *read* like the question; the answer to a
relationship question ("which payment settled INV-889?", "Acme invoices whose PO has no
payment") lies in documents linked to those. :class:`GraphExpander` takes seeds --
document ids (R3's top chunks) and document numbers from the question -- and runs a
parameterized template from :mod:`ledgerlens.storage.neo4j.queries` to collect them::

    expander = GraphExpander(get_driver())
    facts = expander.expand_facts(["INV-889", "5a0aef53-..."])  # invoices within two hops

Each row is rendered as one compact :class:`~ledgerlens.retrieval.GraphFact` an LLM can
read and cite::

    Invoice INV-889 (Acme Supplies Ltd, 4,200.00 GBP, dated 2026-03-14, due 2026-04-13)
    -> PO-1042 -> contract CT-V001 (Net 30) -> no payment

with ``citations`` mapping ``doc:INV-889``, ``doc:PO-1042``, ... to the document ids of the
records that came from a document.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date
from decimal import Decimal
from typing import Any, Protocol

from neo4j import Driver

from ledgerlens.retrieval import GraphFact
from ledgerlens.storage.neo4j.driver import run_read
from ledgerlens.storage.neo4j.queries import (
    CONTRACT_FOR_PO,
    INVOICE_NEIGHBORHOOD,
    PAID_TOLERANCE,
    UNPAID_WITH_PO,
    VENDOR_LEDGER,
    CypherTemplate,
)

Row = dict[str, Any]

DEFAULT_PER_SEED = 10
DEFAULT_LIMIT = 30


def native(value: Any) -> Any:
    """``value`` with Neo4j temporal types (``neo4j.time.Date``, ...) turned into their
    ``datetime`` equivalents, inside lists and maps too."""
    if isinstance(value, Mapping):
        return {key: native(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [native(item) for item in value]
    to_native = getattr(value, "to_native", None)
    return to_native() if callable(to_native) else value


class TemplateRunner(Protocol):
    """Runs a Cypher template (the Neo4j driver in production, a stub in unit tests)."""

    def run(self, template: CypherTemplate, params: Mapping[str, Any]) -> list[Row]: ...


class Neo4jRunner:
    """:class:`TemplateRunner` on a Neo4j driver: a read transaction per query."""

    def __init__(self, driver: Driver, *, database: str | None = None) -> None:
        self.driver = driver
        self.database = database

    def run(self, template: CypherTemplate, params: Mapping[str, Any]) -> list[Row]:
        rows = run_read(self.driver, template.cypher, params, database=self.database)
        return [native(row) for row in rows]


# --------------------------------------------------------------------------- rendering

ARROW = " -> "
MAX_LINES = 6
"""Line items shown per PO or invoice in a ``contract_for_po`` fact."""


def money(amount: float | Decimal | None, currency: str | None = None) -> str:
    """``4200.0, "GBP"`` -> ``"4,200.00 GBP"``; ``None`` -> ``"?"``."""
    if amount is None:
        return "?"
    text = f"{float(amount):,.2f}"
    return f"{text} {currency}" if currency else text


def _quantity(value: float | None) -> str:
    if value is None:
        return "?"
    return f"{value:g}" if float(value).is_integer() else f"{value:,.4g}"


def _day(value: date | None) -> str:
    return value.isoformat() if value else "?"


class _Cites:
    """Collects the documents a fact names: ``doc:<number>`` -> document id."""

    def __init__(self) -> None:
        self.tags: dict[str, str] = {}

    def add(self, number: str | None, document_id: str | None) -> None:
        if number and document_id:
            self.tags.setdefault(f"doc:{number}", document_id)

    def fact(self, text: str, source: str, data: Row) -> GraphFact:
        return GraphFact(
            text=text,
            document_ids=tuple(dict.fromkeys(self.tags.values())),
            source=source,
            citations=self.tags,
            data=data,
        )


def _paid_part(row: Row) -> str:
    payments = row.get("payments") or []
    if not payments:
        return "no payment"
    currency = row.get("currency")
    total, paid = row.get("total"), float(row.get("paid") or 0.0)
    by = ", ".join(
        f"{p.get('payment_reference') or p.get('id')} on {_day(p.get('paid_at'))}"
        f" ({money(p.get('amount'))})"
        for p in payments
    )
    outstanding = (total or 0.0) - paid
    if total is not None and outstanding >= PAID_TOLERANCE:
        return (
            f"paid {money(paid)} of {money(total)} by {by}; "
            f"outstanding {money(outstanding, currency)}"
        )
    return f"paid in full by {by}"


def render_invoice(row: Row, source: str = INVOICE_NEIGHBORHOOD.name) -> GraphFact:
    """An ``invoice_neighborhood`` / ``unpaid_with_po`` row: the invoice, its PO, the
    contract governing the PO, and how it was paid."""
    cites = _Cites()
    number = row.get("invoice_number") or row.get("invoice_id")
    cites.add(number, row.get("document_id"))
    kind = "Receipt" if row.get("doc_type") == "receipt" else "Invoice"
    details = [
        row.get("vendor") or row.get("vendor_id") or "unknown vendor",
        money(row.get("total"), row.get("currency")),
    ]
    if row.get("invoice_date"):
        details.append(f"dated {_day(row['invoice_date'])}")
    if row.get("due_date"):
        details.append(f"due {_day(row['due_date'])}")
    parts = [f"{kind} {number} ({', '.join(details)})"]

    po = row.get("po")
    if po:
        cites.add(po.get("po_number"), po.get("document_id"))
        parts.append(str(po.get("po_number") or po.get("id")))
    else:
        parts.append("no PO")
    contract = row.get("contract")
    if contract:
        cites.add(contract.get("contract_number"), contract.get("document_id"))
        terms = contract.get("payment_terms_days")
        label = f"contract {contract.get('contract_number') or contract.get('id')}"
        parts.append(f"{label} (Net {terms})" if terms is not None else label)
    for payment in row.get("payments") or []:
        cites.add(payment.get("payment_reference"), payment.get("document_id"))
    parts.append(_paid_part(row))
    return cites.fact(ARROW.join(parts), source, row)


def render_ledger(row: Row) -> GraphFact:
    """A ``vendor_ledger`` row: one vendor's invoiced, paid, outstanding and overdue."""
    cites = _Cites()
    currency = row.get("currency")
    text = (
        f"Vendor {row.get('vendor') or row.get('vendor_id')} ({row.get('vendor_id')}), "
        f"{currency}: {row.get('invoices', 0)} invoices dated "
        f"{_day(row.get('first_invoice'))} to {_day(row.get('last_invoice'))}; "
        f"invoiced {money(row.get('invoiced'), currency)}, "
        f"paid {money(row.get('paid'), currency)}, "
        f"outstanding {money(row.get('outstanding'), currency)} "
        f"on {row.get('open_invoices', 0)} invoices "
        f"(overdue {money(row.get('overdue'), currency)})"
    )
    contracts = [c for c in row.get("contracts") or [] if c]
    for contract in contracts:
        cites.add(contract.get("contract_number"), contract.get("document_id"))
    if contracts:
        numbers = (str(c.get("contract_number") or c.get("id")) for c in contracts)
        text += "; contract " + ", ".join(numbers)
    return cites.fact(text, VENDOR_LEDGER.name, row)


Rates = dict[str, tuple[str, float, str]]
"""Rate card by case-folded item: ``(item as written, unit price, unit)``."""


def _rates(contract: Row) -> Rates:
    items = contract.get("rate_items") or []
    units = contract.get("rate_units") or [""] * len(items)
    prices = contract.get("rate_prices") or []
    return {
        str(item).casefold(): (str(item), float(price), str(unit))
        for item, unit, price in zip(items, units, prices, strict=False)
    }


def _line(line: Row, rates: Rates) -> str:
    price = line.get("unit_price")
    text = f"{line.get('description')} {_quantity(line.get('quantity'))} x {money(price)}"
    rate = rates.get(str(line.get("description") or "").casefold())
    if rate is None or price is None:
        return text
    agreed = rate[1]
    if abs(float(price) - agreed) < PAID_TOLERANCE:
        return f"{text} (= contract rate)"
    change = (float(price) - agreed) / agreed * 100 if agreed else 0.0
    return f"{text} (contract rate {money(agreed)}, {change:+.1f}%)"


def _lines(lines: Sequence[Row], rates: Rates) -> str:
    shown = "; ".join(_line(line, rates) for line in lines[:MAX_LINES])
    more = len(lines) - MAX_LINES
    return shown + (f"; +{more} more lines" if more > 0 else "")


def _contract_part(row: Row, cites: _Cites) -> tuple[str, Rates]:
    contract = row.get("contract")
    if not contract:
        if row.get("cited_contract"):
            return f"cites contract {row['cited_contract']}, which is not in the graph", {}
        return "no contract", {}
    number = contract.get("contract_number") or contract.get("id")
    cites.add(number, contract.get("document_id"))
    rates = _rates(contract)
    terms = contract.get("payment_terms_days")
    text = f"governed by contract {number}"
    if contract.get("title"):
        text += f' "{contract["title"]}"'
    text += f" ({_day(contract.get('start_date'))} to {_day(contract.get('end_date'))}"
    text += f", Net {terms})" if terms is not None else ")"
    if rates:
        card = "; ".join(
            f"{item} {money(price)}" + (f"/{unit}" if unit else "")
            for item, price, unit in rates.values()
        )
        text += f"; rate card: {card}"
    return text, rates


def render_contract(row: Row) -> GraphFact:
    """A ``contract_for_po`` row: the PO, the contract governing it with its terms and rate
    card, and the PO's and its invoices' line prices against the agreed rates."""
    cites = _Cites()
    currency = row.get("currency")
    po_number = row.get("po_number") or row.get("po_id")
    cites.add(po_number, row.get("document_id"))
    head = (
        f"PO {po_number} ({row.get('vendor') or row.get('vendor_id')}, "
        f"{money(row.get('total'), currency)}, ordered {_day(row.get('order_date'))})"
    )
    contract, rates = _contract_part(row, cites)
    text = head + ARROW + contract
    if row.get("po_lines"):
        text += f". PO lines: {_lines(row['po_lines'], rates)}"
    for invoice in row.get("invoices") or []:
        number = invoice.get("invoice_number") or invoice.get("id")
        cites.add(number, invoice.get("document_id"))
        text += f". Invoice {number} ({money(invoice.get('total'), currency)})"
        if invoice.get("lines"):
            text += f" lines: {_lines(invoice['lines'], rates)}"
    return cites.fact(text, CONTRACT_FOR_PO.name, row)


def _render_unpaid(row: Row) -> GraphFact:
    return render_invoice(row, UNPAID_WITH_PO.name)


RENDERERS: dict[str, Callable[[Row], GraphFact]] = {
    INVOICE_NEIGHBORHOOD.name: render_invoice,
    UNPAID_WITH_PO.name: _render_unpaid,
    VENDOR_LEDGER.name: render_ledger,
    CONTRACT_FOR_PO.name: render_contract,
}


def render(template: CypherTemplate, rows: Sequence[Row]) -> list[GraphFact]:
    """One fact per row of ``template``."""
    renderer = RENDERERS[template.name]
    return [renderer(row) for row in rows]


# --------------------------------------------------------------------------- expansion


def unique(values: Sequence[str | None]) -> list[str]:
    """``values`` without blanks and repeats, in order."""
    return list(dict.fromkeys(value for value in values if value))


class GraphExpander:
    """Expands seed documents through the knowledge graph."""

    def __init__(
        self,
        driver: Driver | None = None,
        *,
        runner: TemplateRunner | None = None,
        database: str | None = None,
        per_seed: int = DEFAULT_PER_SEED,
        limit: int = DEFAULT_LIMIT,
    ) -> None:
        if runner is None:
            if driver is None:
                raise ValueError("GraphExpander needs a Neo4j driver or a runner")
            runner = Neo4jRunner(driver, database=database)
        self.runner = runner
        self.per_seed = per_seed
        self.limit = limit

    def rows(self, template: CypherTemplate, **params: Any) -> list[Row]:
        """Run one template (unknown or missing parameters are an error)."""
        return self.runner.run(template, template.bind(**params))

    def expand(self, seeds: Sequence[str | None]) -> list[Row]:
        """Invoices within two hops of the seeds -- document ids or document numbers, best
        first -- with their vendor, PO, contract and payments (``invoice_neighborhood``)."""
        wanted = unique(seeds)
        if not wanted:
            return []
        return self.rows(
            INVOICE_NEIGHBORHOOD, seeds=wanted, per_seed=self.per_seed, limit=self.limit
        )

    def expand_facts(self, seeds: Sequence[str | None]) -> list[GraphFact]:
        """:meth:`expand`, rendered as facts."""
        return render(INVOICE_NEIGHBORHOOD, self.expand(seeds))
