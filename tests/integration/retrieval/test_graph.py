"""Graph expansion against the compose Neo4j, on a small graph seeded per test.

Every id carries a per-test tag and everything tagged is deleted afterwards, so the
development graph (the synthetic ledger) is left alone; document numbers carry the tag
too, so templates that match numbers cannot reach development data.

The seeded ledger, for vendor "Acme Supplies" (``V-<tag>``) under contract ``CT-<tag>``::

    INV-1 -> PO-1 (governed by CT)   paid in full by PAY-1
    INV-2 -> PO-2                    no payment
    INV-3 -> PO-1                    paid 100 of 330 by PAY-2 (unit price 33 vs rate 30)
    INV-4 (no PO)                    no payment

plus an unpaid invoice with a PO from another vendor, which a question about Acme must
not return.
"""

import uuid
from collections.abc import Iterator
from datetime import date
from decimal import Decimal

import pytest
from neo4j import Driver

from ledgerlens.core.schemas import (
    Contract,
    Invoice,
    LineItem,
    Payment,
    PaymentAllocation,
    PurchaseOrder,
    RateCardEntry,
)
from ledgerlens.retrieval import GraphFact
from ledgerlens.retrieval.analyze import QueryAnalyzer
from ledgerlens.retrieval.graph import GraphExpander
from ledgerlens.storage.neo4j.driver import apply_schema, get_driver, run_write
from ledgerlens.storage.neo4j.linker import GraphLinker, VendorCandidate

pytestmark = pytest.mark.integration

D = Decimal
TODAY = date(2026, 11, 17)
ACME = "Acme Supplies Ltd"


@pytest.fixture(scope="module")
def driver() -> Driver:
    driver = get_driver()
    apply_schema(driver)
    return driver


@pytest.fixture
def tag(driver: Driver) -> Iterator[str]:
    tag = f"T{uuid.uuid4().hex[:10].upper()}"  # upper: the analyzer upper-cases numbers
    yield tag
    run_write(driver, "MATCH (n) WHERE n.id CONTAINS $tag DETACH DELETE n", {"tag": tag})


def wrap(price: str, quantity: int = 10) -> LineItem:
    return LineItem(
        description="Pallet wrap 500m",
        quantity=D(quantity),
        unit_price=D(price),
        amount=D(price) * quantity,
    )


@pytest.fixture
def ledger(driver: Driver, tag: str) -> dict[str, str]:
    """Seeds the graph; returns the tagged document numbers by short name."""
    linker = GraphLinker(driver)
    acme, other = f"V-{tag}", f"W-{tag}"
    n = {key: f"{key}-{tag}" for key in ("CT", "PO-1", "PO-2", "PO-9", "PAY-1", "PAY-2")}
    n |= {f"INV-{k}": f"INV-{k}-{tag}" for k in (1, 2, 3, 4, 9)}

    def doc(key: str) -> str:
        return f"doc-{key}-{tag}"

    linker.link(
        Contract(
            contract_number=n["CT"],
            vendor_name=ACME,
            start_date=date(2026, 1, 1),
            payment_terms_days=30,
            rate_card=[RateCardEntry(item="Pallet wrap 500m", unit="roll", unit_price=D(30))],
        ),
        id=n["CT"],
        document_id=doc("CT"),
        vendor_id=acme,
    )
    for key, contract in (("PO-1", n["CT"]), ("PO-2", None)):
        purchase_order = PurchaseOrder(
            po_number=n[key],
            vendor_name=ACME,
            order_date=date(2026, 2, 1),
            contract_number=contract,
            total=D(300),
            line_items=[wrap("30")],
        )
        linker.link(purchase_order, id=n[key], document_id=doc(key), vendor_id=acme)
    for key, po, total, price in (
        ("INV-1", "PO-1", 300, "30"),
        ("INV-2", "PO-2", 300, "30"),
        ("INV-3", "PO-1", 330, "33"),
        ("INV-4", None, 300, "30"),
    ):
        day = int(key[-1])
        record = Invoice(
            invoice_number=n[key],
            vendor_name=ACME,
            invoice_date=date(2026, 3, day),
            due_date=date(2026, 4, day),
            po_number=n[po] if po else None,
            total=D(total),
            line_items=[wrap(price)],
        )
        linker.link(record, id=n[key], document_id=doc(key), vendor_id=acme)
    for key, invoice_key, amount in (("PAY-1", "INV-1", 300), ("PAY-2", "INV-3", 100)):
        payment = Payment(
            payment_reference=n[key],
            vendor_name=ACME,
            paid_at=date(2026, 4, 2),
            amount=D(amount),
            allocations=[PaymentAllocation(invoice_number=n[invoice_key], amount=D(amount))],
        )
        linker.link(payment, id=n[key], document_id=doc(key), vendor_id=acme)
    other_po = PurchaseOrder(
        po_number=n["PO-9"], vendor_name="Other Co", order_date=date(2026, 2, 1), total=D(50)
    )
    linker.link(other_po, id=n["PO-9"], vendor_id=other)
    other_invoice = Invoice(
        invoice_number=n["INV-9"],
        vendor_name="Other Co",
        invoice_date=date(2026, 3, 9),
        po_number=n["PO-9"],
        total=D(50),
    )
    linker.link(other_invoice, id=n["INV-9"], vendor_id=other)
    return n


@pytest.fixture
def analyzer(tag: str) -> QueryAnalyzer:
    vendors = [VendorCandidate(f"V-{tag}", ACME), VendorCandidate(f"W-{tag}", "Other Co")]
    return QueryAnalyzer(vendors, today=TODAY)


@pytest.fixture
def expander(driver: Driver) -> GraphExpander:
    return GraphExpander(driver, today=TODAY)


def short(facts: list[GraphFact]) -> list[str]:
    """``INV-2-<tag>`` -> ``INV-2`` for each invoice fact."""
    return [str(fact.data["invoice_number"]).rsplit("-", 1)[0] for fact in facts]


def test_multi_hop_question_returns_the_unpaid_invoices_with_a_po(
    analyzer: QueryAnalyzer, expander: GraphExpander, ledger: dict[str, str]
) -> None:
    analysis = analyzer.analyze("Acme invoices whose PO has no payment?")

    facts = expander.facts(analysis)

    # INV-1 is paid in full, INV-4 has no PO, INV-9 is another vendor's
    assert short(facts) == ["INV-2", "INV-3"]
    assert {fact.source for fact in facts} == {"unpaid_with_po"}
    assert facts[0].text.endswith("-> no payment")
    assert facts[1].text.endswith("outstanding 230.00 USD")


def test_payment_seed_expands_to_the_invoice_it_settled(
    expander: GraphExpander, ledger: dict[str, str], tag: str
) -> None:
    facts = expander.expand_facts([f"doc-PAY-2-{tag}"])

    assert short(facts) == ["INV-3"]
    chain = f"-> {ledger['PO-1']} -> contract {ledger['CT']} (Net 30) -> paid 100.00 of 330.00"
    assert chain in facts[0].text
    assert facts[0].citations == {
        f"doc:{ledger['INV-3']}": f"doc-INV-3-{tag}",
        f"doc:{ledger['PO-1']}": f"doc-PO-1-{tag}",
        f"doc:{ledger['CT']}": f"doc-CT-{tag}",
        f"doc:{ledger['PAY-2']}": f"doc-PAY-2-{tag}",
    }


def test_contract_seed_reaches_invoices_two_hops_away(
    expander: GraphExpander, ledger: dict[str, str], tag: str
) -> None:
    rows = expander.expand([f"doc-CT-{tag}"])

    assert sorted((row["invoice_number"], row["hops"]) for row in rows) == [
        (ledger["INV-1"], 2),
        (ledger["INV-3"], 2),
    ]


def test_document_number_in_the_question_is_a_seed(
    analyzer: QueryAnalyzer, expander: GraphExpander, ledger: dict[str, str]
) -> None:
    analysis = analyzer.analyze(f"Which payment settled {ledger['INV-1']}?")

    facts = expander.facts(analysis)

    assert short(facts) == ["INV-1"]
    assert f"paid in full by {ledger['PAY-1']}" in facts[0].text


def test_price_question_compares_invoice_lines_with_the_contract_rate(
    analyzer: QueryAnalyzer, expander: GraphExpander, ledger: dict[str, str]
) -> None:
    analysis = analyzer.analyze(f"Does {ledger['INV-3']}'s unit price match the contract rate?")

    facts = expander.facts(analysis)

    assert [fact.source for fact in facts] == ["contract_for_po"]
    text = facts[0].text
    assert f"governed by contract {ledger['CT']}" in text
    assert "rate card: Pallet wrap 500m 30.00/roll" in text
    assert f"Invoice {ledger['INV-3']} (330.00 USD) lines: Pallet wrap 500m 10 x 33.00" in text
    assert "(contract rate 30.00, +10.0%)" in text


def test_vendor_ledger_totals(
    analyzer: QueryAnalyzer, expander: GraphExpander, ledger: dict[str, str]
) -> None:
    facts = expander.facts(analyzer.analyze("How much do we owe Acme in total?"))

    assert [fact.source for fact in facts] == ["vendor_ledger"]
    data = facts[0].data
    assert (data["invoices"], data["invoiced"], data["paid"]) == (4, 1230.0, 400.0)
    assert (data["open_invoices"], data["outstanding"], data["overdue"]) == (3, 830.0, 830.0)
