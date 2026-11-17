"""Graph expansion without Neo4j: rendering rows as facts, and choosing the template."""

from collections.abc import Mapping
from datetime import date
from typing import Any

import pytest

from ledgerlens.retrieval import RetrievalResult, ScoredChunk
from ledgerlens.retrieval.analyze import QueryAnalyzer
from ledgerlens.retrieval.graph import (
    GraphExpander,
    GraphQuery,
    native,
    plan_query,
    render_contract,
    render_invoice,
    render_ledger,
)
from ledgerlens.storage.neo4j.linker import VendorCandidate
from ledgerlens.storage.neo4j.queries import INVOICE_NEIGHBORHOOD, CypherTemplate


def invoice_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "invoice_id": "SINV-1",
        "invoice_number": "INV-889",
        "document_id": "doc-inv",
        "doc_type": "invoice",
        "invoice_date": date(2026, 3, 14),
        "due_date": date(2026, 4, 13),
        "total": 4200.0,
        "currency": "GBP",
        "vendor_id": "V001",
        "vendor": "Acme Supplies Ltd",
        "po": {"id": "PO-1042", "po_number": "PO-1042", "document_id": "doc-po"},
        "contract": {
            "id": "CT-V001",
            "contract_number": "CT-V001",
            "document_id": None,
            "payment_terms_days": 30,
        },
        "payments": [],
        "paid": 0.0,
        "outstanding": 4200.0,
        "hops": 0,
    }
    return row | overrides


class StubRunner:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows or []
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def run(self, template: CypherTemplate, params: Mapping[str, Any]) -> list[dict[str, Any]]:
        self.calls.append((template.name, dict(params)))
        return self.rows


# --------------------------------------------------------------------------- rendering


def test_unpaid_invoice_reads_as_a_chain() -> None:
    fact = render_invoice(invoice_row())

    assert fact.text == (
        "Invoice INV-889 (Acme Supplies Ltd, 4,200.00 GBP, dated 2026-03-14, due 2026-04-13)"
        " -> PO-1042 -> contract CT-V001 (Net 30) -> no payment"
    )
    # the contract was loaded without a document: nothing to cite
    assert fact.citations == {"doc:INV-889": "doc-inv", "doc:PO-1042": "doc-po"}
    assert fact.document_ids == ("doc-inv", "doc-po")
    assert fact.source == "invoice_neighborhood"


def test_partly_paid_invoice_shows_what_is_outstanding() -> None:
    payment = {
        "payment_reference": "PAY-7",
        "document_id": "doc-pay",
        "paid_at": date(2026, 4, 20),
        "amount": 1000.0,
    }
    fact = render_invoice(
        invoice_row(po=None, contract=None, payments=[payment], paid=1000.0, outstanding=3200.0)
    )

    assert fact.text.endswith(
        "-> no PO -> paid 1,000.00 of 4,200.00 by PAY-7 on 2026-04-20 (1,000.00); "
        "outstanding 3,200.00 GBP"
    )
    assert fact.citations["doc:PAY-7"] == "doc-pay"


def test_fully_paid_invoice() -> None:
    payment = {"payment_reference": "PAY-8", "paid_at": date(2026, 4, 1), "amount": 4200.0}
    fact = render_invoice(invoice_row(payments=[payment], paid=4200.0))

    assert fact.text.endswith("-> paid in full by PAY-8 on 2026-04-01 (4,200.00)")


def test_vendor_ledger_row() -> None:
    fact = render_ledger(
        {
            "vendor_id": "V001",
            "vendor": "Acme Supplies Ltd",
            "currency": "GBP",
            "invoices": 3,
            "invoiced": 300.0,
            "paid": 200.0,
            "open_invoices": 1,
            "outstanding": 100.0,
            "overdue": 0.0,
            "first_invoice": date(2026, 1, 1),
            "last_invoice": date(2026, 3, 1),
            "contracts": [{"contract_number": "CT-V001", "document_id": "doc-ct"}],
        }
    )

    assert fact.text == (
        "Vendor Acme Supplies Ltd (V001), GBP: 3 invoices dated 2026-01-01 to 2026-03-01; "
        "invoiced 300.00 GBP, paid 200.00 GBP, outstanding 100.00 GBP on 1 invoices "
        "(overdue 0.00 GBP); contract CT-V001"
    )
    assert fact.citations == {"doc:CT-V001": "doc-ct"}


def test_contract_row_compares_invoice_prices_with_the_rate_card() -> None:
    contract = {
        "contract_number": "CT-V002",
        "document_id": "doc-ct",
        "title": "Packaging",
        "start_date": date(2025, 1, 1),
        "end_date": None,
        "payment_terms_days": 30,
        "rate_items": ["Pallet wrap 500m", "Labels"],
        "rate_units": ["roll", ""],
        "rate_prices": [44.67, 20.0],
    }
    lines = [
        {"description": "Pallet wrap 500m", "quantity": 10.0, "unit_price": 44.67},
        {"description": "labels", "quantity": 2.0, "unit_price": 22.0},
        {"description": "Freight", "quantity": 1.0, "unit_price": 15.0},
    ]
    row = {
        "po_id": "PO-9",
        "po_number": "PO-9",
        "document_id": "doc-po",
        "order_date": date(2025, 2, 1),
        "total": 500.0,
        "currency": "GBP",
        "vendor": "Brightline",
        "contract": contract,
        "po_lines": [],
        "invoices": [
            {"invoice_number": "INV-5", "document_id": "doc-inv", "total": 500.0, "lines": lines}
        ],
    }

    fact = render_contract(row)

    assert 'governed by contract CT-V002 "Packaging" (2025-01-01 to ?, Net 30)' in fact.text
    assert "rate card: Pallet wrap 500m 44.67/roll; Labels 20.00" in fact.text
    assert "Pallet wrap 500m 10 x 44.67 (= contract rate)" in fact.text
    assert "labels 2 x 22.00 (contract rate 20.00, +10.0%)" in fact.text
    assert "Freight 1 x 15.00." not in fact.text and fact.text.endswith("Freight 1 x 15.00")
    assert set(fact.citations) == {"doc:PO-9", "doc:CT-V002", "doc:INV-5"}


def test_contract_row_without_a_contract() -> None:
    row = {"po_number": "PO-9", "currency": "GBP", "cited_contract": "CT-X", "contract": None}

    assert "cites contract CT-X, which is not in the graph" in render_contract(row).text


def test_fact_citations_flow_into_the_result() -> None:
    fact = render_invoice(invoice_row())
    chunk = ScoredChunk(id="d#c0", document_id="doc-other", text="x", score=1.0, doc_number="X-1")

    result = RetrievalResult.build("q", "r4", [chunk], graph_facts=[fact])

    assert result.citations == {
        "doc:X-1": "doc-other",
        "doc:INV-889": "doc-inv",
        "doc:PO-1042": "doc-po",
    }


# --------------------------------------------------------------------------- expansion


def test_native_converts_nested_neo4j_dates() -> None:
    class Neo4jDate:
        def to_native(self) -> date:
            return date(2026, 1, 2)

    assert native({"a": [Neo4jDate()], "b": {"c": Neo4jDate()}, "d": 1}) == {
        "a": [date(2026, 1, 2)],
        "b": {"c": date(2026, 1, 2)},
        "d": 1,
    }


def test_expand_dedupes_seeds_and_skips_the_query_without_any() -> None:
    runner = StubRunner([invoice_row()])
    expander = GraphExpander(runner=runner, per_seed=3, limit=9)

    assert expander.expand([None, ""]) == []
    facts = expander.expand_facts(["INV-889", "doc-a", "INV-889", None])

    assert [fact.text[:15] for fact in facts] == ["Invoice INV-889"]
    assert runner.calls == [
        ("invoice_neighborhood", {"seeds": ["INV-889", "doc-a"], "per_seed": 3, "limit": 9})
    ]


# --------------------------------------------------------------------------- template choice

TODAY = date(2026, 11, 17)
ACME = VendorCandidate("V001", "Acme Supplies Ltd")
ANALYZER = QueryAnalyzer([ACME, VendorCandidate("V002", "Brightline Packaging Ltd")], today=TODAY)


def plan(question: str, seeds: tuple[str, ...] = ("doc-a",)) -> GraphQuery | None:
    return plan_query(ANALYZER.analyze(question), seeds, today=TODAY, per_seed=4, limit=20)


@pytest.mark.parametrize(
    ("question", "template"),
    [
        ("Which payment settled INV-889?", "invoice_neighborhood"),
        ("Is INV-889 overdue?", "invoice_neighborhood"),
        ("What PO does INV-889 reference?", "invoice_neighborhood"),
        ("Acme invoices whose PO has no payment?", "unpaid_with_po"),
        ("Which Brightline invoices are still unpaid?", "unpaid_with_po"),
        ("Total unpaid by vendor for Q1?", "vendor_ledger"),
        ("How much have we spent with Acme this year?", "vendor_ledger"),
        ("Does INV-889's unit price match the contract rate?", "contract_for_po"),
        ("Which contract governs PO-1042?", "contract_for_po"),
        ("What are the payment terms under the Acme agreement?", "contract_for_po"),
    ],
)
def test_template_follows_the_question(question: str, template: str) -> None:
    query = plan(question)
    assert query is not None
    assert query.template.name == template


def test_question_numbers_lead_the_seeds() -> None:
    query = plan("Which payment settled INV-889?", seeds=("doc-a", "doc-b", "doc-a"))

    assert query == GraphQuery(
        INVOICE_NEIGHBORHOOD, {"seeds": ["INV-889", "doc-a", "doc-b"], "per_seed": 4, "limit": 20}
    )


def test_vendor_and_dates_scope_the_vendor_wide_templates() -> None:
    unpaid = plan("Acme invoices from March 2026 with no payment")
    ledger = plan("Total spend per vendor in Q1 2026")

    assert unpaid is not None and unpaid.params == {
        "vendor_ids": ["V001"],
        "date_from": "2026-03-01",
        "date_to": "2026-03-31",
        "limit": 20,
    }
    assert ledger is not None and ledger.params == {
        "vendor_ids": [],
        "date_from": "2026-01-01",
        "date_to": "2026-03-31",
        "as_of": "2026-11-17",
        "limit": 20,
    }


def test_nothing_to_start_from_means_no_query() -> None:
    assert plan("Tell me about the documents", seeds=()) is None
    assert plan_query(None, [], today=TODAY) is None
    assert plan_query(None, ["doc-a"], today=TODAY) is not None


def test_facts_runs_the_planned_template() -> None:
    runner = StubRunner([invoice_row()])
    expander = GraphExpander(runner=runner, today=TODAY)

    facts = expander.facts(ANALYZER.analyze("Acme invoices with no payment"))

    assert [fact.source for fact in facts] == ["unpaid_with_po"]
    assert runner.calls[0][0] == "unpaid_with_po"
    assert GraphExpander(runner=runner).facts(None, []) == []
