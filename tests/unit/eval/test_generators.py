"""The golden-set generators state the ground truth of a small, hand-built ledger; the
verifier, the extend mode and the review CLI do what they say."""

from collections.abc import Iterator, Mapping
from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from eval.golden_set.build_golden_set import (
    GENERATORS,
    Draft,
    Fresh,
    Ledger,
    check_grounding,
    extend,
    n_of,
    parse_targets,
    round_robin,
)
from eval.golden_set.review import Review, run_review
from eval.golden_set.schema import GoldenQuestion
from eval.golden_set.verify import KINDS, Verdict, verify_question
from ledgerlens.storage.postgres.models import (
    ContractRow,
    InvoiceLineItemRow,
    InvoiceRow,
    PaymentAllocationRow,
    PaymentRow,
    PurchaseOrderRow,
    VendorRow,
)

ACME, BOREALIS = "Acme Supplies Ltd", "Borealis Freight AB"
WRAP = "Pallet wrap"


def doc(n: int) -> str:
    return f"00000000-0000-4000-8000-{n:012d}"


def line(qty: int, price: str) -> InvoiceLineItemRow:
    return InvoiceLineItemRow(
        line_no=0,
        description=WRAP,
        quantity=Decimal(qty),
        unit_price=Decimal(price),
        amount=Decimal(qty) * Decimal(price),
    )


def invoice(
    n: int, vendor: str, po: str, day: date, due: date, total: str, *lines: InvoiceLineItemRow
) -> InvoiceRow:
    return InvoiceRow(
        id=f"I{n}",
        document_id=doc(n),
        vendor_id=vendor,
        vendor_name=ACME if vendor == "V1" else BOREALIS,
        invoice_number=f"INV-{n}",
        invoice_date=day,
        due_date=due,
        po_number=po,
        currency="GBP",
        total=Decimal(total),
        line_items=list(lines),
    )


def payment(n: int, vendor: str, amount: str, day: date, invoice_id: str) -> PaymentRow:
    return PaymentRow(
        id=f"P{n}",
        document_id=doc(100 + n),
        vendor_id=vendor,
        vendor_name=ACME if vendor == "V1" else BOREALIS,
        payment_reference=f"PAY-{n}",
        paid_at=day,
        currency="GBP",
        amount=Decimal(amount),
        allocations=[
            PaymentAllocationRow(
                invoice_number=f"INV-{invoice_id[1:]}",
                invoice_id=invoice_id,
                amount=Decimal(amount),
            )
        ],
    )


def po(n: int, vendor: str, total: str, day: date, contract: str | None = None) -> PurchaseOrderRow:
    return PurchaseOrderRow(
        id=f"PO{n}",
        document_id=doc(200 + n),
        vendor_id=vendor,
        vendor_name=ACME if vendor == "V1" else BOREALIS,
        po_number=f"PO-{n}",
        order_date=day,
        contract_number=contract,
        currency="GBP",
        total=Decimal(total),
        line_items=[],
    )


def build_ledger() -> Ledger:
    """Acme: INV-1 (PO-1, pallet wrap at 45.00 vs the contract's 40.00, unpaid), INV-2
    (PO-2, paid in full by PAY-1), INV-4 (PO-1, at the contract rate, unpaid). Borealis:
    INV-3 (PO-3, 100.00 of 300.00 paid by PAY-2)."""
    contract = ContractRow(
        id="CT1",
        document_id=doc(300),
        vendor_id="V1",
        vendor_name=ACME,
        contract_number="CT-V1",
        title="Framework supply agreement",
        start_date=date(2025, 1, 1),
        end_date=date(2026, 12, 31),
        payment_terms_days=30,
        currency="GBP",
        rate_card=[{"item": WRAP, "unit": "roll", "unit_price": "40.00"}],
    )
    invoices = [
        invoice(1, "V1", "PO-1", date(2025, 2, 10), date(2025, 3, 12), "90.00", line(2, "45.00")),
        invoice(2, "V1", "PO-2", date(2025, 3, 10), date(2025, 4, 9), "500.00"),
        invoice(3, "V2", "PO-3", date(2025, 4, 10), date(2025, 5, 10), "300.00"),
        invoice(4, "V1", "PO-1", date(2025, 5, 1), date(2025, 5, 31), "80.00", line(2, "40.00")),
    ]
    pos = [
        po(1, "V1", "170.00", date(2025, 2, 1), "CT-V1"),
        po(2, "V1", "500.00", date(2025, 3, 1)),
        po(3, "V2", "300.00", date(2025, 4, 1)),
    ]
    payments = [
        payment(1, "V1", "500.00", date(2025, 4, 1), "I2"),
        payment(2, "V2", "100.00", date(2025, 5, 5), "I3"),
    ]
    texts = {
        i.document_id: f"INVOICE {i.invoice_number} {i.vendor_name} Date {i.invoice_date} "
        f"Due {i.due_date} PO {i.po_number} Total {i.total:,.2f} {i.currency}"
        for i in invoices
        if i.document_id
    }
    texts |= {
        p.document_id: f"PURCHASE ORDER {p.po_number} To {p.vendor_name} Date {p.order_date} "
        f"Total {p.total:,.2f} GBP"
        for p in pos
        if p.document_id
    }
    texts |= {
        p.document_id: f"REMITTANCE {p.payment_reference} {p.vendor_name} Paid {p.paid_at} "
        f"Amount {p.amount:,.2f} GBP"
        for p in payments
        if p.document_id
    }
    texts[doc(300)] = (
        "CONTRACT CT-V1 Acme Supplies Ltd. This agreement runs from 2025-01-01 to 2026-12-31. "
        "Pallet wrap roll 40.00. Payment within 30 days of the invoice date."
    )
    return Ledger(
        vendors={
            "V1": VendorRow(id="V1", canonical_name=ACME, aliases=[]),
            "V2": VendorRow(id="V2", canonical_name=BOREALIS, aliases=[]),
        },
        invoices=invoices,
        purchase_orders={p.id: p for p in pos},
        payments=payments,
        contracts={"CT-V1": contract},
        texts=texts,
        labels={"I1": {"anomalies": ["price_inflation"]}},
    )


@pytest.fixture(scope="module")
def ledger() -> Ledger:
    return build_ledger()


def drafts(ledger: Ledger, category: str, n: int = 50, seed: int = 1) -> list[Draft]:
    import random

    return GENERATORS[category](ledger, random.Random(seed), n)


def by_kind(items: list[Draft], kind: str) -> list[Draft]:
    return [d for d in items if d.check.get("kind") == kind]


# --------------------------------------------------------------------------- every draft


@pytest.mark.parametrize("category", list(GENERATORS))
def test_drafts_are_well_formed_grounded_and_checkable(ledger: Ledger, category: str) -> None:
    made = drafts(ledger, category)
    assert made, category
    relational = category in ("multi_hop", "aggregation", "reconciliation")
    for d in made:
        assert d.requires_graph == relational, d.question
        assert d.check["kind"] in KINDS, d.check
        assert check_grounding(d, ledger.texts) == [], d.question
        question = d.to_question(1)
        assert question.check == d.check
    assert len({d.question for d in made}) == len(made)


# --------------------------------------------------------------------------- ground truth


def test_lookup_answers(ledger: Ledger) -> None:
    made = drafts(ledger, "lookup")
    totals = {d.check["invoice_id"]: d.expected["amount"] for d in by_kind(made, "invoice_total")}
    assert totals and all(
        amount == {"I1": "90.00", "I2": "500.00", "I3": "300.00", "I4": "80.00"}[invoice_id]
        for invoice_id, amount in totals.items()
    )
    for d in by_kind(made, "payment_amount"):
        assert (d.check["payment_id"], d.expected["vendor"]) in {("P1", ACME), ("P2", BOREALIS)}
    asked = [d.check.get("invoice_id") for d in made if "invoice_id" in d.check]
    assert len(asked) == len(set(asked))  # one question per record within a category


def test_multi_hop_answers(ledger: Ledger) -> None:
    made = drafts(ledger, "multi_hop")
    (settled,) = by_kind(made, "settled_by")
    assert settled.check["invoice_id"] == "I2"
    assert settled.expected == {"number": "PAY-1", "date": "2025-04-01"}
    (owed,) = by_kind(made, "owed")
    assert owed.check["invoice_id"] == "I3" and owed.expected["amount"] == "200.00"
    governed = by_kind(made, "governing_contract") + by_kind(made, "terms_via_contract")
    assert {d.check["invoice_id"] for d in governed} <= {"I1", "I4"}
    assert all(d.expected["number"] == "CT-V1" for d in governed)
    assert all(d.expected["days"] == 30 for d in by_kind(made, "terms_via_contract"))
    (open_po,) = by_kind(made, "open_with_po")
    assert open_po.check["vendor_id"] == "V1"
    assert open_po.expected["numbers"] == ["INV-1", "INV-4"]
    billed = by_kind(made, "po_invoices") + by_kind(made, "po_invoiced")
    for d in billed:
        truth = {
            "PO1": (["INV-1", "INV-4"], "170.00"),
            "PO2": (["INV-2"], "500.00"),
            "PO3": (["INV-3"], "300.00"),
        }[d.check["po_id"]]
        if "numbers" in d.expected:
            assert d.expected["numbers"] == truth[0]
        else:
            assert (d.expected["amount"], d.expected["count"]) == (truth[1], len(truth[0]))


def test_aggregation_answers(ledger: Ledger) -> None:
    made = drafts(ledger, "aggregation", seed=3)
    assert all(by_kind(made, k) for k in ("count_year", "balance", "paid_year", "top_vendor"))
    invoices_2025 = {"V1": 3, "V2": 1}
    for d in by_kind(made, "count_year"):
        expected = invoices_2025[d.check["vendor_id"]] if d.check["year"] == 2025 else 0
        assert d.expected["count"] == expected
    owed = {"V1": ("170.00", 2), "V2": ("200.00", 1)}
    for d in by_kind(made, "balance"):
        assert (d.expected["amount"], d.expected["count"]) == owed[d.check["vendor_id"]]
    for d in by_kind(made, "unpaid_count"):
        assert d.expected["count"] == owed[d.check["vendor_id"]][1]
    for d in by_kind(made, "paid_year"):
        assert d.check["year"] == 2025
        assert d.expected["amount"] == {"V1": "500.00", "V2": "100.00"}[d.check["vendor_id"]]
    for d in by_kind(made, "top_vendor"):
        assert d.check == {"kind": "top_vendor", "currency": "GBP", "year": 2025}
        assert (d.expected["vendor"], d.expected["amount"]) == (ACME, "670.00")


def test_reconciliation_answers(ledger: Ledger) -> None:
    made = drafts(ledger, "reconciliation")
    rates = {d.check["invoice_id"]: d.expected for d in by_kind(made, "rate_match")}
    assert rates["I1"] == {"match": False, "items": [WRAP]}
    assert rates.get("I4", {"match": True}) == {"match": True}
    paid = {d.check["invoice_id"]: d.expected for d in by_kind(made, "paid_in_full")}
    assert paid == {"I2": {"match": True}, "I3": {"match": False, "amount": "200.00"}}


def test_unanswerable_questions_name_nothing_that_exists(ledger: Ledger) -> None:
    made = drafts(ledger, "unanswerable", n=30)
    numbers = {i.invoice_number for i in ledger.invoices} | {"PO-1", "PO-2", "PO-3"}
    for d in made:
        assert d.documents == () and d.expected == {"answerable": False}
        assert d.check.get("number") not in numbers
    clauses = by_kind(made, "missing_clause")
    assert clauses and all(d.check["contract_id"] == "CT1" for d in clauses)
    text = ledger.texts[doc(300)].lower()
    assert all(not any(k in text for k in d.check["keywords"]) for d in clauses)


# --------------------------------------------------------------------------- verify


def question(**fields: Any) -> GoldenQuestion:
    values: dict[str, Any] = {
        "id": "q900",
        "category": "lookup",
        "question": "What is the total on invoice INV-1?",
        "ground_truth_answer": "90.00 GBP",
        "relevant_document_ids": [doc(1)],
        "requires_graph": False,
        "difficulty": "easy",
        "expected": {"amount": "90.00", "currency": "GBP"},
        "check": {"kind": "invoice_total", "invoice_id": "I1"},
    }
    return GoldenQuestion.model_validate(values | fields)


def graph(rows: list[dict[str, Any]]) -> Any:
    def run(query: str, params: Mapping[str, Any]) -> list[dict[str, Any]]:
        return rows

    return run


def no_text(contract_id: str) -> str:
    return ""


def test_verify_agrees_and_disagrees() -> None:
    row = [{"i": {"total": 90.0, "currency": "GBP"}}]
    assert verify_question(question(), graph(row), no_text).ok
    wrong = verify_question(
        question(expected={"amount": "91.00", "currency": "GBP"}), graph(row), no_text
    )
    assert not wrong.ok and "amount" in wrong.detail
    missing = verify_question(question(), graph([]), no_text)
    assert not missing.ok and "expected one row" in missing.detail
    unknown = verify_question(question(check={"kind": "astrology"}), graph(row), no_text)
    assert not unknown.ok


def test_verify_unanswerable_clause_by_text() -> None:
    q = question(
        category="unanswerable",
        question="What is the liability cap in contract CT-V1?",
        relevant_document_ids=[],
        expected={"answerable": False},
        check={"kind": "missing_clause", "contract_id": "CT1", "keywords": ["liabil"]},
    )
    assert verify_question(q, graph([]), lambda _: "Payment within 30 days.").ok
    assert not verify_question(q, graph([]), lambda _: "Liability is capped at 1x fees.").ok
    claimed = q.model_copy(update={"expected": {"answerable": True}})
    assert not verify_question(claimed, graph([]), lambda _: "Payment within 30 days.").ok


# --------------------------------------------------------------------------- extend


def test_extend_tops_up_to_targets_without_repeating(ledger: Ledger) -> None:
    existing = [
        d.to_question(n, reviewed=True) for n, d in enumerate(drafts(ledger, "lookup", 3), 1)
    ]
    new, rejected = extend(ledger, existing, {"lookup": 6, "multi_hop": 2}, seed=5)
    assert [q.id for q in new] == ["q004", "q005", "q006", "q007", "q008"]
    assert sum(q.category == "lookup" for q in new) == 3
    assert sum(q.category == "multi_hop" for q in new) == 2
    asked = {q.question for q in existing}
    assert not any(q.question in asked for q in new) and not any(q.reviewed for q in new)
    assert rejected == []


def test_helpers() -> None:
    assert parse_targets("lookup=25, clause=15") == {"lookup": 25, "clause": 15}
    with pytest.raises(Exception, match="category=count"):
        parse_targets("lookups=3")
    assert (n_of(1, "invoice"), n_of(2, "invoice"), n_of(0, "day")) == (
        "1 invoice",
        "2 invoices",
        "0 days",
    )

    def shape(tag: str, n: int) -> Iterator[Draft]:
        for k in range(n):
            yield Draft("lookup", f"{tag}{k}?", "a", (), False, "easy")

    mixed = round_robin(5, shape("a", 1), shape("b", 3), shape("c", 3))
    assert [d.question for d in mixed] == ["a0?", "b0?", "c0?", "b1?", "c1?"]
    fresh = Fresh()
    assert list(fresh([1, 2, 1], str)) == [1, 2] and list(fresh([2, 3], str)) == [3]


# --------------------------------------------------------------------------- review


def test_review_accepts_edits_rejects_and_resumes(ledger: Ledger) -> None:
    pending = [d.to_question(100 + n) for n, d in enumerate(drafts(ledger, "lookup", 4))]
    golden = [question(id="q007", reviewed=True)]
    answers = iter(["a", "e", "What does INV-2 total, tax included?", "", "x", "r", "no", "q"])
    shown: list[str] = []
    review = run_review(
        Review(golden),
        pending,
        ask=lambda _: next(answers),
        say=shown.append,
        verdicts={pending[0].id: Verdict(pending[0].id, "invoice_total", True)},
    )
    assert [q.id for q in review.golden] == ["q007", "q008", "q009"]
    assert all(q.reviewed for q in review.golden)
    assert review.golden[2].question == "What does INV-2 total, tax included?"
    assert "graph:    agrees" in shown[0]
    assert [row["decision"] for row in review.log] == ["accept", "accept", "reject"]
    assert review.pending(pending) == [pending[3]]  # quit before the fourth

    again = run_review(review, pending, ask=lambda _: "a")
    assert len(again.golden) == 4 and again.pending(pending) == []


def test_accept_verified_leaves_disagreements_for_a_person(ledger: Ledger) -> None:
    pending = [d.to_question(100 + n) for n, d in enumerate(drafts(ledger, "lookup", 2))]
    verdicts = {
        pending[0].id: Verdict(pending[0].id, "k", True),
        pending[1].id: Verdict(pending[1].id, "k", False, "amount differs"),
    }

    def never(_: str) -> str:
        raise AssertionError("asked a person")

    review = run_review(Review([]), pending, ask=never, verdicts=verdicts, accept_verified=True)
    assert [q.question for q in review.golden] == [pending[0].question]
    assert review.pending(pending) == [pending[1]]
