"""Draft golden-set questions from the synthetic ledger, where every answer is known.

    python -m eval.golden_set.build_golden_set --out data/golden/golden_set.draft.jsonl

Answers are computed from the stores' own view of the ledger (Postgres: records,
payment allocations, which record came from which document), with the anomaly labels
from the generator's ``data/synthetic/records.jsonl``. Only records rendered as documents
can be evidence, so every question is about rendered documents, and each relevant
document id is a real ``documents.id``.

Each category mixes several question **shapes** (one draft of each in turn, no record
asked about twice within a category):

* **lookup** -- an invoice's total, due date, date or PO; a PO's total or supplier; a
  remittance's amount and vendor, or its date.
* **clause** -- a contract's payment terms, start or end date, a rate on its rate card, the
  unit an item is priced in.
* **multi_hop** -- which payment settled an invoice; which contract governs an invoice's
  PO, and the payment days it allows; what is still owed on a partly paid invoice; a
  vendor's invoices with a PO and no payment; which invoices billed a PO, and how much.
* **aggregation** -- invoice counts per vendor and year or quarter; totals invoiced per
  vendor and quarter; outstanding balances and unpaid counts; payments per vendor and
  year; POs per vendor and year; the top vendor by spend in a currency.
* **reconciliation** -- invoice prices against the contract rate (inflated and clean);
  exact duplicates and near-duplicate resubmissions; payment terms against the contract
  (rushed and normal); whether an invoice is paid in full (yes and no).
* **unanswerable** -- invoices, POs, remittances and vendors that do not exist, and
  clauses a contract does not have (termination notice, late-payment interest, liability
  cap, governing law, renewal: checked absent from the contract's text); the answer is
  that the records do not say.

Every draft is then **checked against the evidence** (:func:`check_grounding`): amounts,
numbers and dates a single-document answer states must appear in that document's text.
A draft that fails is dropped and reported. Drafts are written with ``reviewed: false``;
reviewing them is a separate, manual step before they join the golden set.

Every draft also carries a ``check``: the record ids and the kind of question, from
which ``eval/golden_set/verify.py`` derives the answer a second time from Neo4j.

Sampling is seeded (``--seed``), so the same stores give the same drafts.

To grow an existing set, draft only what it is missing (numbered after its last id, never
repeating one of its questions), then review the drafts into it::

    python -m eval.golden_set.build_golden_set --extend data/golden/golden_set.jsonl
    python -m eval.golden_set.review data/golden/golden_set.draft.jsonl --verify

(``--extend`` writes ``--out``, by default ``data/golden/golden_set.draft.jsonl``; pass
``--seed 63`` to get Day 63's drafts again.)
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from eval.golden_set.schema import (
    GoldenQuestion,
    dump_golden_set,
    load_golden_set,
    problems,
    summary,
)
from ledgerlens.extraction.normalize import find_dates
from ledgerlens.storage.postgres.db import get_session_factory
from ledgerlens.storage.postgres.models import (
    ChunkRow,
    ContractRow,
    InvoiceRow,
    PaymentRow,
    PurchaseOrderRow,
    VendorRow,
)

RECORDS = Path("data/synthetic/records.jsonl")
DEFAULT_COUNTS = {
    "lookup": 12,
    "clause": 8,
    "multi_hop": 12,
    "aggregation": 10,
    "reconciliation": 6,
    "unanswerable": 4,
}
SYNTHETIC_TARGETS = {
    "lookup": 25,
    "clause": 15,
    "multi_hop": 45,
    "aggregation": 30,
    "reconciliation": 20,
    "unanswerable": 20,
}
"""The synthetic part of the ~200-question set (docs/layers/09-evaluation.md): hand-written
SROIE lookups and CUAD clauses make up the rest of those two categories."""
TOLERANCE = Decimal("0.01")
MAX_COUNTED = 25
"""Aggregations over more records than this are skipped: their evidence lists get too long
to score retrieval against."""
T = TypeVar("T")


def money(amount: Decimal | float) -> str:
    return f"{Decimal(str(amount)):,.2f}"


def n_of(n: int, noun: str, plural: str | None = None) -> str:
    """``n_of(1, "invoice")`` -> ``1 invoice``; ``n_of(3, "invoice")`` -> ``3 invoices``."""
    return f"{n} {noun if n == 1 else plural or noun + 's'}"


def long_date(day: date) -> str:
    return f"{day.day} {day.strftime('%B %Y')}"


# --------------------------------------------------------------------------- the ledger


@dataclass
class Ledger:
    """The store's records, indexed for question drafting."""

    vendors: dict[str, VendorRow]
    invoices: list[InvoiceRow]
    purchase_orders: dict[str, PurchaseOrderRow]  # by id
    payments: list[PaymentRow]
    contracts: dict[str, ContractRow]  # by contract number
    texts: dict[str, str]  # document id -> its full text (chunks joined)
    labels: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Generator labels by record id: anomalies, duplicate_of, details, payment_status."""

    def __post_init__(self) -> None:
        self.invoice_by_id = {i.id: i for i in self.invoices}
        self.number_count = Counter(i.invoice_number for i in self.invoices)
        self.po_by_number: dict[str, PurchaseOrderRow] = {}
        for po in self.purchase_orders.values():
            self.po_by_number.setdefault(po.po_number, po)
        by_vendor_number: dict[tuple[str | None, str], list[str]] = defaultdict(list)
        for invoice in self.invoices:
            by_vendor_number[(invoice.vendor_id, invoice.invoice_number)].append(invoice.id)
        self.paid: dict[str, Decimal] = defaultdict(Decimal)  # invoice id -> allocated
        self.paid_by: dict[str, list[PaymentRow]] = defaultdict(list)
        for payment in self.payments:
            for allocation in payment.allocations:
                # Postgres leaves some allocations unlinked; the graph links them by the
                # vendor's invoice number when that is unambiguous, and so do we, so the
                # answers match what both stores say.
                matches = by_vendor_number[(payment.vendor_id, allocation.invoice_number)]
                invoice_id = allocation.invoice_id or (matches[0] if len(matches) == 1 else None)
                if invoice_id:
                    self.paid[invoice_id] += allocation.amount
                    self.paid_by[invoice_id].append(payment)

    def unique_number(self, invoice: InvoiceRow) -> bool:
        """The invoice's number belongs to it alone (duplicates share numbers)."""
        return self.number_count[invoice.invoice_number] == 1

    def outstanding(self, invoice: InvoiceRow) -> Decimal:
        return invoice.total - self.paid[invoice.id]

    def vendor_name(self, vendor_id: str | None) -> str:
        vendor = self.vendors.get(vendor_id or "")
        return vendor.canonical_name if vendor else "unknown vendor"

    def contract_of(self, invoice: InvoiceRow) -> ContractRow | None:
        po = self.po_by_number.get(invoice.po_number or "")
        if po is None or po.contract_number is None:
            return None
        contract = self.contracts.get(po.contract_number)
        return contract if contract and contract.vendor_id == po.vendor_id else None


def load_ledger(session: Session, records: Path = RECORDS) -> Ledger:
    texts: dict[str, list[str]] = defaultdict(list)
    for chunk in session.scalars(
        select(ChunkRow).order_by(ChunkRow.document_id, ChunkRow.chunk_index)
    ):
        texts[chunk.document_id].append(chunk.text)
    labels: dict[str, dict[str, Any]] = {}
    if records.exists():
        for line in records.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["record_type"] == "invoice":
                labels[record["id"]] = record
    return Ledger(
        vendors={v.id: v for v in session.scalars(select(VendorRow))},
        invoices=list(
            session.scalars(
                select(InvoiceRow)
                .options(selectinload(InvoiceRow.line_items))
                .order_by(InvoiceRow.id)
            )
        ),
        purchase_orders={
            po.id: po
            for po in session.scalars(
                select(PurchaseOrderRow).options(selectinload(PurchaseOrderRow.line_items))
            )
        },
        payments=list(
            session.scalars(
                select(PaymentRow)
                .options(selectinload(PaymentRow.allocations))
                .order_by(PaymentRow.id)
            )
        ),
        contracts={
            c.contract_number: c
            for c in session.scalars(select(ContractRow).order_by(ContractRow.id))
            if c.contract_number
        },
        texts={doc: "\n".join(parts) for doc, parts in texts.items()},
        labels=labels,
    )


# --------------------------------------------------------------------------- drafts


@dataclass(frozen=True)
class Draft:
    """A question before it gets an id."""

    category: str
    question: str
    answer: str
    documents: tuple[str, ...]
    requires_graph: bool
    difficulty: str
    expected: dict[str, Any] = field(default_factory=dict)
    grounded_in: str | None = None
    """For single-document answers: the document whose text must contain ``expected``."""
    check: dict[str, Any] = field(default_factory=dict)
    """How ``verify.py`` derives the answer again from the graph (``kind`` + record ids)."""

    def to_question(self, number: int, *, reviewed: bool = False) -> GoldenQuestion:
        return GoldenQuestion(
            id=f"q{number:03d}",
            category=self.category,  # type: ignore[arg-type]
            question=self.question,
            ground_truth_answer=self.answer,
            relevant_document_ids=self.documents,
            requires_graph=self.requires_graph,
            difficulty=self.difficulty,  # type: ignore[arg-type]
            expected=self.expected,
            reviewed=reviewed,
            check=self.check or None,
        )


def _docs(*ids: str | None) -> tuple[str, ...]:
    return tuple(dict.fromkeys(i for i in ids if i))


def _shuffled(rng: random.Random, items: Iterable[T]) -> list[T]:
    out = list(items)
    rng.shuffle(out)
    return out


Shape = Iterator[Draft]


def round_robin(n: int, *shapes: Shape) -> list[Draft]:
    """Up to ``n`` drafts, one from each shape in turn, so a category mixes its shapes."""
    out: list[Draft] = []
    active = list(shapes)
    while active and len(out) < n:
        for shape in list(active):
            draft = next(shape, None)
            if draft is None:
                active.remove(shape)
                continue
            out.append(draft)
            if len(out) == n:
                break
    return out


class Fresh:
    """Hands out each record once within a category, so two shapes never ask about the
    same invoice."""

    def __init__(self) -> None:
        self.used: set[str] = set()

    def __call__(self, items: Iterable[T], key: Callable[[T], str]) -> Iterator[T]:
        for item in items:
            k = key(item)
            if k not in self.used:
                self.used.add(k)
                yield item


# --------------------------------------------------------------------------- lookup


def lookup(ledger: Ledger, rng: random.Random, n: int) -> list[Draft]:
    invoices = _shuffled(
        rng, (i for i in ledger.invoices if i.document_id and ledger.unique_number(i))
    )
    pos = _shuffled(rng, (p for p in ledger.purchase_orders.values() if p.document_id))
    payments = _shuffled(rng, (p for p in ledger.payments if p.document_id))
    fresh = Fresh()

    def inv_key(i: InvoiceRow) -> str:
        return i.id

    def invoice_total() -> Shape:
        for i in fresh(invoices, inv_key):
            yield Draft(
                "lookup",
                f"What is the total on invoice {i.invoice_number}?",
                f"Invoice {i.invoice_number} from {ledger.vendor_name(i.vendor_id)} totals "
                f"{money(i.total)} {i.currency}.",
                _docs(i.document_id),
                False,
                "easy",
                {"amount": money(i.total), "currency": i.currency},
                i.document_id,
                {"kind": "invoice_total", "invoice_id": i.id},
            )

    def invoice_due() -> Shape:
        for i in fresh((i for i in invoices if i.due_date), inv_key):
            assert i.due_date is not None
            yield Draft(
                "lookup",
                f"When is invoice {i.invoice_number} due?",
                f"Invoice {i.invoice_number} is due on {i.due_date.isoformat()}.",
                _docs(i.document_id),
                False,
                "easy",
                {"date": i.due_date.isoformat()},
                i.document_id,
                {"kind": "invoice_due", "invoice_id": i.id},
            )

    def invoice_po() -> Shape:
        for i in fresh((i for i in invoices if i.po_number), inv_key):
            yield Draft(
                "lookup",
                f"Which purchase order does invoice {i.invoice_number} reference?",
                f"Invoice {i.invoice_number} references purchase order {i.po_number}.",
                _docs(i.document_id),
                False,
                "easy",
                {"number": i.po_number},
                i.document_id,
                {"kind": "invoice_po", "invoice_id": i.id},
            )

    def invoice_date() -> Shape:
        for i in fresh(invoices, inv_key):
            yield Draft(
                "lookup",
                f"What is the date of invoice {i.invoice_number}?",
                f"Invoice {i.invoice_number} is dated {i.invoice_date.isoformat()}.",
                _docs(i.document_id),
                False,
                "easy",
                {"date": i.invoice_date.isoformat()},
                i.document_id,
                {"kind": "invoice_date", "invoice_id": i.id},
            )

    def po_total() -> Shape:
        for p in fresh(pos, lambda p: p.id):
            yield Draft(
                "lookup",
                f"What is the total of purchase order {p.po_number}?",
                f"Purchase order {p.po_number} to {ledger.vendor_name(p.vendor_id)} totals "
                f"{money(p.total)} {p.currency}.",
                _docs(p.document_id),
                False,
                "easy",
                {"amount": money(p.total), "currency": p.currency},
                p.document_id,
                {"kind": "po_total", "po_id": p.id},
            )

    def po_vendor() -> Shape:
        for p in fresh(pos, lambda p: p.id):
            vendor = ledger.vendor_name(p.vendor_id)
            yield Draft(
                "lookup",
                f"Which supplier is purchase order {p.po_number} addressed to?",
                f"Purchase order {p.po_number} is addressed to {vendor}.",
                _docs(p.document_id),
                False,
                "easy",
                {"vendor": vendor},
                p.document_id,
                {"kind": "po_vendor", "po_id": p.id},
            )

    def payment_amount() -> Shape:
        for pay in fresh(payments, lambda p: p.id):
            vendor = ledger.vendor_name(pay.vendor_id)
            yield Draft(
                "lookup",
                f"How much was paid in remittance {pay.payment_reference}, and to which vendor?",
                f"Remittance {pay.payment_reference} paid {money(pay.amount)} {pay.currency} to "
                f"{vendor} on {pay.paid_at.isoformat()}.",
                _docs(pay.document_id),
                False,
                "easy",
                {"amount": money(pay.amount), "currency": pay.currency, "vendor": vendor},
                pay.document_id,
                {"kind": "payment_amount", "payment_id": pay.id},
            )

    def payment_date() -> Shape:
        for pay in fresh(payments, lambda p: p.id):
            yield Draft(
                "lookup",
                f"On what date was remittance {pay.payment_reference} paid?",
                f"Remittance {pay.payment_reference} was paid on {pay.paid_at.isoformat()}.",
                _docs(pay.document_id),
                False,
                "easy",
                {"date": pay.paid_at.isoformat()},
                pay.document_id,
                {"kind": "payment_date", "payment_id": pay.id},
            )

    return round_robin(
        n,
        invoice_total(),
        invoice_due(),
        invoice_po(),
        po_total(),
        payment_amount(),
        invoice_date(),
        po_vendor(),
        payment_date(),
    )


# --------------------------------------------------------------------------- clause


def clause(ledger: Ledger, rng: random.Random, n: int) -> list[Draft]:
    contracts = _shuffled(rng, (c for c in ledger.contracts.values() if c.document_id))
    fresh = Fresh()

    def key(c: ContractRow) -> str:
        return c.id

    def terms() -> Shape:
        for c in fresh(contracts, key):
            vendor = ledger.vendor_name(c.vendor_id)
            yield Draft(
                "clause",
                f"Within how many days must invoices be paid under our contract with {vendor}?",
                f"Contract {c.contract_number} requires payment within {c.payment_terms_days} "
                "days of the invoice date.",
                _docs(c.document_id),
                False,
                "easy",
                {"days": c.payment_terms_days},
                c.document_id,
                {"kind": "contract_terms", "contract_id": c.id},
            )

    def rate() -> Shape:
        for c in fresh((c for c in contracts if c.rate_card), key):
            entry = rng.choice(c.rate_card)
            price = money(Decimal(entry["unit_price"]))
            unit = f" per {entry['unit']}" if entry.get("unit") else ""
            yield Draft(
                "clause",
                f"What rate does the {ledger.vendor_name(c.vendor_id)} contract set for "
                f"{entry['item']}?",
                f"Contract {c.contract_number} sets {entry['item']} at {price} {c.currency}{unit}.",
                _docs(c.document_id),
                False,
                "medium",
                {"amount": price, "currency": c.currency},
                c.document_id,
                {"kind": "contract_rate", "contract_id": c.id, "item": entry["item"]},
            )

    def end() -> Shape:
        for c in fresh((c for c in contracts if c.end_date), key):
            assert c.end_date is not None
            yield Draft(
                "clause",
                f"When does contract {c.contract_number} expire?",
                f"Contract {c.contract_number} with {ledger.vendor_name(c.vendor_id)} runs "
                f"until {c.end_date.isoformat()}.",
                _docs(c.document_id),
                False,
                "easy",
                {"date": c.end_date.isoformat()},
                c.document_id,
                {"kind": "contract_end", "contract_id": c.id},
            )

    def start() -> Shape:
        for c in fresh(contracts, key):
            yield Draft(
                "clause",
                f"From what date does our agreement with {ledger.vendor_name(c.vendor_id)} run?",
                f"Contract {c.contract_number} runs from {c.start_date.isoformat()}.",
                _docs(c.document_id),
                False,
                "easy",
                {"date": c.start_date.isoformat()},
                c.document_id,
                {"kind": "contract_start", "contract_id": c.id},
            )

    def unit() -> Shape:
        for c in fresh((c for c in contracts if any(e.get("unit") for e in c.rate_card)), key):
            entry = rng.choice([e for e in c.rate_card if e.get("unit")])
            yield Draft(
                "clause",
                f"In what unit does contract {c.contract_number} price {entry['item']}?",
                f"Contract {c.contract_number} prices {entry['item']} per {entry['unit']}.",
                _docs(c.document_id),
                False,
                "medium",
                {"items": [entry["unit"]]},
                c.document_id,
                {"kind": "contract_unit", "contract_id": c.id, "item": entry["item"]},
            )

    return round_robin(n, terms(), rate(), end(), start(), unit())


# --------------------------------------------------------------------------- multi-hop


def multi_hop(ledger: Ledger, rng: random.Random, n: int) -> list[Draft]:
    fresh = Fresh()

    def rendered(invoice: InvoiceRow) -> bool:
        return bool(invoice.document_id) and ledger.unique_number(invoice)

    def inv_key(i: InvoiceRow) -> str:
        return i.id

    def settled_by() -> Shape:
        """Fully paid by exactly one rendered remittance."""
        settled = (
            i
            for i in _shuffled(rng, ledger.invoices)
            if rendered(i)
            and len(ledger.paid_by[i.id]) == 1
            and ledger.paid_by[i.id][0].document_id
            and ledger.outstanding(i) < TOLERANCE
        )
        for i in fresh(settled, inv_key):
            pay = ledger.paid_by[i.id][0]
            yield Draft(
                "multi_hop",
                f"Which payment settled invoice {i.invoice_number}?",
                f"Invoice {i.invoice_number} was settled by remittance {pay.payment_reference} "
                f"on {pay.paid_at.isoformat()}.",
                _docs(i.document_id, pay.document_id),
                True,
                "medium",
                {"number": pay.payment_reference, "date": pay.paid_at.isoformat()},
                check={"kind": "settled_by", "invoice_id": i.id},
            )

    def governing() -> list[tuple[InvoiceRow, ContractRow]]:
        return [
            (i, c)
            for i in _shuffled(rng, ledger.invoices)
            if rendered(i) and (c := ledger.contract_of(i)) is not None and c.document_id
        ]

    def governing_contract() -> Shape:
        for i, c in fresh(governing(), lambda pair: pair[0].id):
            po = ledger.po_by_number[i.po_number or ""]
            yield Draft(
                "multi_hop",
                f"Which contract governs the purchase order on invoice {i.invoice_number}?",
                f"Invoice {i.invoice_number} references {po.po_number}, which is governed by "
                f"contract {c.contract_number} ({c.title}).",
                _docs(i.document_id, c.document_id, po.document_id),
                True,
                "medium",
                {"number": c.contract_number, "po_number": po.po_number},
                check={"kind": "governing_contract", "invoice_id": i.id},
            )

    def terms_via_contract() -> Shape:
        for i, c in fresh(governing(), lambda pair: pair[0].id):
            po = ledger.po_by_number[i.po_number or ""]
            yield Draft(
                "multi_hop",
                f"How many days does the contract behind invoice {i.invoice_number} allow "
                "for payment?",
                f"Invoice {i.invoice_number} bills purchase order {po.po_number}, governed by "
                f"contract {c.contract_number}, which allows {c.payment_terms_days} days.",
                _docs(i.document_id, c.document_id, po.document_id),
                True,
                "hard",
                {"days": c.payment_terms_days, "number": c.contract_number},
                check={"kind": "terms_via_contract", "invoice_id": i.id},
            )

    def owed() -> Shape:
        """What is still owed on a partly paid invoice (the synthetic remittances each pay
        one invoice, so "which invoices did remittance X pay" has no interesting answer)."""
        partial = (
            i
            for i in _shuffled(rng, ledger.invoices)
            if rendered(i) and ledger.paid_by[i.id] and ledger.outstanding(i) >= TOLERANCE
        )
        for i in fresh(partial, inv_key):
            payments = ledger.paid_by[i.id]
            paid = ", ".join(
                f"{p.payment_reference} ({money(a.amount)} {i.currency})"
                for p in payments
                for a in p.allocations
                if a.invoice_id == i.id
            )
            yield Draft(
                "multi_hop",
                f"How much is still owed on invoice {i.invoice_number}?",
                f"{money(ledger.outstanding(i))} {i.currency}: invoice {i.invoice_number} totals "
                f"{money(i.total)} {i.currency} and was partly paid by {paid}.",
                _docs(i.document_id, *(p.document_id for p in payments)),
                True,
                "hard",
                {"amount": money(ledger.outstanding(i)), "currency": i.currency},
                check={"kind": "owed", "invoice_id": i.id},
            )

    def open_with_po() -> Shape:
        """A vendor's invoices with a PO and no payment at all (no shared numbers:
        duplicates would list one number twice)."""
        by_vendor: dict[str, list[InvoiceRow]] = defaultdict(list)
        for i in ledger.invoices:
            if i.vendor_id and i.po_number and not ledger.paid_by[i.id]:
                by_vendor[i.vendor_id].append(i)
        candidates = [
            (vendor_id, items)
            for vendor_id, items in sorted(by_vendor.items())
            if 1 <= len(items) <= 6
            and any(i.document_id for i in items)
            and all(ledger.unique_number(i) for i in items)
        ]
        for vendor_id, items in _shuffled(rng, candidates):
            items = sorted(items, key=lambda i: i.invoice_date)
            listed = ", ".join(
                f"{i.invoice_number} ({money(i.total)} {i.currency}, PO {i.po_number})"
                for i in items
            )
            yield Draft(
                "multi_hop",
                f"Which {ledger.vendor_name(vendor_id)} invoices reference a purchase order but "
                "have no payment?",
                f"{len(items)} invoice(s): {listed}.",
                _docs(*(i.document_id for i in items)),
                True,
                "hard",
                {"numbers": [i.invoice_number for i in items]},
                check={"kind": "open_with_po", "vendor_id": vendor_id},
            )

    def billed_against(po: PurchaseOrderRow) -> list[InvoiceRow]:
        return sorted(
            (
                i
                for i in ledger.invoices
                if i.po_number == po.po_number and i.vendor_id == po.vendor_id
            ),
            key=lambda i: (i.invoice_date, i.invoice_number),
        )

    def po_candidates() -> list[tuple[PurchaseOrderRow, list[InvoiceRow]]]:
        out = []
        for po in _shuffled(rng, sorted(ledger.purchase_orders.values(), key=lambda p: p.id)):
            items = billed_against(po)
            if (
                po.document_id
                and ledger.po_by_number.get(po.po_number) is po
                and items
                and all(ledger.unique_number(i) for i in items)
                and len({i.currency for i in items}) == 1
            ):
                out.append((po, items))
        return out

    candidates = po_candidates()
    half = len(candidates) // 2

    def po_invoices() -> Shape:
        for po, items in candidates[:half]:
            listed = ", ".join(f"{i.invoice_number} ({money(i.total)} {i.currency})" for i in items)
            yield Draft(
                "multi_hop",
                f"Which invoices were billed against purchase order {po.po_number}?",
                f"{n_of(len(items), 'invoice')} {'references' if len(items) == 1 else 'reference'} "
                f"{po.po_number}: {listed}.",
                _docs(po.document_id, *(i.document_id for i in items)),
                True,
                "medium",
                {"numbers": [i.invoice_number for i in items]},
                check={"kind": "po_invoices", "po_id": po.id},
            )

    def po_invoiced() -> Shape:
        for po, items in candidates[half:]:
            total = sum((i.total for i in items), Decimal(0))
            yield Draft(
                "multi_hop",
                f"How much has been invoiced against purchase order {po.po_number}?",
                f"{money(total)} {items[0].currency} across {n_of(len(items), 'invoice')}, "
                f"against a PO total of {money(po.total)} {po.currency}.",
                _docs(po.document_id, *(i.document_id for i in items)),
                True,
                "hard",
                {"amount": money(total), "count": len(items)},
                check={"kind": "po_invoiced", "po_id": po.id},
            )

    return round_robin(
        n,
        settled_by(),
        governing_contract(),
        owed(),
        open_with_po(),
        po_invoices(),
        terms_via_contract(),
        po_invoiced(),
    )


# --------------------------------------------------------------------------- aggregation


def _period(rng: random.Random) -> tuple[str, date, date]:
    year = rng.choice([2025, 2026])
    quarter = rng.choice([1, 2, 3] if year == 2026 else [1, 2, 3, 4])
    start = date(year, 3 * quarter - 2, 1)
    after = date(year + (quarter == 4), (3 * quarter) % 12 + 1, 1)
    return f"Q{quarter} {year}", start, date.fromordinal(after.toordinal() - 1)


def aggregation(ledger: Ledger, rng: random.Random, n: int) -> list[Draft]:
    by_vendor: dict[str, list[InvoiceRow]] = defaultdict(list)
    for i in ledger.invoices:
        if i.vendor_id in ledger.vendors:
            by_vendor[i.vendor_id].append(i)
    vendors = sorted(v for v, items in by_vendor.items() if any(i.document_id for i in items))
    pos_by_vendor: dict[str, list[PurchaseOrderRow]] = defaultdict(list)
    for po in ledger.purchase_orders.values():
        pos_by_vendor[po.vendor_id or ""].append(po)
    pays_by_vendor: dict[str, list[PaymentRow]] = defaultdict(list)
    for pay in ledger.payments:
        pays_by_vendor[pay.vendor_id or ""].append(pay)
    asked: set[tuple[str, str]] = set()  # (vendor, shape): one question of a shape per vendor

    def docs_of(items: Iterable[InvoiceRow | PurchaseOrderRow | PaymentRow]) -> tuple[str, ...]:
        def when(item: InvoiceRow | PurchaseOrderRow | PaymentRow) -> date:
            if isinstance(item, InvoiceRow):
                return item.invoice_date
            return item.order_date if isinstance(item, PurchaseOrderRow) else item.paid_at

        return _docs(*(i.document_id for i in sorted(items, key=when)))

    def each_vendor(shape: str) -> Iterator[str]:
        for vendor_id in _shuffled(rng, vendors):
            if (vendor_id, shape) not in asked:
                asked.add((vendor_id, shape))
                yield vendor_id

    def count_year() -> Shape:
        for vendor_id in each_vendor("count_year"):
            year = rng.choice([2025, 2026])
            items = [i for i in by_vendor[vendor_id] if i.invoice_date.year == year]
            vendor = ledger.vendor_name(vendor_id)
            if docs_of(items) and len(items) <= MAX_COUNTED:
                yield Draft(
                    "aggregation",
                    f"How many invoices did {vendor} issue in {year}?",
                    f"{vendor} issued {n_of(len(items), 'invoice')} dated in {year}.",
                    docs_of(items),
                    True,
                    "medium",
                    {"count": len(items)},
                    check={"kind": "count_year", "vendor_id": vendor_id, "year": year},
                )

    def count_quarter() -> Shape:
        for vendor_id in each_vendor("count_quarter"):
            label, start, end = _period(rng)
            items = [i for i in by_vendor[vendor_id] if start <= i.invoice_date <= end]
            vendor = ledger.vendor_name(vendor_id)
            if docs_of(items) and len(items) <= MAX_COUNTED:
                yield Draft(
                    "aggregation",
                    f"How many invoices did we receive from {vendor} in {label}?",
                    f"{n_of(len(items), vendor + ' invoice')} "
                    f"{'is' if len(items) == 1 else 'are'} dated between {start.isoformat()} and "
                    f"{end.isoformat()}.",
                    docs_of(items),
                    True,
                    "medium",
                    {"count": len(items)},
                    check={
                        "kind": "count_period",
                        "vendor_id": vendor_id,
                        "start": start.isoformat(),
                        "end": end.isoformat(),
                    },
                )

    def total_quarter() -> Shape:
        for vendor_id in each_vendor("total_quarter"):
            label, start, end = _period(rng)
            items = [i for i in by_vendor[vendor_id] if start <= i.invoice_date <= end]
            currencies = {i.currency for i in items}
            vendor = ledger.vendor_name(vendor_id)
            if docs_of(items) and len(currencies) == 1 and len(items) <= MAX_COUNTED:
                total = sum((i.total for i in items), Decimal(0))
                yield Draft(
                    "aggregation",
                    f"What was the total invoiced by {vendor} in {label}?",
                    f"{vendor} invoiced {money(total)} {currencies.pop()} across "
                    f"{n_of(len(items), 'invoice')} dated {start.isoformat()} to "
                    f"{end.isoformat()}.",
                    docs_of(items),
                    True,
                    "hard",
                    {"amount": money(total), "count": len(items)},
                    check={
                        "kind": "total_period",
                        "vendor_id": vendor_id,
                        "start": start.isoformat(),
                        "end": end.isoformat(),
                    },
                )

    def open_items(vendor_id: str) -> list[InvoiceRow] | None:
        """Not fully paid; ``None`` when shared numbers make it ambiguous (Ledger.paid)."""
        items = [i for i in by_vendor[vendor_id] if ledger.outstanding(i) >= TOLERANCE]
        return items if all(ledger.unique_number(i) for i in items) else None

    def balance() -> Shape:
        for vendor_id in each_vendor("balance"):
            items = open_items(vendor_id)
            if items is None:
                continue
            currencies = {i.currency for i in items}
            vendor = ledger.vendor_name(vendor_id)
            if docs_of(items) and len(currencies) == 1 and len(items) <= 15:
                amount = sum((ledger.outstanding(i) for i in items), Decimal(0))
                yield Draft(
                    "aggregation",
                    f"What is our total outstanding balance with {vendor}?",
                    f"{money(amount)} {currencies.pop()} is outstanding on "
                    f"{n_of(len(items), vendor + ' invoice')}.",
                    docs_of(items),
                    True,
                    "hard",
                    {"amount": money(amount), "count": len(items)},
                    check={"kind": "balance", "vendor_id": vendor_id},
                )

    def unpaid_count() -> Shape:
        for vendor_id in each_vendor("unpaid_count"):
            items = open_items(vendor_id)
            vendor = ledger.vendor_name(vendor_id)
            if items and docs_of(items) and len(items) <= 25:
                yield Draft(
                    "aggregation",
                    f"How many {vendor} invoices are not yet paid in full?",
                    f"{n_of(len(items), vendor + ' invoice')} "
                    f"{'still has' if len(items) == 1 else 'still have'} an amount outstanding.",
                    docs_of(items),
                    True,
                    "medium",
                    {"count": len(items)},
                    check={"kind": "unpaid_count", "vendor_id": vendor_id},
                )

    def paid_year() -> Shape:
        for vendor_id in each_vendor("paid_year"):
            year = rng.choice([2025, 2026])
            items = [p for p in pays_by_vendor[vendor_id] if p.paid_at.year == year]
            currencies = {p.currency for p in items}
            vendor = ledger.vendor_name(vendor_id)
            if docs_of(items) and len(currencies) == 1 and len(items) <= MAX_COUNTED:
                total = sum((p.amount for p in items), Decimal(0))
                yield Draft(
                    "aggregation",
                    f"How much did we pay {vendor} in {year}?",
                    f"{money(total)} {currencies.pop()} across "
                    f"{n_of(len(items), 'payment')} made in {year}.",
                    docs_of(items),
                    True,
                    "hard",
                    {"amount": money(total), "count": len(items)},
                    check={"kind": "paid_year", "vendor_id": vendor_id, "year": year},
                )

    def po_count_year() -> Shape:
        for vendor_id in each_vendor("po_count_year"):
            year = rng.choice([2025, 2026])
            items = [p for p in pos_by_vendor[vendor_id] if p.order_date.year == year]
            vendor = ledger.vendor_name(vendor_id)
            if docs_of(items) and len(items) <= MAX_COUNTED:
                yield Draft(
                    "aggregation",
                    f"How many purchase orders did we place with {vendor} in {year}?",
                    f"{n_of(len(items), 'purchase order')} to {vendor} "
                    f"{'is' if len(items) == 1 else 'are'} dated in {year}.",
                    docs_of(items),
                    True,
                    "medium",
                    {"count": len(items)},
                    check={"kind": "po_count_year", "vendor_id": vendor_id, "year": year},
                )

    def top_vendor() -> Shape:
        spend: dict[tuple[str, int], dict[str, Decimal]] = defaultdict(lambda: defaultdict(Decimal))
        for i in ledger.invoices:
            if i.vendor_id in ledger.vendors:
                spend[(i.currency, i.invoice_date.year)][i.vendor_id or ""] += i.total
        for currency, year in _shuffled(rng, sorted(spend)):
            ranking = sorted(spend[(currency, year)].items(), key=lambda kv: -kv[1])
            if len(ranking) < 2 or ranking[0][1] == ranking[1][1]:
                continue
            top, amount = ranking[0]
            items = [
                i for i in by_vendor[top] if i.invoice_date.year == year and i.currency == currency
            ]
            if not docs_of(items):
                continue
            vendor = ledger.vendor_name(top)
            yield Draft(
                "aggregation",
                f"Which vendor did we spend the most with in {currency} in {year}?",
                f"{vendor}: {money(amount)} {currency} invoiced in {year}.",
                docs_of(items),
                True,
                "hard",
                {"vendor": vendor, "amount": money(amount)},
                check={"kind": "top_vendor", "currency": currency, "year": year},
            )

    return round_robin(
        n,
        count_year(),
        total_quarter(),
        balance(),
        top_vendor(),
        paid_year(),
        po_count_year(),
        count_quarter(),
        unpaid_count(),
    )


def reconciliation(ledger: Ledger, rng: random.Random, n: int) -> list[Draft]:
    fresh = Fresh()

    def inv_key(i: InvoiceRow) -> str:
        return i.id

    def rated(invoice: InvoiceRow) -> list[tuple[str, Decimal, Decimal]]:
        """(item, invoiced unit price, contract rate) for lines the rate card prices."""
        contract = ledger.contract_of(invoice)
        if contract is None:
            return []
        rates = {e["item"]: Decimal(e["unit_price"]) for e in contract.rate_card}
        return [
            (line.description, line.unit_price, rates[line.description])
            for line in invoice.line_items
            if line.unit_price is not None and line.description in rates
        ]

    def with_contract(invoice: InvoiceRow) -> bool:
        contract = ledger.contract_of(invoice)
        return bool(
            invoice.document_id
            and ledger.unique_number(invoice)
            and contract is not None
            and contract.document_id
        )

    def anomalies(invoice: InvoiceRow) -> list[str]:
        return list(ledger.labels.get(invoice.id, {}).get("anomalies", []))

    invoices = _shuffled(rng, ledger.invoices)

    def rate_check(pool: Iterable[InvoiceRow]) -> Shape:
        for i in fresh((i for i in pool if with_contract(i) and rated(i)), inv_key):
            contract = ledger.contract_of(i)
            assert contract is not None
            cur = i.currency
            off = [(item, p, r) for item, p, r in rated(i) if abs(p - r) >= TOLERANCE]
            expected: dict[str, Any]
            if off:
                detail = "; ".join(
                    f"{item} at {money(p)} {cur} vs the contract rate {money(r)} {cur} "
                    f"({(p - r) / r * 100:+.1f}%)"
                    for item, p, r in off
                )
                answer = (
                    f"No. Under {contract.contract_number}, {i.invoice_number} charges {detail}."
                )
                expected = {"match": False, "items": [item for item, _, _ in off]}
            else:
                answer = (
                    f"Yes. Every priced line on invoice {i.invoice_number} matches the rate card "
                    f"of {contract.contract_number}."
                )
                expected = {"match": True}
            yield Draft(
                "reconciliation",
                f"Do the unit prices on invoice {i.invoice_number} match the contract rates?",
                answer,
                _docs(i.document_id, contract.document_id),
                True,
                "hard",
                expected,
                check={"kind": "rate_match", "invoice_id": i.id},
            )

    def inflated() -> Shape:
        return rate_check(i for i in invoices if "price_inflation" in anomalies(i))

    def clean() -> Shape:
        return rate_check(i for i in invoices if not anomalies(i))

    def duplicates(label: str) -> Iterator[tuple[InvoiceRow, InvoiceRow]]:
        for i in invoices:
            original = ledger.invoice_by_id.get(
                ledger.labels.get(i.id, {}).get("duplicate_of") or ""
            )
            if (
                label in anomalies(i)
                and original is not None
                and (i.document_id or original.document_id)  # the graph holds both
                and i.vendor_id == original.vendor_id
                and i.total == original.total
            ):
                yield i, original

    def exact_duplicate() -> Shape:
        for i, original in fresh(duplicates("exact_duplicate"), lambda pair: pair[0].id):
            yield Draft(
                "reconciliation",
                f"Has {ledger.vendor_name(i.vendor_id)} invoice {i.invoice_number} been "
                "submitted more than once?",
                f"Yes. Invoice {i.invoice_number} ({original.invoice_date.isoformat()}, "
                f"{money(i.total)} {i.currency}) appears twice: the second copy duplicates the "
                "first with the same vendor, number, date and total.",
                _docs(i.document_id, original.document_id),
                True,
                "hard",
                {"duplicate": True, "number": i.invoice_number},
                check={"kind": "duplicate", "invoice_id": i.id, "original_id": original.id},
            )

    def near_duplicate() -> Shape:
        pairs = (
            (i, o)
            for i, o in duplicates("near_duplicate")
            if i.invoice_number != o.invoice_number
            and ledger.unique_number(i)
            and ledger.unique_number(o)
        )
        for i, original in fresh(pairs, lambda pair: pair[0].id):
            days = abs((i.invoice_date - original.invoice_date).days)
            yield Draft(
                "reconciliation",
                f"Does {ledger.vendor_name(i.vendor_id)} invoice {i.invoice_number} look like "
                "a duplicate of another invoice?",
                f"Yes. Invoice {i.invoice_number} ({i.invoice_date.isoformat()}) has the same "
                f"vendor and total ({money(i.total)} {i.currency}) as invoice "
                f"{original.invoice_number} ({original.invoice_date.isoformat()}), {days} days "
                "apart.",
                _docs(i.document_id, original.document_id),
                True,
                "hard",
                {"duplicate": True, "number": original.invoice_number},
                check={"kind": "near_duplicate", "invoice_id": i.id, "original_id": original.id},
            )

    def terms_days(i: InvoiceRow) -> int | None:
        return (i.due_date - i.invoice_date).days if i.due_date else None

    def terms_check(pool: Iterable[InvoiceRow]) -> Shape:
        for i in fresh((i for i in pool if with_contract(i) and i.due_date), inv_key):
            contract = ledger.contract_of(i)
            assert contract is not None and contract.payment_terms_days is not None
            days, allowed = terms_days(i), contract.payment_terms_days
            same = days == allowed
            when = (
                "the day it is dated" if days == 0 else f"{n_of(days or 0, 'day')} after its date"
            )
            verdict = "Yes" if same else "No"
            yield Draft(
                "reconciliation",
                f"Do the payment terms on invoice {i.invoice_number} follow the contract?",
                f"{verdict}. Invoice {i.invoice_number} is due {when} "
                f"({i.due_date}); contract {contract.contract_number} allows "
                f"{allowed} days.",
                _docs(i.document_id, contract.document_id),
                True,
                "hard",
                {"match": same, "days": allowed},
                check={"kind": "terms_match", "invoice_id": i.id},
            )

    def rush_terms() -> Shape:
        return terms_check(i for i in invoices if "rush_terms" in anomalies(i))

    def normal_terms() -> Shape:
        return terms_check(i for i in invoices if not anomalies(i))

    def paid_check(paid: bool) -> Shape:
        pool = (
            i
            for i in invoices
            if i.document_id
            and ledger.unique_number(i)
            and ledger.paid_by[i.id]
            and all(p.document_id for p in ledger.paid_by[i.id])
            and (ledger.outstanding(i) < TOLERANCE) == paid
            and ledger.outstanding(i) > -TOLERANCE  # overpaid: not a clean yes or no
        )
        for i in fresh(pool, inv_key):
            payments = ledger.paid_by[i.id]
            refs = ", ".join(p.payment_reference for p in payments)
            if paid:
                answer = (
                    f"Yes. Invoice {i.invoice_number} ({money(i.total)} {i.currency}) was paid "
                    f"in full by {refs}."
                )
            else:
                answer = (
                    f"No. Invoice {i.invoice_number} totals {money(i.total)} {i.currency}; "
                    f"{refs} paid {money(ledger.paid[i.id])} {i.currency}, leaving "
                    f"{money(ledger.outstanding(i))} {i.currency} outstanding."
                )
            yield Draft(
                "reconciliation",
                f"Has invoice {i.invoice_number} been paid in full?",
                answer,
                _docs(i.document_id, *(p.document_id for p in payments)),
                True,
                "medium",
                {"match": paid} | ({} if paid else {"amount": money(ledger.outstanding(i))}),
                check={"kind": "paid_in_full", "invoice_id": i.id},
            )

    return round_robin(
        n,
        inflated(),
        clean(),
        exact_duplicate(),
        near_duplicate(),
        rush_terms(),
        normal_terms(),
        paid_check(True),
        paid_check(False),
    )


# --------------------------------------------------------------------------- unanswerable

UNANSWERABLE = [
    ("What is the total on invoice INV-999999?", "There is no invoice INV-999999 in the records."),
    (
        "Which payment settled invoice INV-000000?",
        "There is no invoice INV-000000, so no payment settled it.",
    ),
    (
        "What rate does our contract with Zenith Robotics Ltd set for drone inspections?",
        "There is no contract with Zenith Robotics Ltd in the records.",
    ),
    (
        "What is the total of purchase order PO-1999-00001?",
        "There is no purchase order PO-1999-00001 in the records.",
    ),
]
"""Golden set v0's four; the generator yields them again, with their checks."""

FAKE_VENDORS = (
    "Zenith Robotics Ltd",
    "Aurora Biofuels GmbH",
    "Kestrel Aviation Services Ltd",
    "Pinnacle Quarry Supplies Pty",
    "Solace Wellness Retreats SARL",
    "Tidewater Desalination Inc",
)
MISSING_CLAUSES = (
    (
        "What notice period does contract {number} require for termination?",
        ("terminat", "notice"),
        "termination notice period",
    ),
    (
        "What late-payment interest does contract {number} charge?",
        ("interest", "late payment", "late-payment"),
        "late-payment interest",
    ),
    (
        "What is the liability cap in contract {number}?",
        ("liabil",),
        "liability cap",
    ),
    (
        "Which country's law governs contract {number}?",
        ("governing law", "governed by", "laws of"),
        "governing-law clause",
    ),
    (
        "Does contract {number} renew automatically?",
        ("renew",),
        "renewal clause",
    ),
)


def unanswerable(ledger: Ledger, rng: random.Random, n: int) -> list[Draft]:
    invoice_numbers = {i.invoice_number for i in ledger.invoices}
    po_numbers = set(ledger.po_by_number)
    references = {p.payment_reference for p in ledger.payments}
    vendor_words = {
        word.lower()
        for v in ledger.vendors.values()
        for name in (v.canonical_name, *(v.aliases or []))
        for word in name.split()
        if len(word) > 3
    }

    def unused(make: Callable[[], str], taken: set[str]) -> str:
        while (value := make()) in taken:
            pass
        taken.add(value)
        return value

    def none(question: str, answer: str, check: dict[str, Any]) -> Draft:
        return Draft(
            "unanswerable",
            question,
            answer,
            (),
            False,
            "medium",
            {"answerable": False},
            check=check,
        )

    def fixed() -> Shape:
        checks: list[dict[str, Any]] = [
            {"kind": "missing_invoice", "number": "INV-999999"},
            {"kind": "missing_invoice", "number": "INV-000000"},
            {"kind": "missing_vendor", "name": "Zenith Robotics Ltd"},
            {"kind": "missing_po", "number": "PO-1999-00001"},
        ]
        for (q, a), check in zip(UNANSWERABLE, checks, strict=True):
            yield none(q, a, check)

    def missing_invoice() -> Shape:
        templates = (
            ("When is invoice {n} due?", "There is no invoice {n} in the records."),
            ("Which purchase order does invoice {n} reference?", "There is no invoice {n}."),
            ("How much is still owed on invoice {n}?", "There is no invoice {n} in the records."),
        )
        while True:
            number = unused(lambda: f"INV-{rng.randrange(900_000, 999_999):06d}", invoice_numbers)
            q, a = rng.choice(templates)
            yield none(
                q.format(n=number),
                a.format(n=number),
                {"kind": "missing_invoice", "number": number},
            )

    def missing_po() -> Shape:
        while True:
            number = unused(
                lambda: f"PO-20{rng.randrange(10, 20)}-{rng.randrange(1, 99999):05d}", po_numbers
            )
            yield none(
                f"Which invoices were billed against purchase order {number}?",
                f"There is no purchase order {number} in the records.",
                {"kind": "missing_po", "number": number},
            )

    def missing_payment() -> Shape:
        while True:
            ref = unused(lambda: f"PAY-2019-{rng.randrange(1, 999999):06d}", references)
            yield none(
                f"How much was paid in remittance {ref}?",
                f"There is no remittance {ref} in the records.",
                {"kind": "missing_payment", "reference": ref},
            )

    def missing_vendor() -> Shape:
        for name in FAKE_VENDORS[1:]:
            if any(word.lower() in vendor_words for word in name.split() if len(word) > 3):
                continue
            yield none(
                f"How many invoices did {name} issue in 2025?",
                f"There is no vendor {name} in the records, so there are no invoices from it.",
                {"kind": "missing_vendor", "name": name},
            )

    def missing_clause() -> Shape:
        contracts = _shuffled(rng, (c for c in ledger.contracts.values() if c.document_id))
        clauses = list(MISSING_CLAUSES)
        for k, c in enumerate(contracts):
            template, keywords, what = clauses[k % len(clauses)]
            text = ledger.texts.get(c.document_id or "", "").lower()
            if not text or any(word in text for word in keywords):
                continue
            yield none(
                template.format(number=c.contract_number),
                f"Contract {c.contract_number} has no {what}; the records do not say.",
                {"kind": "missing_clause", "contract_id": c.id, "keywords": list(keywords)},
            )

    return round_robin(
        n,
        fixed(),
        missing_invoice(),
        missing_clause(),
        missing_po(),
        missing_vendor(),
        missing_payment(),
    )


GENERATORS: dict[str, Callable[[Ledger, random.Random, int], list[Draft]]] = {
    "lookup": lookup,
    "clause": clause,
    "multi_hop": multi_hop,
    "aggregation": aggregation,
    "reconciliation": reconciliation,
    "unanswerable": unanswerable,
}


# --------------------------------------------------------------------------- checks

_SPACE = re.compile(r"\s+")


def _amount_forms(amount: str) -> set[str]:
    plain = amount.replace(",", "")
    whole = plain[:-3] if plain.endswith(".00") else plain
    return {amount, plain, f"{int(Decimal(plain)):,}" if plain.endswith(".00") else amount, whole}


def check_grounding(draft: Draft, texts: dict[str, str]) -> list[str]:
    """What the draft's single source document does not support: an expected amount, number,
    date or day count missing from its text. Empty when grounded (or nothing to check)."""
    if draft.grounded_in is None:
        return []
    text = texts.get(draft.grounded_in)
    if not text:
        return [f"document {draft.grounded_in} has no text"]
    flat = _SPACE.sub(" ", text)
    issues: list[str] = []
    expected = draft.expected
    if "amount" in expected and not any(form in flat for form in _amount_forms(expected["amount"])):
        issues.append(f"amount {expected['amount']} not in the document")
    if "number" in expected and expected["number"] not in flat:
        issues.append(f"number {expected['number']} not in the document")
    if "date" in expected and date.fromisoformat(expected["date"]) not in set(find_dates(flat)):
        issues.append(f"date {expected['date']} not in the document")
    if "vendor" in expected and expected["vendor"].lower() not in flat.lower():
        issues.append(f"vendor {expected['vendor']} not in the document")
    for item in expected.get("items", []):
        if str(item).lower() not in flat.lower():
            issues.append(f"{item} not in the document")
    if "days" in expected and not re.search(rf"\b{expected['days']}\s+days\b", flat):
        issues.append(f"{expected['days']} days not in the document")
    return issues


# --------------------------------------------------------------------------- build


def build(
    ledger: Ledger, *, seed: int = 39, counts: dict[str, int] | None = None
) -> tuple[list[GoldenQuestion], list[str]]:
    """Grounded drafts in category order, numbered q001..., and the rejected drafts."""
    rng = random.Random(seed)
    drafts: list[Draft] = []
    rejected: list[str] = []
    for category, wanted in (counts or DEFAULT_COUNTS).items():
        for draft in GENERATORS[category](ledger, rng, wanted):
            issues = check_grounding(draft, ledger.texts)
            if issues:
                rejected.append(f"{draft.question} -- {'; '.join(issues)}")
            else:
                drafts.append(draft)
    return [draft.to_question(n) for n, draft in enumerate(drafts, start=1)], rejected


def next_number(questions: Sequence[GoldenQuestion]) -> int:
    return max((int(q.id[1:]) for q in questions), default=0) + 1


def extend(
    ledger: Ledger,
    existing: Sequence[GoldenQuestion],
    targets: dict[str, int],
    *,
    seed: int = 63,
) -> tuple[list[GoldenQuestion], list[str]]:
    """New grounded drafts that bring each category of ``existing`` up to its target,
    numbered after the last existing id. A draft asking a question the set already has is
    skipped, so existing ids keep their meaning."""
    rng = random.Random(seed)
    asked = {q.question.strip().lower() for q in existing}
    have: Counter[str] = Counter(str(q.category) for q in existing)
    drafts: list[Draft] = []
    rejected: list[str] = []
    for category, target in targets.items():
        need = target - have[category]
        if need <= 0:
            continue
        taken = 0
        for draft in GENERATORS[category](ledger, rng, need * 3 + 10):
            if taken == need:
                break
            if draft.question.strip().lower() in asked:
                continue
            issues = check_grounding(draft, ledger.texts)
            if issues:
                rejected.append(f"{draft.question} -- {'; '.join(issues)}")
                continue
            asked.add(draft.question.strip().lower())
            drafts.append(draft)
            taken += 1
        if taken < need:
            rejected.append(f"{category}: only {taken} of {need} new drafts")
    start = next_number(existing)
    return [d.to_question(start + k) for k, d in enumerate(drafts)], rejected


def parse_targets(text: str) -> dict[str, int]:
    """``lookup=25,clause=15`` -> ``{"lookup": 25, "clause": 15}``."""
    out: dict[str, int] = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        name, _, value = part.partition("=")
        if name not in GENERATORS or not value.isdigit():
            raise argparse.ArgumentTypeError(f"not a category=count: {part!r}")
        out[name] = int(value)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draft golden-set questions from the stores.")
    parser.add_argument("--out", type=Path, default=Path("data/golden/golden_set.draft.jsonl"))
    parser.add_argument("--records", type=Path, default=RECORDS)
    parser.add_argument("--seed", type=int, default=39)
    parser.add_argument(
        "--extend",
        type=Path,
        metavar="GOLDEN",
        help="draft only new questions, numbered after this set's, up to --targets",
    )
    parser.add_argument(
        "--targets",
        type=parse_targets,
        default=SYNTHETIC_TARGETS,
        help="per-category totals for --extend (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    with get_session_factory()() as session:
        ledger = load_ledger(session, args.records)
    if args.extend:
        questions, rejected = extend(
            ledger, load_golden_set(args.extend), args.targets, seed=args.seed
        )
    else:
        questions, rejected = build(ledger, seed=args.seed)
    for line in rejected:
        print(f"REJECTED {line}")
    for problem in problems(questions):
        print(f"PROBLEM {problem}")
    dump_golden_set(questions, args.out)
    print(summary(questions))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
