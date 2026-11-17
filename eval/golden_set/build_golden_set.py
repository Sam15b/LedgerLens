"""Draft golden-set questions from the synthetic ledger, where every answer is known.

    python -m eval.golden_set.build_golden_set --out data/golden/golden_set.draft.jsonl

Answers are computed from the stores' own view of the ledger (Postgres: records,
payment allocations, which record came from which document), with the anomaly labels
from the generator's ``data/synthetic/records.jsonl``. Only records rendered as documents
can be evidence, so every question is about rendered documents, and each relevant
document id is a real ``documents.id``.

Per category (counts are the defaults):

* **lookup** (12) -- an invoice's total, due date or PO; a PO's total; a remittance's
  amount and vendor.
* **clause** (8) -- a contract's payment terms, a rate on its rate card, its end date.
* **multi_hop** (12) -- which payment settled an invoice; which contract governs an
  invoice's PO; what is still owed on a partly paid invoice; a vendor's invoices with a
  PO and no payment.
* **aggregation** (10) -- invoice counts and totals per vendor and period, outstanding
  balances, the top vendor by spend in a currency.
* **reconciliation** (6) -- invoice prices against the contract rate (inflated and
  clean), and exact duplicate invoices.
* **unanswerable** (4) -- documents that do not exist; the answer is that the context
  has nothing.

Every draft is then **checked against the evidence** (:func:`check_grounding`): amounts,
numbers and dates a single-document answer states must appear in that document's text.
A draft that fails is dropped and reported. Drafts are written with ``reviewed: false``;
reviewing them is a separate, manual step before they join the golden set.

Sampling is seeded (``--seed``), so the same stores give the same drafts.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from eval.golden_set.schema import GoldenQuestion, dump_golden_set, problems, summary
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
TOLERANCE = Decimal("0.01")
T = TypeVar("T")


def money(amount: Decimal | float) -> str:
    return f"{Decimal(str(amount)):,.2f}"


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

    def to_question(self, number: int) -> GoldenQuestion:
        return GoldenQuestion(
            id=f"q{number:03d}",
            category=self.category,  # type: ignore[arg-type]
            question=self.question,
            ground_truth_answer=self.answer,
            relevant_document_ids=self.documents,
            requires_graph=self.requires_graph,
            difficulty=self.difficulty,  # type: ignore[arg-type]
            expected=self.expected,
        )


def _docs(*ids: str | None) -> tuple[str, ...]:
    return tuple(dict.fromkeys(i for i in ids if i))


def _shuffled(rng: random.Random, items: Iterable[T]) -> list[T]:
    out = list(items)
    rng.shuffle(out)
    return out


def lookup(ledger: Ledger, rng: random.Random, n: int) -> list[Draft]:
    invoices = _shuffled(
        rng, (i for i in ledger.invoices if i.document_id and ledger.unique_number(i))
    )
    pos = _shuffled(rng, (p for p in ledger.purchase_orders.values() if p.document_id))
    payments = _shuffled(rng, (p for p in ledger.payments if p.document_id))
    drafts: list[Draft] = []
    for i in invoices[:4]:
        drafts.append(
            Draft(
                "lookup",
                f"What is the total on invoice {i.invoice_number}?",
                f"Invoice {i.invoice_number} from {ledger.vendor_name(i.vendor_id)} totals "
                f"{money(i.total)} {i.currency}.",
                _docs(i.document_id),
                False,
                "easy",
                {"amount": money(i.total), "currency": i.currency},
                i.document_id,
            )
        )
    for i in invoices[4:6]:
        if i.due_date is None:
            continue
        drafts.append(
            Draft(
                "lookup",
                f"When is invoice {i.invoice_number} due?",
                f"Invoice {i.invoice_number} is due on {i.due_date.isoformat()}.",
                _docs(i.document_id),
                False,
                "easy",
                {"date": i.due_date.isoformat()},
                i.document_id,
            )
        )
    for i in [i for i in invoices[6:] if i.po_number][:2]:
        drafts.append(
            Draft(
                "lookup",
                f"Which purchase order does invoice {i.invoice_number} reference?",
                f"Invoice {i.invoice_number} references purchase order {i.po_number}.",
                _docs(i.document_id),
                False,
                "easy",
                {"number": i.po_number},
                i.document_id,
            )
        )
    for p in pos[:2]:
        drafts.append(
            Draft(
                "lookup",
                f"What is the total of purchase order {p.po_number}?",
                f"Purchase order {p.po_number} to {ledger.vendor_name(p.vendor_id)} totals "
                f"{money(p.total)} {p.currency}.",
                _docs(p.document_id),
                False,
                "easy",
                {"amount": money(p.total), "currency": p.currency},
                p.document_id,
            )
        )
    for pay in payments[:2]:
        vendor = ledger.vendor_name(pay.vendor_id)
        drafts.append(
            Draft(
                "lookup",
                f"How much was paid in remittance {pay.payment_reference}, and to which vendor?",
                f"Remittance {pay.payment_reference} paid {money(pay.amount)} {pay.currency} to "
                f"{vendor} on {pay.paid_at.isoformat()}.",
                _docs(pay.document_id),
                False,
                "easy",
                {"amount": money(pay.amount), "currency": pay.currency, "vendor": vendor},
                pay.document_id,
            )
        )
    return drafts[:n]


def clause(ledger: Ledger, rng: random.Random, n: int) -> list[Draft]:
    contracts = _shuffled(rng, (c for c in ledger.contracts.values() if c.document_id))
    drafts: list[Draft] = []
    for c in contracts[:3]:
        vendor = ledger.vendor_name(c.vendor_id)
        drafts.append(
            Draft(
                "clause",
                f"Within how many days must invoices be paid under our contract with {vendor}?",
                f"Contract {c.contract_number} requires payment within {c.payment_terms_days} "
                "days of the invoice date.",
                _docs(c.document_id),
                False,
                "easy",
                {"days": c.payment_terms_days},
                c.document_id,
            )
        )
    for c in contracts[3:6]:
        entry = rng.choice(c.rate_card)
        rate = money(Decimal(entry["unit_price"]))
        unit = f" per {entry['unit']}" if entry.get("unit") else ""
        drafts.append(
            Draft(
                "clause",
                f"What rate does the {ledger.vendor_name(c.vendor_id)} contract set for "
                f"{entry['item']}?",
                f"Contract {c.contract_number} sets {entry['item']} at {rate} {c.currency}{unit}.",
                _docs(c.document_id),
                False,
                "medium",
                {"amount": rate, "currency": c.currency},
                c.document_id,
            )
        )
    for c in contracts[6:]:
        if len(drafts) >= 8 or c.end_date is None:
            continue
        drafts.append(
            Draft(
                "clause",
                f"When does contract {c.contract_number} expire?",
                f"Contract {c.contract_number} with {ledger.vendor_name(c.vendor_id)} runs "
                f"until {c.end_date.isoformat()}.",
                _docs(c.document_id),
                False,
                "easy",
                {"date": c.end_date.isoformat()},
                c.document_id,
            )
        )
    return drafts[:n]


def multi_hop(ledger: Ledger, rng: random.Random, n: int) -> list[Draft]:
    drafts: list[Draft] = []

    def rendered(invoice: InvoiceRow) -> bool:
        return bool(invoice.document_id) and ledger.unique_number(invoice)

    # which payment settled an invoice: fully paid by exactly one rendered remittance
    settled = _shuffled(
        rng,
        (
            i
            for i in ledger.invoices
            if rendered(i)
            and len(ledger.paid_by[i.id]) == 1
            and ledger.paid_by[i.id][0].document_id
            and ledger.outstanding(i) < TOLERANCE
        ),
    )
    for i in settled[:4]:
        pay = ledger.paid_by[i.id][0]
        drafts.append(
            Draft(
                "multi_hop",
                f"Which payment settled invoice {i.invoice_number}?",
                f"Invoice {i.invoice_number} was settled by remittance {pay.payment_reference} "
                f"on {pay.paid_at.isoformat()}.",
                _docs(i.document_id, pay.document_id),
                True,
                "medium",
                {"number": pay.payment_reference, "date": pay.paid_at.isoformat()},
            )
        )
    # which contract governs the PO an invoice references
    governed = [
        (i, c)
        for i in ledger.invoices
        if rendered(i) and (c := ledger.contract_of(i)) is not None and c.document_id
    ]
    for i, c in _shuffled(rng, governed)[:3]:
        po = ledger.po_by_number[i.po_number or ""]
        drafts.append(
            Draft(
                "multi_hop",
                f"Which contract governs the purchase order on invoice {i.invoice_number}?",
                f"Invoice {i.invoice_number} references {po.po_number}, which is governed by "
                f"contract {c.contract_number} ({c.title}).",
                _docs(i.document_id, c.document_id, po.document_id),
                True,
                "medium",
                {"number": c.contract_number, "po_number": po.po_number},
            )
        )
    # what is still owed on a partly paid invoice (the synthetic remittances each pay one
    # invoice, so "which invoices did remittance X pay" has no interesting answer)
    partial = [
        i
        for i in ledger.invoices
        if rendered(i) and ledger.paid_by[i.id] and ledger.outstanding(i) >= TOLERANCE
    ]
    for i in _shuffled(rng, partial)[:3]:
        payments = ledger.paid_by[i.id]
        paid = ", ".join(
            f"{p.payment_reference} ({money(a.amount)} {i.currency})"
            for p in payments
            for a in p.allocations
            if a.invoice_id == i.id
        )
        drafts.append(
            Draft(
                "multi_hop",
                f"How much is still owed on invoice {i.invoice_number}?",
                f"{money(ledger.outstanding(i))} {i.currency}: invoice {i.invoice_number} totals "
                f"{money(i.total)} {i.currency} and was partly paid by {paid}.",
                _docs(i.document_id, *(p.document_id for p in payments)),
                True,
                "hard",
                {"amount": money(ledger.outstanding(i)), "currency": i.currency},
            )
        )
    # a vendor's invoices with a PO and no payment at all (no shared numbers: duplicates
    # would list one number twice)
    open_by_vendor: dict[str, list[InvoiceRow]] = defaultdict(list)
    for i in ledger.invoices:
        if i.vendor_id and i.po_number and not ledger.paid_by[i.id]:
            open_by_vendor[i.vendor_id].append(i)
    candidates = [
        (vendor_id, items)
        for vendor_id, items in sorted(open_by_vendor.items())
        if 1 <= len(items) <= 6
        and any(i.document_id for i in items)
        and all(ledger.unique_number(i) for i in items)
    ]
    for vendor_id, items in _shuffled(rng, candidates)[:3]:
        items = sorted(items, key=lambda i: i.invoice_date)
        listed = ", ".join(
            f"{i.invoice_number} ({money(i.total)} {i.currency}, PO {i.po_number})" for i in items
        )
        drafts.append(
            Draft(
                "multi_hop",
                f"Which {ledger.vendor_name(vendor_id)} invoices reference a purchase order but "
                "have no payment?",
                f"{len(items)} invoice(s): {listed}.",
                _docs(*(i.document_id for i in items)),
                True,
                "hard",
                {"numbers": [i.invoice_number for i in items]},
            )
        )
    return drafts[:n]


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
    vendors = _shuffled(
        rng, sorted(v for v, items in by_vendor.items() if any(i.document_id for i in items))
    )

    def docs_of(items: Iterable[InvoiceRow]) -> tuple[str, ...]:
        return _docs(*(i.document_id for i in sorted(items, key=lambda i: i.invoice_date)))

    counts: list[Draft] = []
    totals: list[Draft] = []
    balances: list[Draft] = []
    tops: list[Draft] = []
    for vendor_id in vendors:
        vendor = ledger.vendor_name(vendor_id)
        if len(counts) < 3:  # invoice count in a year
            year = rng.choice([2025, 2026])
            items = [i for i in by_vendor[vendor_id] if i.invoice_date.year == year]
            if docs_of(items):
                counts.append(
                    Draft(
                        "aggregation",
                        f"How many invoices did {vendor} issue in {year}?",
                        f"{vendor} issued {len(items)} invoices dated in {year}.",
                        docs_of(items),
                        True,
                        "medium",
                        {"count": len(items)},
                    )
                )
                continue
        if len(totals) < 3:  # total invoiced in a quarter
            label, start, end = _period(rng)
            items = [i for i in by_vendor[vendor_id] if start <= i.invoice_date <= end]
            currencies = {i.currency for i in items}
            if docs_of(items) and len(currencies) == 1:
                total = sum((i.total for i in items), Decimal(0))
                totals.append(
                    Draft(
                        "aggregation",
                        f"What was the total invoiced by {vendor} in {label}?",
                        f"{vendor} invoiced {money(total)} {currencies.pop()} across "
                        f"{len(items)} invoices dated {start.isoformat()} to {end.isoformat()}.",
                        docs_of(items),
                        True,
                        "hard",
                        {"amount": money(total), "count": len(items)},
                    )
                )
                continue
        if len(balances) < 2:  # outstanding balance
            items = [i for i in by_vendor[vendor_id] if ledger.outstanding(i) >= TOLERANCE]
            currencies = {i.currency for i in items}
            unambiguous = all(ledger.unique_number(i) for i in items)  # see Ledger.paid
            if docs_of(items) and len(currencies) == 1 and len(items) <= 15 and unambiguous:
                balance = sum((ledger.outstanding(i) for i in items), Decimal(0))
                balances.append(
                    Draft(
                        "aggregation",
                        f"What is our total outstanding balance with {vendor}?",
                        f"{money(balance)} {currencies.pop()} is outstanding on {len(items)} "
                        f"{vendor} invoices.",
                        docs_of(items),
                        True,
                        "hard",
                        {"amount": money(balance), "count": len(items)},
                    )
                )
    spend: dict[tuple[str, int], dict[str, Decimal]] = defaultdict(lambda: defaultdict(Decimal))
    for i in ledger.invoices:
        if i.vendor_id in ledger.vendors:
            spend[(i.currency, i.invoice_date.year)][i.vendor_id or ""] += i.total
    for currency, year in _shuffled(rng, sorted(spend)):
        ranking = sorted(spend[(currency, year)].items(), key=lambda kv: -kv[1])
        if len(tops) == 2 or len(ranking) < 2:
            continue
        top, amount = ranking[0]
        items = [
            i for i in by_vendor[top] if i.invoice_date.year == year and i.currency == currency
        ]
        if not docs_of(items):
            continue
        vendor = ledger.vendor_name(top)
        tops.append(
            Draft(
                "aggregation",
                f"Which vendor did we spend the most with in {currency} in {year}?",
                f"{vendor}: {money(amount)} {currency} invoiced in {year}.",
                docs_of(items),
                True,
                "hard",
                {"vendor": vendor, "amount": money(amount)},
            )
        )
    return [*counts, *totals, *balances, *tops][:n]


def reconciliation(ledger: Ledger, rng: random.Random, n: int) -> list[Draft]:
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

    def usable(invoice: InvoiceRow) -> bool:
        contract = ledger.contract_of(invoice)
        return bool(
            invoice.document_id
            and ledger.unique_number(invoice)
            and contract is not None
            and contract.document_id
            and rated(invoice)
        )

    def anomalies(invoice: InvoiceRow) -> list[str]:
        return list(ledger.labels.get(invoice.id, {}).get("anomalies", []))

    inflated = [i for i in ledger.invoices if "price_inflation" in anomalies(i) and usable(i)]
    clean = [i for i in ledger.invoices if not anomalies(i) and usable(i)]
    drafts: list[Draft] = []
    for i in _shuffled(rng, inflated)[:2] + _shuffled(rng, clean)[:2]:
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
            answer = f"No. Under {contract.contract_number}, {i.invoice_number} charges {detail}."
            expected = {"match": False, "items": [item for item, _, _ in off]}
        else:
            answer = (
                f"Yes. Every priced line on invoice {i.invoice_number} matches the rate card of "
                f"{contract.contract_number}."
            )
            expected = {"match": True}
        drafts.append(
            Draft(
                "reconciliation",
                f"Do the unit prices on invoice {i.invoice_number} match the contract rates?",
                answer,
                _docs(i.document_id, contract.document_id),
                True,
                "hard",
                expected,
            )
        )
    duplicates = [
        i
        for i in ledger.invoices
        if i.document_id
        and "exact_duplicate" in anomalies(i)
        and ledger.labels[i.id].get("duplicate_of") in ledger.invoice_by_id
    ]
    for i in _shuffled(rng, duplicates)[:2]:
        original = ledger.invoice_by_id[ledger.labels[i.id]["duplicate_of"]]
        drafts.append(
            Draft(
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
            )
        )
    return drafts[:n]


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


def unanswerable(ledger: Ledger, rng: random.Random, n: int) -> list[Draft]:
    return [
        Draft("unanswerable", q, a, (), False, "medium", {"answerable": False})
        for q, a in UNANSWERABLE[:n]
    ]


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draft golden-set questions from the stores.")
    parser.add_argument("--out", type=Path, default=Path("data/golden/golden_set.draft.jsonl"))
    parser.add_argument("--records", type=Path, default=RECORDS)
    parser.add_argument("--seed", type=int, default=39)
    args = parser.parse_args(argv)
    with get_session_factory()() as session:
        ledger = load_ledger(session, args.records)
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
