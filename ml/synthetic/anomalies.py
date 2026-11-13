"""Inject labelled anomalies into a linked ledger (labels for the anomaly model, layer 8).

Rates are shares of the **final** invoice count (``docs/layers/07-synthetic-data.md``):

==================  =====  ===========================================================
type                rate   how
==================  =====  ===========================================================
exact_duplicate     2 %    a copy of an invoice under a new record id (same number,
                           vendor, dates and amounts); ``duplicate_of`` names the source
near_duplicate      2 %    same vendor, lines and amounts; invoice number +/- 1 and the
                           date moved 1-3 days; ``duplicate_of`` names the source
price_inflation     2 %    one or more lines billed 10-40 % over the contract rate (the PO
                           keeps the contract rate, so the 3-way match breaks)
round_amount        1 %    total rounded up/down to whole thousands (millions for IDR),
                           no line items, no subtotal or tax
unknown_vendor      1 %    billed by a vendor that is not in the vendor master, with no
                           PO (its PO is removed from the ledger)
rush_terms          1 %    dated on a weekend and due within 0-3 days
==================  =====  ===========================================================

Duplicates are **added** rows; the other types **modify** an existing invoice in place.
Counts are exact (``round(rate x total)``), and every anomaly lands on a different source
invoice, so each invoice carries at most one label. Run :func:`inject_anomalies` before
:func:`ml.synthetic.link.make_payments`, so anomalous invoices get paid like any other.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal

from ledgerlens.core.schemas import LineItem
from ml.synthetic.link import Ledger, LinkedInvoice, money
from ml.synthetic.seed import SEED, VAT

ANOMALY_RATES: dict[str, float] = {
    "exact_duplicate": 0.02,
    "near_duplicate": 0.02,
    "price_inflation": 0.02,
    "round_amount": 0.01,
    "unknown_vendor": 0.01,
    "rush_terms": 0.01,
}
ADDED = ("exact_duplicate", "near_duplicate")  # new rows; everything else is in place
INFLATION = (0.10, 0.40)
ROUND_STEP = {"IDR": Decimal(1_000_000)}  # whole thousands otherwise
UNKNOWN_VENDORS = [
    "Apex Global Solutions LLC",
    "Quickserve Consulting Ltd",
    "Prime Procurement Services",
    "Westgate Trading FZE",
    "Northstar Advisory Group",
    "BlueOcean Supplies Inc",
    "Summit Office Partners",
    "Vertex Facility Services",
]


@dataclass(frozen=True)
class AnomalyPlan:
    base: int  # clean invoices to generate
    counts: dict[str, int]

    @property
    def total(self) -> int:
        return self.base + sum(self.counts[k] for k in ADDED if k in self.counts)


def plan_for_total(total: int, rates: Mapping[str, float] = ANOMALY_RATES) -> AnomalyPlan:
    """How many clean invoices to generate so that, with duplicates added, the ledger
    has ``total`` invoices and each anomaly type is ``round(rate x total)`` of them."""
    counts = {kind: round(rate * total) for kind, rate in rates.items()}
    base = total - sum(counts[k] for k in ADDED if k in counts)
    in_place = sum(n for k, n in counts.items() if k not in ADDED)
    if base <= 0 or in_place + sum(counts.get(k, 0) for k in ADDED) > base:
        raise ValueError(f"too few invoices ({total}) for these anomaly rates")
    return AnomalyPlan(base, counts)


def inject_anomalies(
    ledger: Ledger, plan: AnomalyPlan, *, seed: int = SEED
) -> dict[str, list[LinkedInvoice]]:
    """Apply ``plan`` to ``ledger`` (which must hold ``plan.base`` clean invoices).

    Returns the labelled invoices by anomaly type.
    """
    if len(ledger.invoices) != plan.base:
        raise ValueError(f"plan expects {plan.base} invoices, ledger has {len(ledger.invoices)}")
    rng = random.Random(seed + 4)
    order = list(range(plan.base))
    rng.shuffle(order)
    pool = iter(order)
    picked: dict[str, list[LinkedInvoice]] = {}
    for kind in ANOMALY_RATES:
        n = plan.counts.get(kind, 0)
        picked[kind] = [ledger.invoices[next(pool)] for _ in range(n)]

    labelled: dict[str, list[LinkedInvoice]] = {kind: [] for kind in plan.counts}
    next_id = len(ledger.invoices) + 1
    for kind in ADDED:
        for source in picked.get(kind, []):
            duplicate = _duplicate(source, f"SINV-{next_id:06d}", kind, ledger, rng)
            next_id += 1
            ledger.invoices.append(duplicate)
            ledger.purchase_order(source.po_number or "").invoice_ids.append(duplicate.id)
            labelled[kind].append(duplicate)
    for kind, targets in picked.items():
        if kind in ADDED:
            continue
        for target in targets:
            _MODIFY[kind](target, ledger, rng)
            target.anomalies.append(kind)
            labelled[kind].append(target)
    return labelled


# --------------------------------------------------------------------------- the anomalies


def _duplicate(
    source: LinkedInvoice, new_id: str, kind: str, ledger: Ledger, rng: random.Random
) -> LinkedInvoice:
    duplicate = LinkedInvoice(
        id=new_id,
        invoice=source.invoice,
        vendor_id=source.vendor_id,
        contract_number=source.contract_number,
        anomalies=[kind],
        duplicate_of=source.id,
        details=copy.deepcopy(source.details),
    )
    if kind == "near_duplicate":
        taken = {
            i.invoice.invoice_number for i in ledger.invoices if i.vendor_id == source.vendor_id
        }
        number = int(source.invoice.invoice_number.removeprefix("INV-"))
        steps = [1, -1] if rng.random() < 0.5 else [-1, 1]
        new_number = next(
            (f"INV-{number + s:06d}" for s in steps if f"INV-{number + s:06d}" not in taken),
            f"INV-{number:06d}A",  # both neighbours taken: a typo-style suffix
        )
        shift = timedelta(days=rng.choice([-3, -2, -1, 1, 2, 3]))
        invoice = source.invoice
        due = invoice.due_date + shift if invoice.due_date else None
        duplicate.replace(
            invoice_number=new_number, invoice_date=invoice.invoice_date + shift, due_date=due
        )
        duplicate.details = {"date_shift_days": shift.days}
    return duplicate


def _inflate(target: LinkedInvoice, ledger: Ledger, rng: random.Random) -> None:
    invoice = target.invoice
    vendor = ledger.vendor(target.vendor_id or "")
    chosen = set(
        rng.sample(range(len(invoice.line_items)), rng.randint(1, len(invoice.line_items)))
    )
    lines, inflation = [], {}
    for i, line in enumerate(invoice.line_items):
        if i in chosen and line.unit_price is not None and line.quantity is not None:
            factor = Decimal(str(round(rng.uniform(*INFLATION), 3)))
            unit_price = money(line.unit_price * (1 + factor), invoice.currency)
            line = LineItem(
                description=line.description,
                quantity=line.quantity,
                unit_price=unit_price,
                amount=money(unit_price * line.quantity, invoice.currency),
            )
            inflation[line.description] = float(factor)
        lines.append(line)
    subtotal = sum((line.amount for line in lines), Decimal(0))
    tax = money(subtotal * Decimal(str(VAT[vendor.country])), invoice.currency)
    target.replace(line_items=lines, subtotal=subtotal, tax=tax, total=subtotal + tax)
    target.details = {"inflation": inflation}


def _round_amount(target: LinkedInvoice, ledger: Ledger, rng: random.Random) -> None:
    invoice = target.invoice
    step = ROUND_STEP.get(invoice.currency, Decimal(1000))
    rounded = max(step, (invoice.total / step).quantize(Decimal(1)) * step)
    if rounded == invoice.total:
        rounded += step
    target.replace(line_items=[], subtotal=None, tax=None, total=rounded)
    target.details = {"original_total": str(invoice.total)}


def _unknown_vendor(target: LinkedInvoice, ledger: Ledger, rng: random.Random) -> None:
    known = {v.name for v in ledger.vendors}
    name = rng.choice([n for n in UNKNOWN_VENDORS if n not in known])
    po_number = target.po_number
    ledger.purchase_orders = [p for p in ledger.purchase_orders if p.id != po_number]
    target.replace(vendor_name=name, vendor_tax_id=None, po_number=None)
    target.vendor_id = None
    target.contract_number = None
    target.details = {"removed_po": po_number}


def _rush_terms(target: LinkedInvoice, ledger: Ledger, rng: random.Random) -> None:
    invoice = target.invoice
    weekend_day = rng.choice([5, 6])  # Saturday or Sunday
    issued = invoice.invoice_date + timedelta(
        days=(weekend_day - invoice.invoice_date.weekday()) % 7
    )
    due = issued + timedelta(days=rng.randint(0, 3))
    target.replace(invoice_date=issued, due_date=due)
    target.details = {"terms_days": (due - issued).days}


_MODIFY = {
    "price_inflation": _inflate,
    "round_amount": _round_amount,
    "unknown_vendor": _unknown_vendor,
    "rush_terms": _rush_terms,
}
