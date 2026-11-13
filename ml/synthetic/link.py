"""Turn sampled invoice rows into a linked ledger: vendors, contracts, POs and invoices.

Every vendor in the hand-made vendor master (``vendor_master.csv``) gets one framework
**contract** with Northwind Traders: a rate card pricing every item of its category's
catalog (in the vendor's currency) and the vendor's payment terms.

Each sampled row (GAN output, :mod:`ml.synthetic.gan.sample`) becomes one **invoice**
billed against its own **purchase order**:

* the row decides the *shape*: ``vendor_category`` picks the vendor, ``line_count`` the
  number of lines (at most the rate card's size), ``subtotal`` the order size (converted
  to the vendor's currency), ``day_of_month`` the issue day;
* the contract decides the *prices*: every line is priced at the contract rate, with the
  quantity chosen so the lines add up to about the sampled subtotal. Tax comes from the
  vendor's country (``VAT``) and the due date from the contract's payment terms.

So before anomalies are injected (:mod:`ml.synthetic.anomalies`) every invoice is clean:
it matches its PO line for line, its prices match the contract, it adds up
(``Invoice.consistency_issues() == []``), and it is dated on a business day.

Identifiers: ``SINV-000001`` is the record id (unique, even for duplicate invoices);
``invoice_number`` is the vendor's own number (``INV-004213``, increasing per vendor with
gaps, as a vendor billing other customers too would have); POs are ``PO-2025-00042``.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from ledgerlens.core.schemas import Contract, Invoice, LineItem, PurchaseOrder, RateCardEntry
from ml.synthetic.seed import BUYER, CATALOG, CENTS, FX, SEED, VAT, Vendor

START = date(2025, 1, 6)  # first invoice date
DAYS = 640  # invoice dates fall in [START, START + DAYS)
CONTRACT_YEARS = 3
MAX_QTY = 9999


def money(value: float | Decimal, currency: str) -> Decimal:
    """``value`` rounded half-up to the currency's step (may be 0)."""
    step = CENTS.get(currency, Decimal("0.01"))
    steps = (Decimal(str(value)) / step).quantize(Decimal(1), rounding=ROUND_HALF_UP)
    return steps * step


def business_day(day: date) -> date:
    """``day``, or the Monday after it if it falls on a weekend."""
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


# --------------------------------------------------------------------------- records


@dataclass
class LinkedPO:
    po: PurchaseOrder
    vendor_id: str
    contract_number: str
    invoice_ids: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.po.po_number

    def to_record(self) -> dict[str, Any]:
        return {
            "record_type": "purchase_order",
            "id": self.id,
            "vendor_id": self.vendor_id,
            "contract_number": self.contract_number,
            "invoice_ids": self.invoice_ids,
            "data": self.po.model_dump(mode="json"),
        }


@dataclass
class LinkedInvoice:
    id: str
    invoice: Invoice
    vendor_id: str | None  # None: the vendor is not in the master (an anomaly)
    contract_number: str | None
    anomalies: list[str] = field(default_factory=list)
    duplicate_of: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    payment_status: str | None = None

    @property
    def po_number(self) -> str | None:
        return self.invoice.po_number

    def replace(self, **changes: Any) -> None:
        """Re-validate the invoice with ``changes`` applied."""
        self.invoice = Invoice.model_validate(self.invoice.model_dump() | changes)

    def to_record(self) -> dict[str, Any]:
        return {
            "record_type": "invoice",
            "id": self.id,
            "vendor_id": self.vendor_id,
            "contract_number": self.contract_number,
            "po_number": self.po_number,
            "payment_status": self.payment_status,
            "anomalies": self.anomalies,
            "duplicate_of": self.duplicate_of,
            "details": self.details,
            "data": self.invoice.model_dump(mode="json"),
        }


@dataclass
class Ledger:
    vendors: list[Vendor]
    contracts: dict[str, Contract]  # by vendor_id
    purchase_orders: list[LinkedPO] = field(default_factory=list)
    invoices: list[LinkedInvoice] = field(default_factory=list)

    def vendor(self, vendor_id: str) -> Vendor:
        return next(v for v in self.vendors if v.vendor_id == vendor_id)

    def purchase_order(self, po_number: str) -> LinkedPO:
        return next(p for p in self.purchase_orders if p.id == po_number)

    def records(self) -> list[dict[str, Any]]:
        """Every record as one JSON-able dict (one ``records.jsonl`` line each)."""
        out: list[dict[str, Any]] = [
            {"record_type": "vendor", "id": v.vendor_id, "data": vars(v)} for v in self.vendors
        ]
        out += [
            {
                "record_type": "contract",
                "id": c.contract_number,
                "vendor_id": vendor_id,
                "data": c.model_dump(mode="json"),
            }
            for vendor_id, c in self.contracts.items()
        ]
        out += [p.to_record() for p in self.purchase_orders]
        out += [i.to_record() for i in self.invoices]
        return out


# --------------------------------------------------------------------------- contracts


def contract_number(vendor: Vendor) -> str:
    return f"CT-{vendor.vendor_id}"


def make_contract(vendor: Vendor, rng: random.Random) -> Contract:
    """A framework contract pricing the vendor's whole catalog, on the vendor's terms."""
    start = date(2024, 7, 1) + timedelta(days=rng.randrange(150))
    rate_card = [
        RateCardEntry(
            item=item,
            unit=unit,
            unit_price=max(
                money(rng.uniform(low, high) * FX[vendor.currency], vendor.currency),
                CENTS.get(vendor.currency, Decimal("0.01")),
            ),
        )
        for item, unit, low, high in CATALOG[vendor.category]
    ]
    return Contract(
        contract_number=contract_number(vendor),
        title=f"Framework supply agreement: {vendor.category.replace('_', ' ')}",
        vendor_name=vendor.name,
        customer_name=BUYER,
        start_date=start,
        end_date=start.replace(year=start.year + CONTRACT_YEARS) - timedelta(days=1),
        payment_terms_days=vendor.payment_terms_days,
        currency=vendor.currency,
        rate_card=rate_card,
    )


def make_contracts(vendors: Sequence[Vendor], *, seed: int = SEED) -> dict[str, Contract]:
    rng = random.Random(seed)
    return {v.vendor_id: make_contract(v, rng) for v in vendors}


# --------------------------------------------------------------------------- POs + invoices


def _convert(amount: float, from_currency: str, to_currency: str) -> float:
    if from_currency not in FX or from_currency == to_currency:
        return amount
    return amount / FX[from_currency] * FX[to_currency]


def _issue_date(day_of_month: int, rng: random.Random) -> date:
    month_start = (START + timedelta(days=rng.randrange(DAYS))).replace(day=1)
    next_month = (month_start + timedelta(days=32)).replace(day=1)
    last_day = (next_month - timedelta(days=1)).day
    day = month_start.replace(day=min(max(day_of_month, 1), last_day))
    return business_day(max(day, START))


def price_lines(
    contract: Contract, line_count: int, subtotal: float, rng: random.Random
) -> list[LineItem]:
    """Lines at contract rates whose amounts add up to about ``subtotal``."""
    entries = rng.sample(contract.rate_card, min(max(line_count, 1), len(contract.rate_card)))
    per_line = subtotal / len(entries)
    lines = []
    for entry in entries:
        rate = entry.unit_price
        qty = min(MAX_QTY, max(1, round(per_line / float(rate)))) if rate > 0 else 1
        lines.append(
            LineItem(
                description=entry.item,
                quantity=Decimal(qty),
                unit_price=rate,
                amount=money(rate * qty, contract.currency),
            )
        )
    return lines


def _totals(lines: Sequence[LineItem], vendor: Vendor) -> tuple[Decimal, Decimal, Decimal]:
    subtotal = sum((line.amount for line in lines), Decimal(0))
    tax = money(subtotal * Decimal(str(VAT[vendor.country])), vendor.currency)
    return subtotal, tax, subtotal + tax


def link_rows(
    rows: Sequence[Mapping[str, Any]],
    vendors: Sequence[Vendor],
    contracts: Mapping[str, Contract],
    *,
    seed: int = SEED,
) -> Ledger:
    """One PO + one invoice per row (see the module docstring)."""
    rng = random.Random(seed + 2)
    by_category: dict[str, list[Vendor]] = defaultdict(list)
    for vendor in vendors:
        by_category[vendor.category].append(vendor)

    # Pick vendor and date first, so numbers can be handed out in date order.
    planned = []
    for row in rows:
        vendor = rng.choice(by_category.get(str(row["vendor_category"])) or list(vendors))
        planned.append((_issue_date(int(row["day_of_month"]), rng), vendor, row))
    planned.sort(key=lambda p: p[0])

    ledger = Ledger(list(vendors), dict(contracts))
    next_number = {v.vendor_id: rng.randint(1000, 9000) for v in vendors}
    po_sequence: dict[int, int] = defaultdict(int)
    for i, (issued, vendor, row) in enumerate(planned, start=1):
        contract = contracts[vendor.vendor_id]
        number = contract_number(vendor)
        target = _convert(float(row["subtotal"]), str(row["currency"]), vendor.currency)
        lines = price_lines(contract, int(row["line_count"]), target, rng)
        subtotal, tax, total = _totals(lines, vendor)

        delivered = issued - timedelta(days=rng.randint(0, 5))
        ordered = business_day(delivered - timedelta(days=rng.randint(2, 21)))
        delivered = max(delivered, ordered)
        po_sequence[ordered.year] += 1
        po = PurchaseOrder(
            po_number=f"PO-{ordered.year}-{po_sequence[ordered.year]:05d}",
            vendor_name=vendor.name,
            vendor_tax_id=vendor.tax_id,
            buyer_name=BUYER,
            order_date=ordered,
            delivery_date=delivered,
            contract_number=number,
            currency=vendor.currency,
            subtotal=subtotal,
            tax=tax,
            total=total,
            line_items=lines,
        )
        next_number[vendor.vendor_id] += rng.randint(2, 40)
        invoice = Invoice(
            invoice_number=f"INV-{next_number[vendor.vendor_id]:06d}",
            vendor_name=vendor.name,
            vendor_tax_id=vendor.tax_id,
            customer_name=BUYER,
            invoice_date=issued,
            due_date=issued + timedelta(days=contract.payment_terms_days or 0),
            po_number=po.po_number,
            currency=vendor.currency,
            subtotal=subtotal,
            tax=tax,
            total=total,
            line_items=lines,
        )
        invoice_id = f"SINV-{i:06d}"
        ledger.purchase_orders.append(LinkedPO(po, vendor.vendor_id, number, [invoice_id]))
        ledger.invoices.append(LinkedInvoice(invoice_id, invoice, vendor.vendor_id, number))
    return ledger
