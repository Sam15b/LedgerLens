"""Seed records for the tabular GAN: one flat row per real (or hand-made) document.

Every row has the GAN's features (``docs/layers/07-synthetic-data.md``):

==================  ===========  ==============================================
column              kind         meaning
==================  ===========  ==============================================
line_count          integer      item lines on the document
qty_mean            continuous   mean quantity per line
unit_price_mean     amount       mean unit price
subtotal            amount       before tax
tax_rate            continuous   tax / subtotal (0 when the document shows none)
total               amount       payable
days_to_due         integer      due date - issue date (0 for paid receipts)
day_of_month        integer      issue date's day
vendor_category     categorical  what the vendor sells
currency            categorical  ISO code
==================  ===========  ==============================================

plus ``source``, ``source_id`` and ``imputed`` (columns not read from the document but
filled by a documented rule) for provenance.

Sources:

* **CORD** (``data/raw/cord``): ~1,000 Indonesian receipts with item lines, subtotal, tax
  and total (IDR). They carry no date and no vendor category: receipts are paid on the
  spot (``days_to_due = 0``), the vendor category is ``food_beverage`` (CORD is restaurant
  and cafe receipts), and ``day_of_month`` is drawn uniformly from 1-28 with a fixed seed
  -- each marked as imputed.
* **SROIE** (``data/raw/sroie``, if downloaded): Malaysian receipts with date and total but
  no item lines.
* **Seed POs** from the hand-made vendor master (``vendor_master.csv``: 50 fictional
  vendors in 10 categories and 7 countries, with tax ids and payment terms): B2B orders
  from Northwind Traders priced from a per-category catalog, each billed by one invoice due
  on the vendor's terms. Receipts alone would teach the GAN that every document is a
  paid-on-the-spot cafe bill.

    python -m ml.synthetic.seed --cord data/raw/cord --out data/synthetic/seed

writes ``seed_invoices.csv`` (all seed rows) and ``seed_pos.json`` (the POs, as
``PurchaseOrder`` records, for linking invoices to POs later).
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import random
import re
import statistics
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from ledgerlens.core.schemas import LineItem, PurchaseOrder
from ledgerlens.extraction.normalize import parse_amount, parse_date

SEED = 20261112
FIELDS = (
    "line_count",
    "qty_mean",
    "unit_price_mean",
    "subtotal",
    "tax_rate",
    "total",
    "days_to_due",
    "day_of_month",
    "vendor_category",
    "currency",
)


@dataclass
class SeedRecord:
    line_count: int
    qty_mean: float
    unit_price_mean: float
    subtotal: float
    tax_rate: float
    total: float
    days_to_due: int
    day_of_month: int
    vendor_category: str
    currency: str
    source: str
    source_id: str
    imputed: list[str] = field(default_factory=list)

    def features(self) -> dict[str, Any]:
        """Just the GAN's columns."""
        return {name: getattr(self, name) for name in FIELDS}


# --------------------------------------------------------------------------- CORD

_COUNT = re.compile(r"\d+(?:[.,]\d+)?")


def _amount(value: Any) -> Decimal | None:
    """A CORD price ("16,500", "23.000", "-1,000") as a Decimal; lists take the first."""
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return None
    return parse_amount(str(value))


def _count(value: Any) -> float | None:
    if isinstance(value, list):
        value = value[0] if value else None
    match = _COUNT.search(str(value)) if value is not None else None
    if not match:
        return None
    count = float(match.group(0).replace(",", "."))
    return count if count > 0 else None


def cord_record(
    gt_parse: Mapping[str, Any], source_id: str, rng: random.Random
) -> SeedRecord | None:
    """One seed row from a CORD ``gt_parse`` dict, or ``None`` if it has no usable total."""
    menu = gt_parse.get("menu") or []
    items = [menu] if isinstance(menu, dict) else [m for m in menu if isinstance(m, dict)]
    totals = gt_parse.get("total") or {}
    sub = gt_parse.get("sub_total") or {}
    total = _amount(totals.get("total_price")) if isinstance(totals, dict) else None
    if total is None or total <= 0 or not items:
        return None
    counts, unit_prices, line_amounts = [], [], []
    for item in items:
        count = _count(item.get("cnt")) or 1.0
        price = _amount(item.get("price"))
        unit = _amount(item.get("unitprice"))
        if unit is None and price is not None:
            unit = price / Decimal(str(count))
        counts.append(count)
        if unit is not None and unit > 0:
            unit_prices.append(float(unit))
        if price is not None:
            line_amounts.append(price)
    if not unit_prices:
        return None
    subtotal = _amount(sub.get("subtotal_price")) if isinstance(sub, dict) else None
    tax = _amount(sub.get("tax_price")) if isinstance(sub, dict) else None
    imputed = ["days_to_due", "day_of_month", "vendor_category"]
    if subtotal is None or subtotal <= 0:
        subtotal = sum(line_amounts, Decimal(0)) or total
        imputed.append("subtotal")
    tax_rate = float(tax / subtotal) if tax is not None and tax >= 0 and subtotal > 0 else 0.0
    return SeedRecord(
        line_count=len(items),
        qty_mean=statistics.fmean(counts),
        unit_price_mean=statistics.fmean(unit_prices),
        subtotal=float(subtotal),
        tax_rate=round(tax_rate, 4),
        total=float(total),
        days_to_due=0,
        day_of_month=rng.randint(1, 28),
        vendor_category="food_beverage",
        currency="IDR",
        source="cord",
        source_id=source_id,
        imputed=imputed,
    )


def read_cord(
    root: Path, *, splits: Sequence[str] = ("train", "validation", "test")
) -> Iterator[tuple[str, dict[str, Any]]]:
    """``(id, gt_parse)`` for every CORD receipt; reads only the ground-truth column."""
    parquet = importlib.import_module("pyarrow.parquet")
    for split in splits:
        for path in sorted((root / "data").glob(f"{split}-*.parquet")):
            table = parquet.read_table(path, columns=["ground_truth"])
            for i, row in enumerate(table.column("ground_truth").to_pylist()):
                gt = json.loads(row).get("gt_parse", {})
                yield f"{split}/{path.stem.split('-')[1]}/{i}", gt


MAX_TAX_RATE = 0.3
TOTAL_TO_SUBTOTAL = (0.5, 1.5)


def is_plausible(record: SeedRecord) -> bool:
    """Drop rows whose annotation is internally inconsistent (a misread or mislabelled
    amount): tax rate above 30 %, or a total far from the subtotal. A GAN would learn
    these as real modes."""
    low, high = TOTAL_TO_SUBTOTAL
    ratio = record.total / record.subtotal if record.subtotal > 0 else 0.0
    return 0.0 <= record.tax_rate <= MAX_TAX_RATE and low <= ratio <= high


def cord_records(
    rows: Iterable[tuple[str, Mapping[str, Any]]], *, seed: int = SEED
) -> list[SeedRecord]:
    rng = random.Random(seed)
    records = [cord_record(gt, source_id, rng) for source_id, gt in rows]
    records = [r for r in records if r is None or is_plausible(r)]
    return [r for r in records if r is not None]


# --------------------------------------------------------------------------- SROIE

_SROIE_CATEGORIES = [
    (re.compile(r"hardware|trading|enterprise|machinery|electrical", re.I), "hardware"),
    (re.compile(r"book|stationery|printing|paper", re.I), "office_supplies"),
    (
        re.compile(r"restaurant|cafe|kopi|food|bakery|kitchen|noodle|mart|bistro", re.I),
        "food_beverage",
    ),
]


def sroie_record(sample_id: str, fields: Mapping[str, str]) -> SeedRecord | None:
    total = parse_amount(fields.get("total", ""))
    day = parse_date(fields.get("date", ""), day_first=True)
    if total is None or total <= 0 or day is None:
        return None
    company = fields.get("company", "")
    category = next((c for p, c in _SROIE_CATEGORIES if p.search(company)), "retail")
    return SeedRecord(
        line_count=1,
        qty_mean=1.0,
        unit_price_mean=float(total),
        subtotal=float(total),
        tax_rate=0.0,
        total=float(total),
        days_to_due=0,
        day_of_month=day.day,
        vendor_category=category,
        currency="MYR",
        source="sroie",
        source_id=sample_id,
        imputed=["line_count", "qty_mean", "unit_price_mean", "subtotal", "tax_rate"],
    )


def sroie_records(root: Path, split: str = "train") -> list[SeedRecord]:
    from eval.datasets.sroie import iter_sroie

    records = (sroie_record(s.sample_id, s.fields) for s in iter_sroie(root, split))
    return [r for r in records if r is not None]


# --------------------------------------------------------------------------- vendor master + POs

VENDOR_MASTER = Path(__file__).with_name("vendor_master.csv")
BUYER = "Northwind Traders"

# What each category sells: (item, unit, low, high unit price in USD).
CATALOG: dict[str, list[tuple[str, str, float, float]]] = {
    "packaging": [
        ("Pallet wrap 500m", "roll", 30, 60),
        ("Shipping labels (box of 500)", "box", 20, 45),
        ("Corrugated boxes 40x30x30", "bundle", 25, 70),
        ("Packing tape 48mm", "case", 35, 80),
    ],
    "office_supplies": [
        ("A4 copy paper (box of 5 reams)", "box", 18, 35),
        ("Toner cartridge", "each", 45, 140),
        ("Ballpoint pens (pack of 50)", "pack", 8, 20),
        ("Archive files", "pack", 10, 30),
    ],
    "logistics": [
        ("Courier", "hour", 30, 60),
        ("Pallet delivery", "pallet", 60, 180),
        ("Same-day dispatch", "consignment", 25, 90),
        ("Fuel surcharge", "each", 5, 40),
    ],
    "it_services": [
        ("Cloud hosting", "month", 200, 2500),
        ("Support hours", "hour", 70, 180),
        ("Software licence", "seat", 15, 90),
        ("Laptop setup", "each", 60, 200),
    ],
    "food_beverage": [
        ("Coffee beans 1kg", "bag", 15, 40),
        ("Catering lunch", "person", 10, 35),
        ("Bottled water (case of 24)", "case", 6, 15),
        ("Pastry platter", "platter", 25, 70),
    ],
    "facilities": [
        ("Office cleaning", "visit", 80, 300),
        ("HVAC maintenance", "visit", 150, 600),
        ("Light fittings", "each", 20, 90),
        ("Waste collection", "month", 60, 250),
    ],
    "professional_services": [
        ("Consulting", "day", 600, 1800),
        ("Audit fieldwork", "day", 700, 1500),
        ("Legal review", "hour", 180, 450),
        ("Tax filing", "each", 300, 1200),
    ],
    "hardware": [
        ("Steel bolts M10 (box of 100)", "box", 12, 30),
        ("Cordless drill", "each", 90, 250),
        ("Safety gloves (pack of 12)", "pack", 15, 40),
        ("Extension cable 10m", "each", 12, 35),
    ],
    "utilities": [
        ("Electricity", "month", 300, 3000),
        ("Water and sewerage", "month", 80, 600),
        ("Business broadband", "month", 40, 250),
    ],
    "marketing": [
        ("Campaign design", "day", 400, 1200),
        ("Printed brochures (500)", "batch", 150, 600),
        ("Social media management", "month", 500, 2500),
        ("Event banner", "each", 60, 300),
    ],
}
UNIT_QTY = {  # (low, high) quantity per line by unit
    "month": (1, 1),
    "day": (1, 10),
    "hour": (2, 40),
    "visit": (1, 4),
    "person": (10, 80),
    "seat": (5, 50),
    "each": (1, 12),
    "consignment": (1, 6),
    "pallet": (1, 8),
}
# Currency per unit of USD (rough, fixed: the GAN needs realistic magnitudes, not FX).
FX = {"USD": 1.0, "GBP": 0.79, "EUR": 0.92, "SGD": 1.34, "MYR": 4.45, "IDR": 15800.0}
VAT = {
    "GB": 0.20,
    "DE": 0.19,
    "FR": 0.20,
    "NL": 0.21,
    "MY": 0.08,
    "SG": 0.09,
    "ID": 0.11,
    "US": 0.0,
}
CENTS = {"IDR": Decimal("100")}  # IDR prices are whole hundreds


@dataclass(frozen=True)
class Vendor:
    vendor_id: str
    name: str
    category: str
    country: str
    currency: str
    tax_id: str
    payment_terms_days: int


def load_vendor_master(path: Path = VENDOR_MASTER) -> list[Vendor]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [
            Vendor(
                vendor_id=row["vendor_id"],
                name=row["name"],
                category=row["category"],
                country=row["country"],
                currency=row["currency"],
                tax_id=row["tax_id"],
                payment_terms_days=int(row["payment_terms_days"]),
            )
            for row in csv.DictReader(fh)
        ]


def _money(value: float, currency: str) -> Decimal:
    step = CENTS.get(currency, Decimal("0.01"))
    return (Decimal(str(value)) / step).quantize(Decimal(1)) * step


def seed_purchase_orders(
    vendors: Sequence[Vendor],
    *,
    per_vendor: int = 10,
    seed: int = SEED,
    start: date = date(2025, 1, 6),
    days: int = 640,
) -> list[PurchaseOrder]:
    """Hand-made-style POs from Northwind Traders to every vendor (deterministic)."""
    rng = random.Random(seed)
    orders: list[PurchaseOrder] = []
    for vendor in vendors:
        catalog = CATALOG[vendor.category]
        for _ in range(per_vendor):
            ordered = start + timedelta(days=rng.randrange(days))
            while ordered.weekday() >= 5:  # POs are raised on business days
                ordered += timedelta(days=1)
            lines = []
            for item, unit, low, high in rng.sample(catalog, rng.randint(1, len(catalog))):
                qty_low, qty_high = UNIT_QTY.get(unit, (1, 20))
                qty = rng.randint(qty_low, qty_high)
                unit_price = _money(rng.uniform(low, high) * FX[vendor.currency], vendor.currency)
                lines.append(
                    LineItem(
                        description=item,
                        quantity=Decimal(qty),
                        unit_price=unit_price,
                        amount=unit_price * qty,
                    )
                )
            subtotal = sum((line.amount for line in lines), Decimal(0))
            tax = _money(float(subtotal) * VAT[vendor.country], vendor.currency)
            orders.append(
                PurchaseOrder(
                    po_number=f"PO-{len(orders) + 1:05d}",
                    vendor_name=vendor.name,
                    vendor_tax_id=vendor.tax_id,
                    buyer_name=BUYER,
                    order_date=ordered,
                    delivery_date=ordered + timedelta(days=rng.randint(2, 21)),
                    currency=vendor.currency,
                    subtotal=subtotal,
                    tax=tax,
                    total=subtotal + tax,
                    line_items=lines,
                )
            )
    return orders


def po_invoice_records(
    orders: Sequence[PurchaseOrder], vendors: Sequence[Vendor], *, seed: int = SEED
) -> list[SeedRecord]:
    """One seed invoice per PO: billed on delivery, due on the vendor's payment terms."""
    by_name = {v.name: v for v in vendors}
    rng = random.Random(seed + 1)
    records = []
    for po in orders:
        vendor = by_name[po.vendor_name]
        invoiced = (po.delivery_date or po.order_date) + timedelta(days=rng.randint(0, 5))
        quantities = [float(line.quantity or 1) for line in po.line_items]
        prices = [float(line.unit_price or line.amount) for line in po.line_items]
        records.append(
            SeedRecord(
                line_count=len(po.line_items),
                qty_mean=statistics.fmean(quantities),
                unit_price_mean=statistics.fmean(prices),
                subtotal=float(po.subtotal or po.total),
                tax_rate=VAT[vendor.country],
                total=float(po.total),
                days_to_due=vendor.payment_terms_days,
                day_of_month=invoiced.day,
                vendor_category=vendor.category,
                currency=po.currency,
                source="seed_po",
                source_id=po.po_number,
            )
        )
    return records


def write_purchase_orders(orders: Sequence[PurchaseOrder], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([po.model_dump(mode="json") for po in orders], indent=1), encoding="utf-8"
    )


# --------------------------------------------------------------------------- output

COLUMNS = [f.name for f in fields(SeedRecord)]


def write_csv(records: Sequence[SeedRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["imputed"] = ";".join(record.imputed)
            writer.writerow(row)


def read_csv(path: Path) -> list[SeedRecord]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [
            SeedRecord(
                line_count=int(row["line_count"]),
                qty_mean=float(row["qty_mean"]),
                unit_price_mean=float(row["unit_price_mean"]),
                subtotal=float(row["subtotal"]),
                tax_rate=float(row["tax_rate"]),
                total=float(row["total"]),
                days_to_due=int(row["days_to_due"]),
                day_of_month=int(row["day_of_month"]),
                vendor_category=row["vendor_category"],
                currency=row["currency"],
                source=row["source"],
                source_id=row["source_id"],
                imputed=[c for c in row["imputed"].split(";") if c],
            )
            for row in csv.DictReader(fh)
        ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build GAN seed records")
    parser.add_argument("--cord", type=Path, default=Path("data/raw/cord"))
    parser.add_argument("--sroie", type=Path, default=Path("data/raw/sroie"))
    parser.add_argument("--out", type=Path, default=Path("data/synthetic/seed"))
    args = parser.parse_args(argv)

    vendors = load_vendor_master()
    orders = seed_purchase_orders(vendors)
    records = po_invoice_records(orders, vendors)
    if (args.cord / "data").is_dir():
        records += cord_records(read_cord(args.cord))
    if args.sroie.is_dir():
        records += sroie_records(args.sroie)
    write_csv(records, args.out / "seed_invoices.csv")
    write_purchase_orders(orders, args.out / "seed_pos.json")
    by_source: dict[str, int] = {}
    for r in records:
        by_source[r.source] = by_source.get(r.source, 0) + 1
    print(f"{len(records)} seed records {by_source} -> {args.out / 'seed_invoices.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
