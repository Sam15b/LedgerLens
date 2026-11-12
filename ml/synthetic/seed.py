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

    python -m ml.synthetic.seed --cord data/raw/cord --out data/synthetic/seed
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
from decimal import Decimal
from pathlib import Path
from typing import Any

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

    records: list[SeedRecord] = []
    if (args.cord / "data").is_dir():
        records += cord_records(read_cord(args.cord))
    if args.sroie.is_dir():
        records += sroie_records(args.sroie)
    if not records:
        raise SystemExit(f"no seed sources found under {args.cord} or {args.sroie}")
    write_csv(records, args.out / "seed_invoices.csv")
    by_source: dict[str, int] = {}
    for r in records:
        by_source[r.source] = by_source.get(r.source, 0) + 1
    print(f"{len(records)} seed records {by_source} -> {args.out / 'seed_invoices.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
