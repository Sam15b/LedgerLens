"""Build the synthetic ledger: ``data/synthetic/records.jsonl`` with ground-truth links.

    python -m ml.synthetic.build --checkpoint ml/synthetic/gan/runs/<run>/final.pt \\
        --invoices 5000 --out data/synthetic/records.jsonl

Steps: sample invoice rows -> contracts + one PO per invoice (:mod:`ml.synthetic.link`)
-> labelled anomalies (:mod:`ml.synthetic.anomalies`) -> payments. Where the rows come
from, in order of preference:

1. ``--checkpoint``: a trained GAN (the conditional one samples the seed category mix);
2. ``--rows``: a sampled-rows CSV (``python -m ml.synthetic.gan.sample``);
3. otherwise the seed rows themselves, resampled with replacement (no GAN needed --
   handy for tests and a first look, but every row repeats a real one).

Each line of ``records.jsonl`` is one record::

    {"record_type": "vendor" | "contract" | "purchase_order" | "invoice" | "payment",
     "id": ..., "data": {...the schema model as JSON...}, ...links and labels...}

Invoices carry ``vendor_id``, ``contract_number``, ``po_number``, ``payment_status``,
``anomalies`` (``[]`` when clean), ``duplicate_of`` and ``details``. A summary with counts
and rates is written next to it (``records.summary.json``).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.synthetic.anomalies import ANOMALY_RATES, inject_anomalies, plan_for_total
from ml.synthetic.link import Ledger, link_rows, make_contracts, make_payments
from ml.synthetic.seed import SEED, load_vendor_master, read_csv

OUT = Path("data/synthetic/records.jsonl")
SEED_CSV = Path("data/synthetic/seed/seed_invoices.csv")


@dataclass
class BuildResult:
    ledger: Ledger
    source: str
    summary: dict[str, Any]


def rows_from_checkpoint(path: Path, n: int, *, seed: int, seed_csv: Path) -> list[dict[str, Any]]:
    from ml.synthetic.gan.sample import category_mix, sample_rows
    from ml.synthetic.gan.train import load_generator, load_seed_rows

    generator, transformer = load_generator(path)
    categories = None
    if generator.cond_dim:
        categories = category_mix(load_seed_rows(seed_csv)[0], n, seed=seed)
    return sample_rows(generator, transformer, n, categories=categories, seed=seed)


def resample(rows: Sequence[Mapping[str, Any]], n: int, *, seed: int) -> list[dict[str, Any]]:
    """``n`` rows: all of ``rows`` (in order) if there are enough, else drawn with replacement."""
    if len(rows) >= n:
        return [dict(r) for r in rows[:n]]
    return [dict(r) for r in random.Random(seed).choices(rows, k=n)]


def build(
    rows: Sequence[Mapping[str, Any]],
    *,
    invoices: int,
    seed: int = SEED,
    source: str = "rows",
    rates: Mapping[str, float] = ANOMALY_RATES,
) -> BuildResult:
    """A complete ledger with ``invoices`` invoices from (at least ``plan.base``) ``rows``."""
    plan = plan_for_total(invoices, rates)
    vendors = load_vendor_master()
    ledger = link_rows(resample(rows, plan.base, seed=seed), vendors, make_contracts(vendors))
    inject_anomalies(ledger, plan, seed=seed)
    make_payments(ledger, seed=seed)
    return BuildResult(ledger, source, summarize(ledger, source=source))


def summarize(ledger: Ledger, *, source: str) -> dict[str, Any]:
    n = len(ledger.invoices)
    labels = Counter(a for i in ledger.invoices for a in i.anomalies)
    return {
        "source": source,
        "counts": {
            "vendors": len(ledger.vendors),
            "contracts": len(ledger.contracts),
            "purchase_orders": len(ledger.purchase_orders),
            "invoices": n,
            "payments": len(ledger.payments),
        },
        "anomalies": {k: {"count": labels[k], "rate": labels[k] / n} for k in ANOMALY_RATES},
        "clean_invoices": sum(1 for i in ledger.invoices if not i.anomalies),
        "payment_status": dict(Counter(str(i.payment_status) for i in ledger.invoices)),
        "currencies": dict(Counter(i.invoice.currency for i in ledger.invoices)),
    }


def write_records(ledger: Ledger, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = ledger.records()
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def read_records(path: Path, record_type: str | None = None) -> Iterator[dict[str, Any]]:
    """Records from ``records.jsonl``, optionally only one ``record_type``."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record: dict[str, Any] = json.loads(line)
                if record_type is None or record["record_type"] == record_type:
                    yield record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build data/synthetic/records.jsonl")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path, help="sample rows from this GAN checkpoint")
    source.add_argument("--rows", type=Path, help="a sampled-rows CSV")
    parser.add_argument("--invoices", type=int, default=5000, help="invoices in the ledger")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--seed-csv", type=Path, default=SEED_CSV)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    base = plan_for_total(args.invoices).base
    if args.checkpoint:
        rows = rows_from_checkpoint(args.checkpoint, base, seed=args.seed, seed_csv=args.seed_csv)
        label = f"gan:{args.checkpoint.as_posix()}"
    elif args.rows:
        rows = [r.features() for r in read_csv(args.rows)]
        label = f"rows:{args.rows.as_posix()}"
    else:
        from ml.synthetic.gan.train import load_seed_rows  # vendor-master fallback built in

        rows = load_seed_rows(args.seed_csv)[0]
        label = "seed rows resampled (no GAN)"
    result = build(rows, invoices=args.invoices, seed=args.seed, source=label)
    n = write_records(result.ledger, args.out)
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(result.summary, indent=1) + "\n", encoding="utf-8")
    counts = result.summary["counts"]
    print(f"{n} records ({counts}) -> {args.out}")
    for kind, stats in result.summary["anomalies"].items():
        print(f"  {kind:<16} {stats['count']:>4}  {stats['rate']:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
