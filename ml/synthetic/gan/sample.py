"""Sample synthetic invoice rows from a trained generator.

    python -m ml.synthetic.gan.sample ml/synthetic/gan/runs/<run>/final.pt -n 5000 \\
        --out data/synthetic/gan_samples.csv

A generator emits each column independently, so a raw row need not add up: its ``total``
is rarely exactly ``subtotal x (1 + tax_rate)``. Every row is therefore post-processed
with fixed business rules (:func:`postprocess`):

* integers stay in their valid range (``line_count >= 1``, ``days_to_due >= 0``,
  ``1 <= day_of_month <= 31``) and ``0 <= tax_rate <= 30 %`` (rounded to 4 places);
* money is rounded to the currency's smallest step (whole hundreds for IDR, cents
  otherwise), and is never below one step;
* ``total`` is **recomputed** from the rounded subtotal and tax rate, never sampled.

:func:`rule_violations` checks the same rules, so callers (and tests) can confirm that no
sampled row breaks them. Rows come out as :class:`~ml.synthetic.seed.SeedRecord` with
``source="gan"``, so the seed CSV reader and writer work on samples too.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # run as a script: make the repo root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from ml.synthetic.gan.dataset import TabularTransformer
from ml.synthetic.gan.model import Generator
from ml.synthetic.gan.train import SEED_CSV, condition_vector, load_generator, load_seed_rows
from ml.synthetic.seed import CENTS, MAX_TAX_RATE, SeedRecord, write_csv

DEFAULT_STEP = Decimal("0.01")
AMOUNTS = ("unit_price_mean", "subtotal", "total")
CONDITION_COLUMN = "vendor_category"


def money_step(currency: str) -> Decimal:
    """Smallest amount a document in ``currency`` shows (IDR: whole hundreds)."""
    return CENTS.get(currency, DEFAULT_STEP)


def round_money(value: float | Decimal, currency: str) -> Decimal:
    """``value`` rounded half-up to the currency step, and at least one step."""
    step = money_step(currency)
    steps = (Decimal(str(value)) / step).quantize(Decimal(1), rounding=ROUND_HALF_UP)
    return max(steps, Decimal(1)) * step


def expected_total(subtotal: float | Decimal, tax_rate: float, currency: str) -> Decimal:
    """The business rule: ``total = subtotal x (1 + tax_rate)``, rounded to the currency."""
    gross = Decimal(str(subtotal)) * (Decimal(1) + Decimal(str(tax_rate)))
    return round_money(gross, currency)


def postprocess(row: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the business rules (module docstring) to one raw generated row."""
    currency = str(row["currency"])
    tax_rate = round(min(max(float(row["tax_rate"]), 0.0), MAX_TAX_RATE), 4)
    subtotal = round_money(float(row["subtotal"]), currency)
    return {
        "line_count": max(1, int(row["line_count"])),
        "qty_mean": round(max(float(row["qty_mean"]), 0.01), 4),
        "unit_price_mean": float(round_money(float(row["unit_price_mean"]), currency)),
        "subtotal": float(subtotal),
        "tax_rate": tax_rate,
        "total": float(expected_total(subtotal, tax_rate, currency)),
        "days_to_due": max(0, int(row["days_to_due"])),
        "day_of_month": min(max(int(row["day_of_month"]), 1), 31),
        "vendor_category": str(row["vendor_category"]),
        "currency": currency,
    }


def rule_violations(row: Mapping[str, Any]) -> list[str]:
    """Business rules ``row`` breaks; empty when the row is valid."""
    issues: list[str] = []
    currency = str(row["currency"])
    step = money_step(currency)
    for name in AMOUNTS:
        amount = Decimal(str(row[name]))
        if amount < step:
            issues.append(f"{name} {amount} is below one {currency} step ({step})")
        elif amount % step:
            issues.append(f"{name} {amount} is not a multiple of {step} {currency}")
    if not 0.0 <= float(row["tax_rate"]) <= MAX_TAX_RATE:
        issues.append(f"tax_rate {row['tax_rate']} outside [0, {MAX_TAX_RATE}]")
    expected = expected_total(row["subtotal"], float(row["tax_rate"]), currency)
    if Decimal(str(row["total"])) != expected:
        issues.append(f"total {row['total']} != subtotal x (1 + tax_rate) = {expected}")
    if int(row["line_count"]) < 1:
        issues.append(f"line_count {row['line_count']} < 1")
    if float(row["qty_mean"]) <= 0:
        issues.append(f"qty_mean {row['qty_mean']} <= 0")
    if int(row["days_to_due"]) < 0:
        issues.append(f"days_to_due {row['days_to_due']} < 0")
    if not 1 <= int(row["day_of_month"]) <= 31:
        issues.append(f"day_of_month {row['day_of_month']} outside 1-31")
    return issues


def sample_rows(
    generator: Generator,
    transformer: TabularTransformer,
    n: int,
    *,
    categories: Sequence[str] | None = None,
    seed: int = 0,
    batch_size: int = 2048,
) -> list[dict[str, Any]]:
    """``n`` post-processed rows.

    A conditional generator needs ``categories``: one ``vendor_category`` per row (or a
    single one for all rows). Sampling is reproducible for a given ``seed``.
    """
    if generator.cond_dim and not categories:
        raise ValueError("a conditional generator needs categories to sample")
    if not generator.cond_dim and categories:
        raise ValueError("this generator is not conditional; drop categories")
    if categories is not None and len(categories) not in (1, n):
        raise ValueError(f"need 1 or {n} categories, got {len(categories)}")
    torch.manual_seed(seed)  # the Gumbel-softmax heads draw from the global generator
    noise_rng = torch.Generator().manual_seed(seed)
    rows: list[dict[str, Any]] = []
    for start in range(0, n, batch_size):
        size = min(batch_size, n - start)
        condition = None
        if categories is not None:
            if len(categories) == 1:
                chosen = list(categories) * size
            else:
                chosen = list(categories[start : start + size])
            condition = torch.stack(
                [condition_vector(transformer, CONDITION_COLUMN, c) for c in chosen]
            )
        matrix = generator.sample(size, condition, generator=noise_rng)
        rows += [postprocess(row) for row in transformer.inverse_transform(matrix.cpu().numpy())]
    return rows


def category_mix(
    rows: Sequence[Mapping[str, Any]], n: int, *, seed: int = 0, column: str = CONDITION_COLUMN
) -> list[str]:
    """``n`` categories drawn in the proportions they have in ``rows`` (for conditional
    sampling that should match the real mix)."""
    counts = Counter(str(row[column]) for row in rows)
    names = sorted(counts)
    return random.Random(seed).choices(names, weights=[counts[c] for c in names], k=n)


def to_records(rows: Sequence[Mapping[str, Any]], *, prefix: str = "gan") -> list[SeedRecord]:
    return [
        SeedRecord(**dict(row), source="gan", source_id=f"{prefix}-{i:06d}")
        for i, row in enumerate(rows, start=1)
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sample rows from a trained tabular GAN")
    parser.add_argument("checkpoint", type=Path, help="final.pt / checkpoint_*.pt")
    parser.add_argument("-n", type=int, default=5000, help="rows to sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--category",
        help="conditional GAN: sample only this vendor_category (default: the "
        "seed CSV's category mix)",
    )
    parser.add_argument("--seed-csv", type=Path, default=SEED_CSV)
    parser.add_argument("--out", type=Path, default=Path("data/synthetic/gan_samples.csv"))
    args = parser.parse_args(argv)

    generator, transformer = load_generator(args.checkpoint)
    categories: list[str] | None = None
    if generator.cond_dim:
        if args.category:
            categories = [args.category]
        else:
            categories = category_mix(load_seed_rows(args.seed_csv)[0], args.n, seed=args.seed)
    rows = sample_rows(generator, transformer, args.n, categories=categories, seed=args.seed)
    bad = sum(1 for row in rows if rule_violations(row))
    write_csv(to_records(rows), args.out)
    print(f"{len(rows)} rows ({bad} breaking business rules) -> {args.out}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
