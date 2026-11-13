import json
import random
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from ml.synthetic.gan import evaluate as gan_eval
from ml.synthetic.gan import sample as gan_sample
from ml.synthetic.gan.dataset import SeedDataset, TabularTransformer
from ml.synthetic.gan.evaluate import (
    baseline,
    compare,
    ks_statistic,
    render_markdown,
    total_variation,
    wasserstein,
)
from ml.synthetic.gan.model import Generator
from ml.synthetic.gan.sample import (
    category_mix,
    expected_total,
    postprocess,
    round_money,
    rule_violations,
    sample_rows,
    to_records,
)
from ml.synthetic.gan.train import TrainConfig, load_generator, train
from ml.synthetic.seed import read_csv, write_csv

CATEGORIES = ["logistics", "marketing", "packaging"]


def seed_rows(n: int = 120, seed: int = 3) -> list[dict[str, Any]]:
    """Rows shaped like the real seed table, which already obey the business rules."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        currency = rng.choice(["GBP", "IDR", "USD"])
        tax_rate = rng.choice([0.0, 0.11, 0.2])
        subtotal = round_money(10 ** rng.uniform(1, 6), currency)
        out.append(
            {
                "line_count": rng.randint(1, 12),
                "qty_mean": round(rng.uniform(1, 20), 2),
                "unit_price_mean": float(round_money(10 ** rng.uniform(0, 4), currency)),
                "subtotal": float(subtotal),
                "tax_rate": tax_rate,
                "total": float(expected_total(subtotal, tax_rate, currency)),
                "days_to_due": rng.choice([0, 14, 30, 60]),
                "day_of_month": rng.randint(1, 31),
                "vendor_category": rng.choice(CATEGORIES),
                "currency": currency,
            }
        )
    return out


def untrained(transformer: TabularTransformer, *, conditional: bool = False) -> Generator:
    """A random generator: its raw rows are as far from valid as samples get."""
    torch.manual_seed(0)
    span = transformer.column("vendor_category")
    return Generator(
        transformer.output_dim,
        transformer.categorical_spans,
        noise_dim=16,
        hidden=(32, 32),
        condition_span=(span.start, span.width) if conditional else None,
    )


@pytest.fixture
def transformer() -> TabularTransformer:
    return TabularTransformer().fit(seed_rows())


# --------------------------------------------------------------------------- business rules


def test_round_money_uses_the_currency_step_and_never_rounds_to_zero() -> None:
    assert str(round_money(12.345, "USD")) == "12.35"  # half-up
    assert round_money(15_849, "IDR") == 15_800
    assert round_money(15_850, "IDR") == 15_900
    assert str(round_money(0.001, "GBP")) == "0.01"
    assert round_money(3, "IDR") == 100


def test_postprocess_recomputes_total_and_clamps_ranges() -> None:
    raw = {
        "line_count": 0,
        "qty_mean": -2.0,
        "unit_price_mean": 9.999,
        "subtotal": 100.004,
        "tax_rate": 0.45,
        "total": 1.0,  # sampled totals are ignored
        "days_to_due": -4,
        "day_of_month": 40,
        "vendor_category": "logistics",
        "currency": "GBP",
    }

    row = postprocess(raw)

    assert row["line_count"] == 1
    assert row["qty_mean"] > 0
    assert row["tax_rate"] == 0.3
    assert row["subtotal"] == 100.0
    assert row["total"] == 130.0
    assert (row["days_to_due"], row["day_of_month"]) == (0, 31)
    assert rule_violations(row) == []


def test_rule_violations_name_each_broken_rule() -> None:
    good = postprocess(seed_rows(1)[0])
    assert rule_violations(good) == []

    bad = {**good, "total": good["total"] + 1, "tax_rate": -0.1, "line_count": 0}
    issues = rule_violations(bad)
    assert any(i.startswith("total") for i in issues)
    assert any(i.startswith("tax_rate") for i in issues)
    assert any(i.startswith("line_count") for i in issues)
    assert any("not a multiple" in i for i in rule_violations({**good, "subtotal": 10.005}))


# --------------------------------------------------------------------------- sampling


def test_no_sampled_row_breaks_the_business_rules(transformer: TabularTransformer) -> None:
    rows = sample_rows(untrained(transformer), transformer, 500, seed=1, batch_size=128)

    assert len(rows) == 500
    assert [r for r in rows if rule_violations(r)] == []
    assert {r["vendor_category"] for r in rows} <= set(CATEGORIES)
    seen_ranges = {c.name: (c.low, c.high) for c in transformer.columns if c.kind == "integer"}
    low, high = seen_ranges["line_count"]
    assert all(low <= r["line_count"] <= high for r in rows)


def test_sampling_is_reproducible_per_seed(transformer: TabularTransformer) -> None:
    generator = untrained(transformer)

    first = sample_rows(generator, transformer, 50, seed=7)

    assert sample_rows(generator, transformer, 50, seed=7) == first
    assert sample_rows(generator, transformer, 50, seed=8) != first


def test_conditional_sampling_honours_the_requested_categories(
    transformer: TabularTransformer,
) -> None:
    generator = untrained(transformer, conditional=True)
    wanted = ["marketing", "packaging", "logistics"] * 20

    rows = sample_rows(generator, transformer, 60, categories=wanted, seed=2, batch_size=16)
    assert [r["vendor_category"] for r in rows] == wanted

    only = sample_rows(generator, transformer, 10, categories=["logistics"])
    assert {r["vendor_category"] for r in only} == {"logistics"}

    with pytest.raises(ValueError, match="needs categories"):
        sample_rows(generator, transformer, 5)
    with pytest.raises(ValueError, match="need 1 or 5"):
        sample_rows(generator, transformer, 5, categories=["logistics", "marketing"])
    with pytest.raises(ValueError, match="not conditional"):
        sample_rows(untrained(transformer), transformer, 5, categories=["logistics"])


def test_category_mix_follows_the_real_proportions() -> None:
    real = [{"vendor_category": "a"}] * 300 + [{"vendor_category": "b"}] * 100

    mix = Counter(category_mix(real, 4000, seed=1))

    assert set(mix) == {"a", "b"}
    assert mix["a"] / 4000 == pytest.approx(0.75, abs=0.03)


def test_samples_round_trip_through_the_seed_csv(
    transformer: TabularTransformer, tmp_path: Path
) -> None:
    rows = sample_rows(untrained(transformer), transformer, 20)

    write_csv(to_records(rows), tmp_path / "samples.csv")
    back = read_csv(tmp_path / "samples.csv")

    assert [r.features() for r in back] == rows
    assert back[0].source == "gan" and back[0].source_id == "gan-000001"
    assert all(rule_violations(r.features()) == [] for r in back)


def test_sample_cli_on_a_trained_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = SeedDataset.from_rows(seed_rows(), condition="vendor_category")
    config = TrainConfig(
        steps=2,
        batch_size=32,
        n_critic=1,
        hidden=(16, 16),
        noise_dim=8,
        condition="vendor_category",
    )
    train(dataset, config, run_dir=tmp_path / "run")
    checkpoint = tmp_path / "run" / "final.pt"
    write_csv(to_records(seed_rows()), tmp_path / "seed.csv")
    out = tmp_path / "samples.csv"

    code = gan_sample.main(
        [str(checkpoint), "-n", "40", "--seed-csv", str(tmp_path / "seed.csv"), "--out", str(out)]
    )

    assert code == 0
    assert "40 rows (0 breaking business rules)" in capsys.readouterr().out
    assert len(read_csv(out)) == 40
    generator, _ = load_generator(checkpoint)
    assert generator.cond_dim == len(CATEGORIES)

    gan_sample.main([str(checkpoint), "-n", "5", "--category", "marketing", "--out", str(out)])
    assert {r.vendor_category for r in read_csv(out)} == {"marketing"}


# --------------------------------------------------------------------------- metrics


def test_ks_and_wasserstein_match_scipy() -> None:
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 300), rng.normal(0.4, 1.5, 200)

    assert ks_statistic(a, b) == pytest.approx(stats.ks_2samp(a, b).statistic)
    assert wasserstein(a, b) == pytest.approx(stats.wasserstein_distance(a, b))


def test_distances_are_zero_for_identical_samples_and_grow_with_a_shift() -> None:
    a = np.linspace(0, 1, 101)

    assert ks_statistic(a, a) == 0.0
    assert wasserstein(a, a) == 0.0
    assert wasserstein(a, a + 0.5) == pytest.approx(0.5)
    assert ks_statistic(a, a + 2) == 1.0
    assert total_variation(["x", "y"], ["y", "x"]) == 0.0
    assert total_variation(["x"] * 3 + ["y"], ["x"] * 4) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        ks_statistic([], [1.0])


def test_compare_scores_good_samples_close_to_the_baseline() -> None:
    real = seed_rows(400, seed=3)
    tr = TabularTransformer().fit(real)
    same_distribution = seed_rows(400, seed=4)
    garbage = sample_rows(untrained(tr), tr, 400)

    good = compare(tr, real, same_distribution)
    bad = compare(tr, real, garbage)
    ref = baseline(tr, real)

    assert good.violation_rate == 0.0 and bad.violation_rate == 0.0
    assert good.mean_ks < 0.15 and good.mean_ks < bad.mean_ks
    assert good.mean_wasserstein < bad.mean_wasserstein
    assert ref.n_real + ref.n_synthetic == len(real)
    assert good.joint_tvd is not None and good.joint_tvd < 0.2
    assert {c.column for c in good.columns} == {c.name for c in tr.columns}
    assert good.metric("currency").ks is None and good.metric("currency").tvd is not None
    assert 0 <= good.correlation.mean_abs <= good.correlation.max_abs <= 2


def test_compare_reports_rows_that_break_rules(transformer: TabularTransformer) -> None:
    real = seed_rows()
    broken = [{**row, "total": row["total"] + 5} for row in real[:10]] + real[10:]

    report = compare(transformer, real, broken)

    assert report.violation_rate == pytest.approx(10 / len(real))
    assert report.violation_examples and report.violation_examples[0].startswith("total")


def test_report_lists_every_column_and_the_baseline(transformer: TabularTransformer) -> None:
    real = seed_rows()
    report = render_markdown(
        compare(transformer, real, seed_rows(seed=9)),
        baseline(transformer, real),
        title="GAN quality",
        run_date=date(2026, 11, 13),
    )

    assert report.startswith("# GAN quality — 2026-11-13")
    for column in ("line_count", "subtotal", "vendor_category", "currency"):
        assert f"| {column} |" in report
    assert "Rows breaking business rules | 0.0%" in report


def test_evaluate_cli_writes_report_and_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = seed_rows()
    write_csv(to_records(rows), tmp_path / "seed.csv")
    config = TrainConfig(steps=2, batch_size=32, n_critic=1, hidden=(16, 16), noise_dim=8)
    train(SeedDataset.from_rows(rows), config, run_dir=tmp_path / "run")

    code = gan_eval.main(
        [
            str(tmp_path / "run" / "final.pt"),
            "-n",
            "60",
            "--seed-csv",
            str(tmp_path / "seed.csv"),
            "--report",
            str(tmp_path / "gan.md"),
            "--json",
            str(tmp_path / "gan.json"),
        ]
    )

    assert code == 0
    assert "Per column" in (tmp_path / "gan.md").read_text(encoding="utf-8")
    metrics = json.loads((tmp_path / "gan.json").read_text(encoding="utf-8"))
    assert metrics["synthetic"]["n_synthetic"] == 60
    assert metrics["synthetic"]["violation_rate"] == 0.0
    capsys.readouterr()
