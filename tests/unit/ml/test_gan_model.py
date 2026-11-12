import json
import math
import random
from pathlib import Path
from typing import Any

import pytest
import torch

from ml.synthetic.gan.dataset import SeedDataset
from ml.synthetic.gan.model import (
    Critic,
    Generator,
    critic_loss,
    generator_loss,
    gradient_penalty,
)
from ml.synthetic.gan.train import (
    TrainConfig,
    TrainingDivergedError,
    condition_vector,
    load_generator,
    load_seed_rows,
    main,
    train,
)

SPANS = [(8, 4), (12, 3)]  # vendor_category (4), currency (3)
FEATURES = 15


def rows(n: int = 96, seed: int = 1) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        subtotal = round(10 ** rng.uniform(1, 5), 2)
        out.append(
            {
                "line_count": rng.randint(1, 10),
                "qty_mean": rng.uniform(1, 10),
                "unit_price_mean": round(10 ** rng.uniform(0, 4), 2),
                "subtotal": subtotal,
                "tax_rate": rng.choice([0.0, 0.2]),
                "total": subtotal * 1.2,
                "days_to_due": rng.choice([0, 30]),
                "day_of_month": rng.randint(1, 28),
                "vendor_category": rng.choice(["a", "b", "c", "d"]),
                "currency": rng.choice(["GBP", "IDR", "USD"]),
            }
        )
    return out


def tiny(**overrides: Any) -> TrainConfig:
    settings: dict[str, Any] = {
        "steps": 3,
        "batch_size": 32,
        "n_critic": 2,
        "hidden": (32, 32),
        "noise_dim": 16,
        "log_every": 1,
        "checkpoint_every": 2,
    }
    return TrainConfig(**(settings | overrides))


# --------------------------------------------------------------------------- shapes


def test_generator_output_shape_and_ranges() -> None:
    generator = Generator(FEATURES, SPANS)

    out = generator(torch.randn(10, 128))

    assert out.shape == (10, FEATURES)
    assert out[:, :8].abs().max() <= 1  # tanh
    for start, width in SPANS:
        torch.testing.assert_close(out[:, start : start + width].sum(dim=1), torch.ones(10))
    assert out.requires_grad  # soft Gumbel-softmax keeps gradients


def test_sampling_is_hard_one_hot_without_gradients() -> None:
    sample = Generator(FEATURES, SPANS).sample(6)

    assert not sample.requires_grad
    for start, width in SPANS:
        block = sample[:, start : start + width]
        assert ((block == 0) | (block == 1)).all() and (block.sum(dim=1) == 1).all()


def test_critic_scores_one_number_per_row() -> None:
    critic = Critic(FEATURES)

    assert critic(torch.randn(7, FEATURES)).shape == (7,)


def test_losses_and_penalty_are_finite_and_differentiable() -> None:
    generator, critic = Generator(FEATURES, SPANS), Critic(FEATURES)
    real = torch.rand(16, FEATURES) * 2 - 1
    fake = generator(torch.randn(16, 128))

    penalty = gradient_penalty(critic, real, fake)
    loss_c = critic_loss(critic, real, fake)
    loss_g = generator_loss(critic, fake)

    assert penalty.item() >= 0
    assert all(math.isfinite(x.item()) for x in (penalty, loss_c, loss_g))
    torch.autograd.backward(loss_g)
    assert generator.head.weight.grad is not None


def test_conditional_networks_need_and_respect_the_condition() -> None:
    generator = Generator(FEATURES, SPANS, condition_span=(8, 4))
    critic = Critic(FEATURES, cond_dim=4)
    condition = torch.eye(4)[torch.tensor([2, 0, 3])]

    out = generator(torch.randn(3, 128), condition)

    assert torch.equal(out[:, 8:12], condition)  # rows get the category they asked for
    assert critic(out, condition).shape == (3,)
    assert torch.equal(generator.sample(5, torch.eye(4)[1])[:, 8:12], torch.eye(4)[1].expand(5, 4))
    with pytest.raises(ValueError, match="expected a condition with 4 columns"):
        generator(torch.randn(3, 128))
    with pytest.raises(ValueError, match="got \\(3, 2\\)"):
        critic(out, torch.zeros(3, 2))


# --------------------------------------------------------------------------- training


def test_a_few_training_steps_run_without_nans(tmp_path: Path) -> None:
    dataset = SeedDataset.from_rows(rows())

    result = train(dataset, tiny(), run_dir=tmp_path / "run")

    assert result.steps == 3 and [h["step"] for h in result.history] == [1, 2, 3]
    for entry in result.history:
        assert all(math.isfinite(v) for v in entry.values())
    run = json.loads((tmp_path / "run" / "run.json").read_text(encoding="utf-8"))
    assert run["config"]["n_critic"] == 2 and run["data"]["rows"] == 96
    assert len(run["history"]) == 3
    assert {p.name for p in (tmp_path / "run").glob("*.pt")} == {"checkpoint_000002.pt", "final.pt"}


def test_training_is_reproducible(tmp_path: Path) -> None:
    dataset = SeedDataset.from_rows(rows())

    first = train(dataset, tiny(seed=5), run_dir=tmp_path / "a").history
    second = train(dataset, tiny(seed=5), run_dir=tmp_path / "b").history

    assert first == second


def test_checkpoint_restores_a_generator_that_samples_rows(tmp_path: Path) -> None:
    dataset = SeedDataset.from_rows(rows(), condition="vendor_category")
    train(dataset, tiny(condition="vendor_category"), run_dir=tmp_path / "run")

    generator, transformer = load_generator(tmp_path / "run" / "final.pt")
    condition = condition_vector(transformer, "vendor_category", "c")
    sampled = transformer.inverse_transform(generator.sample(8, condition).numpy())

    assert len(sampled) == 8 and {r["vendor_category"] for r in sampled} == {"c"}
    assert all(r["currency"] in {"GBP", "IDR", "USD"} and r["total"] >= 0 for r in sampled)


def test_condition_mismatch_and_divergence_are_reported(tmp_path: Path) -> None:
    dataset = SeedDataset.from_rows(rows())

    with pytest.raises(ValueError, match="conditioned on None"):
        train(dataset, tiny(condition="vendor_category"), run_dir=tmp_path / "x")
    dataset.features[:, 0] = float("nan")  # e.g. a corrupt seed column
    with pytest.raises(TrainingDivergedError, match="step 1"):
        train(dataset, tiny(), run_dir=tmp_path / "y")


def test_seed_rows_fall_back_to_vendor_master_pos(tmp_path: Path) -> None:
    data, summary = load_seed_rows(tmp_path / "missing.csv")

    assert len(data) == 500 and summary["by_source"] == {"seed_po": 500}
    assert "not built" in summary["note"]


def test_cli_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "--steps",
            "2",
            "--batch-size",
            "64",
            "--seed-csv",
            str(tmp_path / "none.csv"),
            "--runs",
            str(tmp_path),
        ]
    )

    assert code == 0
    assert "2 steps on 500 rows" in capsys.readouterr().out
    assert len(list(tmp_path.glob("*/run.json"))) == 1
