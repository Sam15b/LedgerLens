"""Train the tabular WGAN-GP on the seed records.

    python ml/synthetic/gan/train.py --steps 5000                # CPU is fine for 1.5k rows
    python ml/synthetic/gan/train.py --steps 20                  # smoke test

Reads ``data/synthetic/seed/seed_invoices.csv`` (``python -m ml.synthetic.seed``); if that
has not been built, it falls back to the seed invoices of the hand-made vendor master's
POs, which need no downloaded data.

Optimizers: Adam, lr 2e-4, betas (0.5, 0.9), 5 critic steps per generator step (the
WGAN-GP paper's settings). Every run writes to ``ml/synthetic/gan/runs/<run_id>/``
(git-ignored): ``run.json`` (config, data summary, loss history, timing) and checkpoints
holding the generator, critic and the fitted transformer, so a checkpoint alone is enough
to sample rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # run as a script: make the repo root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from torch import Tensor

from ml.synthetic.gan.dataset import SeedDataset, TabularTransformer, make_loader
from ml.synthetic.gan.model import Critic, Generator, generator_loss, gradient_penalty
from ml.synthetic.seed import load_vendor_master, po_invoice_records, read_csv, seed_purchase_orders

RUNS_DIR = Path(__file__).resolve().parent / "runs"
SEED_CSV = Path("data/synthetic/seed/seed_invoices.csv")


@dataclass
class TrainConfig:
    steps: int = 5000  # generator updates
    batch_size: int = 256
    lr: float = 2e-4
    betas: tuple[float, float] = (0.5, 0.9)
    n_critic: int = 5
    gp_weight: float = 10.0
    noise_dim: int = 128
    hidden: tuple[int, ...] = (256, 256)
    dropout: float = 0.2
    seed: int = 0
    log_every: int = 50
    checkpoint_every: int = 1000
    device: str = "cpu"
    condition: str | None = None  # e.g. "vendor_category" for a conditional GAN


@dataclass
class TrainResult:
    run_dir: Path
    steps: int
    history: list[dict[str, float]] = field(default_factory=list)
    generator: Generator | None = None

    @property
    def last(self) -> dict[str, float]:
        return self.history[-1] if self.history else {}


class TrainingDivergedError(RuntimeError):
    """A loss became NaN or infinite."""


def _batches(dataset: SeedDataset, config: TrainConfig) -> Iterator[tuple[Tensor, Tensor]]:
    epoch = 0
    while True:
        loader = make_loader(dataset, batch_size=config.batch_size, seed=config.seed + epoch)
        yield from loader
        epoch += 1


def train(
    dataset: SeedDataset,
    config: TrainConfig | None = None,
    *,
    run_dir: Path | None = None,
    data_summary: dict[str, Any] | None = None,
) -> TrainResult:
    """Train a generator/critic pair on ``dataset`` and log the run to ``run_dir``."""
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    run_dir = run_dir or RUNS_DIR / f"{datetime.now(UTC):%Y%m%d-%H%M%S}-seed{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if dataset.condition != config.condition:
        raise ValueError(
            f"dataset is conditioned on {dataset.condition!r}, config on {config.condition!r}"
        )
    spans = dataset.transformer.categorical_spans
    condition_span = _condition_span(dataset.transformer, config.condition)
    generator = Generator(
        dataset.feature_dim,
        spans,
        noise_dim=config.noise_dim,
        hidden=config.hidden,
        condition_span=condition_span,
    ).to(device)
    critic = Critic(
        dataset.feature_dim,
        hidden=config.hidden,
        dropout=config.dropout,
        cond_dim=dataset.condition_dim,
    ).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=config.lr, betas=config.betas)
    opt_c = torch.optim.Adam(critic.parameters(), lr=config.lr, betas=config.betas)

    batches = _batches(dataset, config)
    result = TrainResult(run_dir=run_dir, steps=0, generator=generator)
    started, start = datetime.now(UTC), time.perf_counter()
    for step in range(1, config.steps + 1):
        for _ in range(config.n_critic):
            real, cond = (t.to(device) for t in next(batches))
            condition = cond if dataset.condition_dim else None
            noise = torch.randn(real.shape[0], config.noise_dim, device=device)
            fake = generator(noise, condition)
            fake_score: Tensor = critic(fake.detach(), condition)
            real_score: Tensor = critic(real, condition)
            wasserstein = real_score.mean() - fake_score.mean()
            penalty = gradient_penalty(critic, real, fake, condition)
            loss_c = -wasserstein + config.gp_weight * penalty
            opt_c.zero_grad(set_to_none=True)
            torch.autograd.backward(loss_c)
            opt_c.step()

        # Generator step: conditions drawn from the data, so categories keep their mix.
        _, cond = next(batches)
        condition = cond.to(device) if dataset.condition_dim else None
        noise = torch.randn(cond.shape[0], config.noise_dim, device=device)
        loss_g = generator_loss(critic, generator(noise, condition), condition)
        opt_g.zero_grad(set_to_none=True)
        torch.autograd.backward(loss_g)
        opt_g.step()

        losses = {
            "critic_loss": loss_c.item(),
            "generator_loss": loss_g.item(),
            "wasserstein": wasserstein.item(),
            "gradient_penalty": penalty.item(),
        }
        if not all(math.isfinite(v) for v in losses.values()):
            raise TrainingDivergedError(f"step {step}: {losses}")
        result.steps = step
        if step % config.log_every == 0 or step in (1, config.steps):
            result.history.append({"step": step, **losses})
        if step % config.checkpoint_every == 0:
            _checkpoint(
                run_dir / f"checkpoint_{step:06d}.pt", generator, critic, dataset, config, step
            )

    _checkpoint(run_dir / "final.pt", generator, critic, dataset, config, result.steps)
    run = {
        "run_id": run_dir.name,
        "started_at": started.isoformat(),
        "seconds": round(time.perf_counter() - start, 2),
        "config": asdict(config),
        "data": {
            "rows": len(dataset),
            "feature_dim": dataset.feature_dim,
            "condition_dim": dataset.condition_dim,
            **(data_summary or {}),
        },
        "history": result.history,
    }
    (run_dir / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    return result


def _checkpoint(
    path: Path,
    generator: Generator,
    critic: Critic,
    dataset: SeedDataset,
    config: TrainConfig,
    step: int,
) -> None:
    torch.save(
        {
            "step": step,
            "config": asdict(config),
            "transformer": dataset.transformer.to_dict(),
            "categorical_spans": dataset.transformer.categorical_spans,
            "condition_span": generator.condition_span,
            "generator": generator.state_dict(),
            "critic": critic.state_dict(),
        },
        path,
    )


def load_generator(path: Path | str) -> tuple[Generator, TabularTransformer]:
    """A trained generator and its transformer from a checkpoint (for sampling)."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    transformer = TabularTransformer.from_dict(state["transformer"])
    config = state["config"]
    condition_span = state.get("condition_span")
    generator = Generator(
        transformer.output_dim,
        [tuple(span) for span in state["categorical_spans"]],
        noise_dim=config["noise_dim"],
        hidden=config["hidden"],
        condition_span=tuple(condition_span) if condition_span else None,
    )
    generator.load_state_dict(state["generator"])
    generator.eval()
    return generator, transformer


def _condition_span(transformer: TabularTransformer, name: str | None) -> tuple[int, int] | None:
    if name is None:
        return None
    info = transformer.column(name)
    return info.start, info.width


def condition_vector(transformer: TabularTransformer, column: str, value: str) -> Tensor:
    """The one-hot condition for ``column == value`` (for ``Generator.sample``)."""
    info = transformer.column(column)
    vector = torch.zeros(info.width)
    vector[info.categories.index(value)] = 1.0
    return vector


def load_seed_rows(path: Path = SEED_CSV) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Seed rows for training, and a summary of where they came from."""
    if path.is_file():
        records = read_csv(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        source: dict[str, Any] = {"seed_csv": str(path), "sha256": digest}
    else:
        vendors = load_vendor_master()
        records = po_invoice_records(seed_purchase_orders(vendors), vendors)
        source = {"seed_csv": None, "note": f"{path} not built; used vendor-master POs only"}
    by_source: dict[str, int] = {}
    for record in records:
        by_source[record.source] = by_source.get(record.source, 0) + 1
    return [r.features() for r in records], {**source, "by_source": by_source}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the tabular WGAN-GP on seed records")
    parser.add_argument("--steps", type=int, default=TrainConfig.steps)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--seed-csv", type=Path, default=SEED_CSV)
    parser.add_argument("--runs", type=Path, default=RUNS_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--condition", choices=["vendor_category"], help="train a conditional GAN on this column"
    )
    args = parser.parse_args(argv)

    rows, summary = load_seed_rows(args.seed_csv)
    dataset = SeedDataset.from_rows(rows, condition=args.condition)
    config = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        condition=args.condition,
        log_every=max(1, min(TrainConfig.log_every, args.steps // 10 or 1)),
    )
    run_dir = args.runs / f"{datetime.now(UTC):%Y%m%d-%H%M%S}-seed{args.seed}"
    result = train(dataset, config, run_dir=run_dir, data_summary=summary)
    last = result.last
    print(
        f"{result.steps} steps on {len(dataset)} rows -> {run_dir}\n"
        f"critic {last.get('critic_loss', float('nan')):.4f}  "
        f"generator {last.get('generator_loss', float('nan')):.4f}  "
        f"W-distance {last.get('wasserstein', float('nan')):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
