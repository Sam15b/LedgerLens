"""Tabular GAN for seed invoice rows.

The generator maps noise to one row in :class:`~ml.synthetic.gan.dataset.TabularTransformer`
space: numeric columns through ``tanh`` (the transformer scales them to ``[-1, 1]``) and
each categorical block through a **Gumbel-softmax** head, so it can emit (nearly) one-hot
categories while staying differentiable.

    noise (128) -> Linear 256 -> LeakyReLU -> Linear 256 -> LeakyReLU -> Linear features
        features: tanh on numeric columns, Gumbel-softmax on each categorical block

The critic scores rows with an MLP (LeakyReLU + dropout, no batch norm) and is trained as a
Wasserstein critic with a gradient penalty (WGAN-GP), which keeps GAN training on small,
mixed-type tables stable.

**Conditioning** (optional): with ``cond_dim > 0`` both networks also take a condition
vector -- the ``vendor_category`` one-hot -- so rows can be generated for a chosen
category ("give me 500 logistics invoices"). The generator's own output for that block is
replaced by the condition, so a generated row always has the category it was asked for.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
import torch.nn.functional as F  # noqa: N812 -- the PyTorch convention
from torch import Tensor, nn

NOISE_DIM = 128
HIDDEN = (256, 256)
TAU = 0.2  # Gumbel-softmax temperature: low = close to one-hot


def _mlp(sizes: Sequence[int], *, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for n_in, n_out in zip(sizes, sizes[1:], strict=False):
        layers += [nn.Linear(n_in, n_out), nn.LeakyReLU(0.2)]
        if dropout:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    """Noise -> a transformed seed row.

    ``categorical_spans`` are the ``(start, width)`` one-hot blocks of the feature vector
    (``TabularTransformer.categorical_spans``); every other position is numeric.
    ``condition_span`` is the block the condition vector stands for (conditional GAN).
    """

    def __init__(
        self,
        feature_dim: int,
        categorical_spans: Sequence[tuple[int, int]],
        *,
        noise_dim: int = NOISE_DIM,
        hidden: Sequence[int] = HIDDEN,
        tau: float = TAU,
        condition_span: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.noise_dim = noise_dim
        self.tau = tau
        self.condition_span = condition_span
        self.cond_dim = condition_span[1] if condition_span else 0
        self.spans = [(int(s), int(w)) for s, w in categorical_spans]
        categorical = {i for start, width in self.spans for i in range(start, start + width)}
        self.register_buffer(
            "numeric_index",
            torch.tensor([i for i in range(feature_dim) if i not in categorical], dtype=torch.long),
        )
        self.body = _mlp([noise_dim + self.cond_dim, *hidden])
        self.head = nn.Linear(hidden[-1], feature_dim)

    def forward(
        self, noise: Tensor, condition: Tensor | None = None, *, hard: bool = False
    ) -> Tensor:
        """Rows for ``noise`` (batch, noise_dim) and, for a conditional generator, the
        ``condition`` one-hots (batch, cond_dim). ``hard=True`` emits exact one-hot blocks
        (for sampling); training uses the soft relaxation so gradients flow."""
        inputs = _with_condition(noise, condition, self.cond_dim)
        logits = self.head(self.body(inputs))
        out = torch.empty_like(logits)
        index = cast(Tensor, self.numeric_index)
        out[:, index] = torch.tanh(logits[:, index])
        for start, width in self.spans:
            block = logits[:, start : start + width]
            out[:, start : start + width] = F.gumbel_softmax(block, tau=self.tau, hard=hard)
        if condition is not None and self.condition_span is not None:
            start, width = self.condition_span
            out[:, start : start + width] = condition
        return out

    def sample(
        self,
        n: int,
        condition: Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """``n`` hard samples, without gradients. A conditional generator needs
        ``condition``: one one-hot row (repeated) or ``n`` of them."""
        device = self.head.weight.device
        if condition is not None and condition.dim() == 1:
            condition = condition.expand(n, -1)
        with torch.no_grad():
            noise = torch.randn(n, self.noise_dim, generator=generator, device=device)
            return self.forward(noise, condition, hard=True)


class Critic(nn.Module):
    """Row -> realness score (a WGAN critic: unbounded, no sigmoid).

    No batch normalization: the gradient penalty is computed per sample, and batch norm
    would couple the samples in a batch.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        hidden: Sequence[int] = HIDDEN,
        dropout: float = 0.2,
        cond_dim: int = 0,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.cond_dim = cond_dim
        self.body = _mlp([feature_dim + cond_dim, *hidden], dropout=dropout)
        self.head = nn.Linear(hidden[-1], 1)

    def forward(self, rows: Tensor, condition: Tensor | None = None) -> Tensor:
        inputs = _with_condition(rows, condition, self.cond_dim)
        score: Tensor = self.head(self.body(inputs)).squeeze(-1)
        return score


def _with_condition(x: Tensor, condition: Tensor | None, cond_dim: int) -> Tensor:
    """``x`` with the condition appended; checks a conditional network got one."""
    if cond_dim == 0:
        return x
    if condition is None or condition.shape[-1] != cond_dim:
        got = None if condition is None else tuple(condition.shape)
        raise ValueError(f"expected a condition with {cond_dim} columns, got {got}")
    return torch.cat([x, condition], dim=1)


GP_WEIGHT = 10.0


def gradient_penalty(
    critic: Critic, real: Tensor, fake: Tensor, condition: Tensor | None = None
) -> Tensor:
    """WGAN-GP penalty (Gulrajani et al., 2017): ``E[(||grad critic(x_hat)||_2 - 1)^2]``
    on random interpolations ``x_hat`` between real and generated rows, which keeps the
    critic close to 1-Lipschitz without weight clipping."""
    alpha = torch.rand(real.shape[0], 1, device=real.device)
    mixed = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)
    scores: Tensor = critic(mixed, condition)
    grad: Tensor
    (grad,) = torch.autograd.grad(
        outputs=scores,
        inputs=mixed,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
    )
    slopes: Tensor = grad.norm(2, dim=1)
    return ((slopes - 1) ** 2).mean()


def critic_loss(
    critic: Critic,
    real: Tensor,
    fake: Tensor,
    condition: Tensor | None = None,
    *,
    gp_weight: float = GP_WEIGHT,
) -> Tensor:
    """Wasserstein critic loss ``E[D(fake)] - E[D(real)]`` plus the gradient penalty."""
    fake_score: Tensor = critic(fake.detach(), condition)
    real_score: Tensor = critic(real, condition)
    wasserstein = fake_score.mean() - real_score.mean()
    return wasserstein + gp_weight * gradient_penalty(critic, real, fake, condition)


def generator_loss(critic: Critic, fake: Tensor, condition: Tensor | None = None) -> Tensor:
    score: Tensor = critic(fake, condition)
    return -score.mean()
