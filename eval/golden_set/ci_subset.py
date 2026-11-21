"""Pick the golden set's CI subset: a small, stratified sample for fast regression checks.

    python -m eval.golden_set.ci_subset              # writes data/golden/golden_ci.jsonl
    python -m eval.golden_set.ci_subset --check      # is the committed subset current?

Size and paths come from ``configs/eval.yaml`` (``ci_subset_size``, ``golden_set``,
``ci_subset``). The subset is chosen in two steps, so every kind of question the full set
holds is represented:

1. **Per category**, a quota proportional to the category's share of the set (largest
   remainder), with at least one question from every category.
2. **Within a category**, the quota is split across sources (``synthetic``, ``cord``,
   ``cuad``) the same way, so half the lookups come from real receipts when half the set's
   lookups do. Within a source, the questions are grouped by shape (the ``check`` kind;
   ``hand`` for hand-written and v0 questions), and the share is filled one shape at a time
   in turn, so each shape appears before any repeats.

The choice is seeded and depends only on the set's content, so the same set always gives
the same subset. Subset questions keep their ids.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from eval.golden_set.schema import (
    CATEGORIES,
    GoldenQuestion,
    dump_golden_set,
    load_golden_set,
    summary,
)

SEED = 40


def quotas(counts: dict[str, int], size: int) -> dict[str, int]:
    """``size`` split across categories in proportion to ``counts`` (largest remainder),
    at least one each."""
    present = {c: n for c, n in counts.items() if n}
    if size < len(present):
        raise ValueError(f"a subset of {size} cannot cover {len(present)} categories")
    total = sum(present.values())
    exact = {c: size * n / total for c, n in present.items()}
    out = {c: max(1, int(x)) for c, x in exact.items()}
    by_remainder = sorted(present, key=lambda c: (-(exact[c] - int(exact[c])), c))
    k = 0
    while sum(out.values()) < size:
        category = by_remainder[k % len(by_remainder)]
        if out[category] < present[category]:
            out[category] += 1
        k += 1
    while sum(out.values()) > size:  # the "at least one" floor overshot
        largest = max(out, key=lambda c: (out[c], c))
        out[largest] -= 1
    return out


def stratum(question: GoldenQuestion) -> tuple[str, str]:
    kind = str((question.check or {}).get("kind", "hand"))
    return question.source, kind


def ci_subset(
    questions: Sequence[GoldenQuestion], size: int, *, seed: int = SEED
) -> list[GoldenQuestion]:
    rng = random.Random(seed)
    by_category: dict[str, list[GoldenQuestion]] = defaultdict(list)
    for q in sorted(questions, key=lambda q: q.id):
        by_category[q.category].append(q)
    chosen: list[GoldenQuestion] = []
    counts = {c: len(by_category[c]) for c in CATEGORIES}
    for category, quota in quotas(counts, size).items():
        by_source: dict[str, list[GoldenQuestion]] = defaultdict(list)
        for q in by_category[category]:
            by_source[q.source].append(q)
        source_counts = {s: len(items) for s, items in sorted(by_source.items())}
        for source, share in quotas(source_counts, min(quota, sum(source_counts.values()))).items():
            chosen += _round_robin(by_source[source], share, rng)
    return sorted(chosen, key=lambda q: int(q.id[1:]))


def _round_robin(
    questions: Sequence[GoldenQuestion], quota: int, rng: random.Random
) -> list[GoldenQuestion]:
    """``quota`` questions, one question shape at a time in turn."""
    strata: dict[tuple[str, str], list[GoldenQuestion]] = defaultdict(list)
    for q in questions:
        strata[stratum(q)].append(q)
    queues = []
    for key in sorted(strata):
        items = list(strata[key])
        rng.shuffle(items)
        queues.append(items)
    rng.shuffle(queues)
    taken: list[GoldenQuestion] = []
    while len(taken) < quota:
        for queue in queues:
            if queue and len(taken) < quota:
                taken.append(queue.pop())
    return taken


def main(argv: Sequence[str] | None = None) -> int:
    from ledgerlens.core.config import get_config, repo_path

    config = get_config().eval
    parser = argparse.ArgumentParser(description="Pick the golden set's CI subset.")
    parser.add_argument("--golden", type=Path, default=repo_path(config.golden_set))
    parser.add_argument("--out", type=Path, default=repo_path(config.ci_subset))
    parser.add_argument("--size", type=int, default=config.ci_subset_size)
    parser.add_argument("--check", action="store_true", help="fail if --out is not current")
    args = parser.parse_args(argv)
    subset = ci_subset(load_golden_set(args.golden), args.size)
    if args.check:
        current = load_golden_set(args.out) if args.out.exists() else []
        if current != subset:
            print(f"{args.out} is out of date: rerun without --check", file=sys.stderr)
            return 1
        print(f"{args.out} is current")
        print(summary(subset))
        return 0
    dump_golden_set(subset, args.out)
    print(summary(subset))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
