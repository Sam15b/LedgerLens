"""The CI regression gate: this run against ``main``'s last report.

    python -m eval.regression --current eval/reports/ci-run/summary_ci.json --base-ref origin/main

The baseline is the newest ``eval/reports/<date>/summary_ci.json`` committed on ``main``
(read with ``git show``, so the checkout's own reports do not count); ``--baseline`` names
a file instead. Each metric is compared in points (one point = 0.01 of a 0-1 score):

* **gated** metrics (``configs/eval.yaml`` ``gate_metrics``: faithfulness and recall@5)
  fail the gate when they drop by more than ``regression_threshold_pts`` (3), or when
  ``main`` measured one and this run did not (a broken answer step must not pass);
* the other scores, latency and cost are shown for context and never fail it.

With no baseline on ``main`` yet the gate passes and says so. Exit code 1 means it failed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ledgerlens.core.config import get_config, repo_path

REPO = repo_path(".")
REPORTS = "eval/reports"

Status = Literal["ok", "improved", "regressed", "missing", "no baseline", "not measured"]
FAILING: frozenset[str] = frozenset({"regressed", "missing"})

SCORES = (
    "recall@5",
    "faithfulness",
    "mrr",
    "evidence_recall",
    "context_recall",
    "context_precision",
    "answer_relevancy",
    "answer_correctness",
    "facts_found",
    "citations_ok",
)
"""0-1 scores, compared in points."""
CONTEXT = ("latency.query.p95_ms", "latency.retrieval.p95_ms", "cost.query.mean_usd")
"""Shown next to the scores, never gated."""


@dataclass(frozen=True)
class MetricCheck:
    name: str
    baseline: float | None
    current: float | None
    gated: bool
    status: Status | None
    """``None`` for context metrics (latency, cost)."""

    @property
    def delta_pts(self) -> float | None:
        if self.baseline is None or self.current is None or self.status is None:
            return None
        return round((self.current - self.baseline) * 100, 2)


@dataclass
class Comparison:
    checks: list[MetricCheck]
    threshold_pts: float
    baseline_source: str | None
    """Where the baseline came from (``<ref>:<path>``); ``None``: no baseline."""
    notes: list[str] = field(default_factory=list)

    @property
    def failures(self) -> list[MetricCheck]:
        return [c for c in self.checks if c.gated and c.status in FAILING]

    @property
    def passed(self) -> bool:
        return not self.failures


def _value(metrics: Mapping[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    return float(value) if isinstance(value, int | float) else None


def check_score(
    name: str, baseline: float | None, current: float | None, *, gated: bool, threshold_pts: float
) -> MetricCheck:
    status: Status
    if current is None:
        status = "missing" if baseline is not None else "not measured"
    elif baseline is None:
        status = "no baseline"
    else:
        delta = (current - baseline) * 100
        if delta < -threshold_pts:
            status = "regressed"
        elif delta > threshold_pts:
            status = "improved"
        else:
            status = "ok"
    return MetricCheck(name, baseline, current, gated, status)


def compare(
    current: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    *,
    gated: Sequence[str],
    threshold_pts: float,
    baseline_source: str | None = None,
) -> Comparison:
    """Compare two ``summary_*.json`` documents (their ``metrics``)."""
    now = current.get("metrics", {})
    before = baseline.get("metrics", {}) if baseline else {}
    names = [*gated, *(s for s in SCORES if s not in gated)]
    checks = []
    for name in names:
        b, c = _value(before, name), _value(now, name)
        if name not in gated and b is None and c is None:
            continue  # an ungated score neither run measured: nothing to show
        checks.append(check_score(name, b, c, gated=name in gated, threshold_pts=threshold_pts))
    for name in CONTEXT:
        b, c = _value(before, name), _value(now, name)
        if b is not None or c is not None:
            checks.append(MetricCheck(name, b, c, gated=False, status=None))
    comparison = Comparison(checks, threshold_pts, baseline_source if baseline else None)
    if baseline is None:
        comparison.notes.append("No report on main yet: nothing to compare with, so it passes.")
    elif baseline.get("suite") != current.get("suite"):
        comparison.notes.append(
            f"The baseline is the {baseline.get('suite')} set, this run the "
            f"{current.get('suite')} set: the numbers are not like for like."
        )
    return comparison


# --------------------------------------------------------------------------- baseline


def summary_name(suite: str) -> str:
    return "summary.json" if suite == "full" else f"summary_{suite}.json"


def _git(*args: str, repo: Path = REPO) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout


def find_baseline(
    ref: str, suite: str = "ci", *, repo: Path = REPO
) -> tuple[str, dict[str, Any]] | None:
    """The newest committed ``eval/reports/<date>/summary_<suite>.json`` at ``ref`` (dated
    directories sort by name), as ``("<ref>:<path>", data)``; ``None`` if there is none."""
    name = summary_name(suite)
    try:
        listed = _git("ls-tree", "-r", "--name-only", ref, "--", REPORTS, repo=repo)
    except (OSError, subprocess.CalledProcessError):
        return None
    paths = sorted(p for p in listed.splitlines() if p.endswith(f"/{name}"))
    if not paths:
        return None
    path = paths[-1]
    return f"{ref}:{path}", json.loads(_git("show", f"{ref}:{path}", repo=repo))


def load_summary(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


# --------------------------------------------------------------------------- output


def _fmt(name: str, value: float | None) -> str:
    if value is None:
        return "-"
    if name.endswith("_ms"):
        return f"{value:,.0f} ms"
    if name.endswith("_usd"):
        return f"${value:.5f}"
    return f"{value:.3f}"


_LABEL: dict[str | None, str] = {
    "ok": "ok",
    "improved": "improved",
    "regressed": "**regressed**",
    "missing": "**not measured in this run**",
    "no baseline": "new (no baseline)",
    "not measured": "not measured",
    None: "",
}


def render_table(comparison: Comparison) -> list[str]:
    lines = [
        "| metric | main | this PR | change (pts) | gate |",
        "|---|---:|---:|---:|---|",
    ]
    for check in comparison.checks:
        delta = "" if check.delta_pts is None else f"{check.delta_pts:+.1f}"
        gate = _LABEL[check.status] if check.gated else "info"
        lines.append(
            f"| {check.name} | {_fmt(check.name, check.baseline)} | "
            f"{_fmt(check.name, check.current)} | {delta} | {gate} |"
        )
    return lines


def verdict(comparison: Comparison) -> str:
    threshold = f"{comparison.threshold_pts:g} points"
    if comparison.passed:
        return f"Eval gate passed: no gated metric dropped more than {threshold}."
    names = ", ".join(c.name for c in comparison.failures)
    return f"Eval gate FAILED on {names} (threshold {threshold})."


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.regression", description="Fail on a regression vs main's report."
    )
    parser.add_argument("--current", type=Path, required=True, help="this run's summary JSON")
    parser.add_argument("--baseline", type=Path, help="baseline summary JSON (default: git)")
    parser.add_argument("--base-ref", default="origin/main", help="where to find the baseline")
    parser.add_argument("--suite", default=None, help="default: the current run's suite")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = get_config().eval
    current = load_summary(args.current)
    suite = args.suite or current.get("suite", "ci")
    if args.baseline:
        found: tuple[str, dict[str, Any]] | None = (
            str(args.baseline),
            load_summary(args.baseline),
        )
    else:
        found = find_baseline(args.base_ref, suite)
    source, baseline = found if found else (None, None)
    comparison = compare(
        current,
        baseline,
        gated=config.gate_metrics,
        threshold_pts=config.regression_threshold_pts,
        baseline_source=source,
    )
    print(verdict(comparison))
    print(f"baseline: {source or 'none'}")
    print("\n".join(render_table(comparison)))
    for note in comparison.notes:
        print(note)
    return 0 if comparison.passed else 1


if __name__ == "__main__":
    sys.exit(main())
