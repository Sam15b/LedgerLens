"""The CI regression gate: which changes fail it, where the baseline comes from, and what
the PR comment says."""

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from eval.regression import (
    COMMENT_MARKER,
    check_score,
    compare,
    find_baseline,
    main,
    render_comment,
    render_table,
    verdict,
)

GATED = ["faithfulness", "recall@5"]


def summary(suite: str = "ci", /, **metrics: Any) -> dict[str, Any]:
    return {
        "suite": suite,
        "git_sha": "a" * 40,
        "config_hash": "c" * 64,
        "llm_status": "ok",
        "metrics": metrics,
    }


MAIN = summary(
    faithfulness=0.90,
    **{"recall@5": 0.87},
    mrr=0.89,
    **{"latency.query.p95_ms": 9000.0, "cost.query.mean_usd": 0.004},
)


# --------------------------------------------------------------------------- one metric


@pytest.mark.parametrize(
    ("baseline", "current", "status"),
    [
        (0.90, 0.90, "ok"),
        (0.90, 0.87, "ok"),  # exactly 3 points: not more than the threshold
        (0.90, 0.8699, "regressed"),
        (0.90, 0.94, "improved"),
        (0.90, None, "missing"),  # main measured it, this run did not
        (None, 0.80, "no baseline"),
        (None, None, "not measured"),
    ],
)
def test_check_score(baseline: float | None, current: float | None, status: str) -> None:
    check = check_score("faithfulness", baseline, current, gated=True, threshold_pts=3)
    assert check.status == status


def test_delta_is_in_points() -> None:
    check = check_score("recall@5", 0.869, 0.533, gated=True, threshold_pts=3)
    assert check.delta_pts == -33.6


# --------------------------------------------------------------------------- comparisons


def test_a_broken_prompt_fails_the_gate() -> None:
    """A prompt change that makes answers less grounded: faithfulness 0.90 -> 0.78."""
    current = summary(faithfulness=0.78, **{"recall@5": 0.87}, mrr=0.89)

    result = compare(current, MAIN, gated=GATED, threshold_pts=3, baseline_source="main:x")

    assert not result.passed
    assert [c.name for c in result.failures] == ["faithfulness"]
    assert verdict(result) == "Eval gate FAILED on faithfulness (threshold 3 points)."


def test_ungated_drops_and_slower_runs_do_not_fail() -> None:
    current = summary(
        faithfulness=0.91,
        **{"recall@5": 0.88},
        mrr=0.50,  # a big drop, but MRR is not gated
        **{"latency.query.p95_ms": 30000.0, "cost.query.mean_usd": 0.05},
    )

    result = compare(current, MAIN, gated=GATED, threshold_pts=3)

    assert result.passed
    by_name = {c.name: c for c in result.checks}
    assert by_name["mrr"].status == "regressed" and not by_name["mrr"].gated
    assert by_name["latency.query.p95_ms"].status is None
    assert [c.name for c in result.checks][:2] == GATED  # gated first


def test_a_run_that_could_not_answer_fails_when_main_could() -> None:
    current = summary(**{"recall@5": 0.87})  # answers skipped: no faithfulness

    result = compare(current, MAIN, gated=GATED, threshold_pts=3)

    assert [(c.name, c.status) for c in result.failures] == [("faithfulness", "missing")]


def test_metrics_neither_side_measured_pass() -> None:
    main_report = summary(**{"recall@5": 0.869})
    current = summary(**{"recall@5": 0.86})

    result = compare(current, main_report, gated=GATED, threshold_pts=3)

    assert result.passed
    assert {c.name: c.status for c in result.checks}["faithfulness"] == "not measured"
    assert "context_recall" not in {c.name for c in result.checks}  # ungated, never measured


def test_no_baseline_passes_and_says_so() -> None:
    result = compare(summary(**{"recall@5": 0.2}), None, gated=GATED, threshold_pts=3)

    assert result.passed and result.baseline_source is None
    assert result.notes == ["No report on main yet: nothing to compare with, so it passes."]


def test_a_different_suite_is_flagged() -> None:
    result = compare(
        summary(**{"recall@5": 0.9}),
        summary("full", **{"recall@5": 0.9}),
        gated=GATED,
        threshold_pts=3,
    )
    assert "not like for like" in result.notes[0]


# --------------------------------------------------------------------------- output


def test_table_and_comment() -> None:
    current = summary(faithfulness=0.78, **{"recall@5": 0.87}, mrr=0.89)
    result = compare(current, MAIN, gated=GATED, threshold_pts=3, baseline_source="origin/main:r")

    table = render_table(result)
    assert table[2] == "| faithfulness | 0.900 | 0.780 | -12.0 | **regressed** |"
    assert table[3] == "| recall@5 | 0.870 | 0.870 | +0.0 | ok |"
    assert "| latency.query.p95_ms | 9,000 ms | - |  | info |" in table
    assert "| cost.query.mean_usd | $0.00400 | - |  | info |" in table

    comment = render_comment(result, current, run_url="https://ci/run/1")
    assert comment.startswith(COMMENT_MARKER + "\n### :x: Eval gate (ci set)")
    assert "- baseline: origin/main:r" in comment
    assert "- this PR: git `aaaaaaaaaaaa`, config `cccccccccccc`; answers: ok" in comment
    assert "[workflow run](https://ci/run/1)" in comment


# --------------------------------------------------------------------------- baseline


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_baseline_is_the_newest_committed_report(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "user.email", "ci@example.invalid")
    git(tmp_path, "config", "user.name", "ci")
    for day, recall in (("2026-11-20", 0.80), ("2026-11-22", 0.87)):
        folder = tmp_path / "eval" / "reports" / day
        folder.mkdir(parents=True)
        (folder / "summary_ci.json").write_text(json.dumps(summary(**{"recall@5": recall})))
        (folder / "summary.json").write_text(json.dumps(summary("full", **{"recall@5": 0.5})))
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "reports")
    # a newer report that is not committed does not count
    newer = tmp_path / "eval" / "reports" / "2026-11-30"
    newer.mkdir()
    (newer / "summary_ci.json").write_text(json.dumps(summary(**{"recall@5": 0.1})))

    found = find_baseline("main", "ci", repo=tmp_path)

    assert found is not None
    source, data = found
    assert source == "main:eval/reports/2026-11-22/summary_ci.json"
    assert data["metrics"]["recall@5"] == 0.87
    full = find_baseline("main", "full", repo=tmp_path)
    assert full is not None and full[0].endswith("2026-11-22/summary.json")
    assert find_baseline("no-such-ref", "ci", repo=tmp_path) is None
    assert find_baseline("main", "nightly", repo=tmp_path) is None


# --------------------------------------------------------------------------- CLI


def write(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_cli_exit_codes_and_comment(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    baseline = write(tmp_path / "main.json", MAIN)
    good = write(tmp_path / "good.json", summary(faithfulness=0.89, **{"recall@5": 0.86}))
    bad = write(tmp_path / "bad.json", summary(faithfulness=0.89, **{"recall@5": 0.50}))
    comment = tmp_path / "out" / "comment.md"

    assert main(["--current", str(good), "--baseline", str(baseline)]) == 0
    assert "Eval gate passed" in capsys.readouterr().out
    assert (
        main(["--current", str(bad), "--baseline", str(baseline), "--comment", str(comment)]) == 1
    )
    assert "FAILED on recall@5" in comment.read_text(encoding="utf-8")
