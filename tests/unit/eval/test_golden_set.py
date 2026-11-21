"""The committed golden set validates, and the validator catches what it should."""

import json
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from eval.golden_set.build_golden_set import Draft, _period, check_grounding
from eval.golden_set.ci_subset import ci_subset, quotas
from eval.golden_set.schema import (
    CATEGORIES,
    GoldenSetError,
    load_golden_set,
    main,
    missing_documents,
    parse_golden_set,
    problems,
)
from eval.golden_set.verify import KINDS
from ledgerlens.core.config import get_config, repo_path

GOLDEN = repo_path("data/golden/golden_set.jsonl")
DOC = "22bac600-2669-42aa-bd5a-469d93c382a0"


def row(**overrides: Any) -> dict[str, Any]:
    return {
        "id": "q001",
        "category": "lookup",
        "question": "What is the total on invoice INV-007324?",
        "ground_truth_answer": "12,656.66 GBP",
        "relevant_document_ids": [DOC],
        "requires_graph": False,
        "difficulty": "easy",
    } | overrides


def parse(*rows: dict[str, Any]) -> list[Any]:
    return parse_golden_set([json.dumps(r) for r in rows])


# --------------------------------------------------------------------------- the committed set


@pytest.fixture(scope="module")
def golden() -> list[Any]:
    return load_golden_set(GOLDEN)


def test_golden_set_validates(golden: list[Any]) -> None:
    assert 45 <= len(golden) <= 250
    assert problems(golden) == []


def test_every_category_is_covered(golden: list[Any]) -> None:
    counts = {category: sum(q.category == category for q in golden) for category in CATEGORIES}
    assert all(n >= 4 for n in counts.values()), counts


def test_every_question_is_reviewed_and_has_expected_facts(golden: list[Any]) -> None:
    assert all(q.reviewed for q in golden)
    assert all(q.expected for q in golden)


def test_graph_questions_are_the_relational_categories(golden: list[Any]) -> None:
    for q in golden:
        relational = q.category in ("multi_hop", "aggregation", "reconciliation")
        assert q.requires_graph == relational, q.id


def test_questions_do_not_leak_document_ids(golden: list[Any]) -> None:
    for q in golden:
        assert not any(doc in q.question for doc in q.relevant_document_ids), q.id


def test_cli_accepts_the_committed_set(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(GOLDEN)]) == 0
    assert "questions: lookup" in capsys.readouterr().out


# --------------------------------------------------------------------------- the full set

DOC_LABELS = {
    "Single-doc lookup": "lookup",
    "Contract clause": "clause",
    "Multi-hop / relationship": "multi_hop",
    "Aggregation": "aggregation",
    "Reconciliation": "reconciliation",
    "Unanswerable": "unanswerable",
}


def doc_counts() -> dict[str, int]:
    """The category table in docs/layers/09-evaluation.md."""
    text = repo_path("docs/layers/09-evaluation.md").read_text(encoding="utf-8")
    counts = {}
    for label, category in DOC_LABELS.items():
        match = re.search(rf"^\| {re.escape(label)} \| (\d+) \|", text, re.M)
        assert match, label
        counts[category] = int(match.group(1))
    return counts


def test_the_full_set_matches_the_documented_counts(golden: list[Any]) -> None:
    counts = Counter(q.category for q in golden)
    assert dict(counts) == doc_counts()
    assert len(golden) == sum(doc_counts().values()) == 200


def test_ids_run_without_gaps(golden: list[Any]) -> None:
    assert [q.id for q in golden] == [f"q{n:03d}" for n in range(1, len(golden) + 1)]


def test_sources(golden: list[Any]) -> None:
    by_source = Counter((q.source, q.category) for q in golden)
    assert by_source[("cord", "lookup")] == 25 and by_source[("cuad", "clause")] == 20
    assert {s for s, _ in by_source} == {"synthetic", "cord", "cuad"}
    for q in golden:
        if q.source != "synthetic":  # hand-written on one real document, no graph
            assert q.check is None and len(q.relevant_document_ids) == 1, q.id
            assert not q.requires_graph and q.expected, q.id


def test_generated_questions_carry_a_known_check(golden: list[Any]) -> None:
    generated = [q for q in golden if q.check is not None]
    assert len(generated) == 104
    assert all(q.source == "synthetic" and q.check["kind"] in KINDS for q in generated)


def test_ci_subset_is_current_and_covers_every_category_and_source(golden: list[Any]) -> None:
    config = get_config().eval
    subset = load_golden_set(repo_path(config.ci_subset))
    assert len(subset) == config.ci_subset_size == 40
    assert subset == ci_subset(golden, config.ci_subset_size)  # regenerate if this fails
    full = {q.id: q for q in golden}
    assert all(full[q.id] == q for q in subset)
    assert {q.category for q in subset} == set(CATEGORIES)
    assert {q.source for q in subset} == {"synthetic", "cord", "cuad"}


@pytest.mark.parametrize(
    ("counts", "size", "expected"),
    [
        ({"a": 50, "b": 50}, 10, {"a": 5, "b": 5}),
        ({"a": 90, "b": 5, "c": 5}, 4, {"a": 2, "b": 1, "c": 1}),
        ({"a": 3, "b": 0}, 2, {"a": 2}),
    ],
)
def test_quotas(counts: dict[str, int], size: int, expected: dict[str, int]) -> None:
    assert quotas(counts, size) == expected


def test_quotas_need_room_for_every_category() -> None:
    with pytest.raises(ValueError, match="cannot cover"):
        quotas({"a": 1, "b": 1, "c": 1}, 2)


# --------------------------------------------------------------------------- the validator


def test_a_valid_row_parses() -> None:
    (question,) = parse(row())
    assert question.relevant_document_ids == (DOC,)
    assert question.source == "synthetic" and question.reviewed is False


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"id": "1"}, "id"),
        ({"category": "trivia"}, "category"),
        ({"difficulty": "extreme"}, "difficulty"),
        ({"question": "Total INV-1."}, "ends with '?'"),
        ({"relevant_document_ids": []}, "needs relevant documents"),
        ({"category": "unanswerable"}, "no relevant documents"),
        ({"relevant_document_ids": [DOC, DOC]}, "repeats"),
        ({"relevant_document_ids": ["INV-007324"]}, "not document ids"),
        ({"surprise": 1}, "surprise"),
    ],
)
def test_bad_rows_are_rejected_with_their_line(overrides: dict[str, Any], message: str) -> None:
    lines = ["", json.dumps(row(id="q001")), json.dumps(row(id="q002") | overrides)]
    with pytest.raises(GoldenSetError, match=rf"(?s):3: .*{message}"):
        parse_golden_set(lines)


def test_unanswerable_needs_no_documents() -> None:
    (question,) = parse(
        row(category="unanswerable", relevant_document_ids=[], question="Total on INV-0?")
    )
    assert question.relevant_document_ids == ()


def test_repeated_ids_and_questions_are_problems() -> None:
    questions = parse(
        row(id="q001"),
        row(id="q001"),
        row(id="q002", question=" what is the total on invoice inv-007324?"),
    )
    assert problems(questions) == [
        "id 'q001' appears more than once",
        "question 'what is the total on invoice inv-007324?' appears more than once",
    ]


def test_missing_documents() -> None:
    other = "00000000-0000-0000-0000-000000000000"
    questions = parse(row(id="q001"), row(id="q002", relevant_document_ids=[DOC, other]))

    assert missing_documents(questions, {DOC}) == {"q002": [other]}


def test_cli_reports_a_bad_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(row(id="q001")) + "\n" + json.dumps(row(id="q001")) + "\n")

    assert main([str(path)]) == 1
    assert "appears more than once" in capsys.readouterr().err


# --------------------------------------------------------------------------- the builder


def draft(expected: dict[str, Any]) -> Draft:
    return Draft("lookup", "q?", "a", (DOC,), False, "easy", expected, DOC)


@pytest.mark.parametrize(
    ("expected", "issues"),
    [
        ({"amount": "12,656.66"}, []),
        ({"amount": "9,000.00"}, []),  # printed as 9000.00
        ({"amount": "1,234.00"}, ["amount 1,234.00 not in the document"]),
        ({"number": "PO-2025-00734"}, []),
        ({"date": "2025-05-30"}, []),  # printed as 30 May 2025
        ({"date": "2025-05-31"}, ["date 2025-05-31 not in the document"]),
        ({"days": 30}, []),
        ({"days": 45}, ["45 days not in the document"]),
    ],
)
def test_grounding_check(expected: dict[str, Any], issues: list[str]) -> None:
    text = (
        "INVOICE INV-007324 Your order PO-2025-00734 Due 30 May 2025 Total Due: GBP 12,656.66 "
        "Deposit 9000.00 Payment terms: net 30 days."
    )
    assert check_grounding(draft(expected), {DOC: text}) == issues


def test_grounding_skips_multi_document_drafts_and_flags_missing_text() -> None:
    multi = Draft("multi_hop", "q?", "a", (DOC,), True, "medium", {"amount": "1.00"})
    assert check_grounding(multi, {}) == []
    assert check_grounding(draft({"amount": "1.00"}), {}) == [f"document {DOC} has no text"]


def test_periods_are_whole_quarters() -> None:
    import random

    rng = random.Random(0)
    for _ in range(50):
        label, start, end = _period(rng)
        quarter, year = int(label[1]), int(label[-4:])
        assert start == date(year, 3 * quarter - 2, 1)
        assert end.month == 3 * quarter and (end.replace(day=end.day) - start).days >= 89
