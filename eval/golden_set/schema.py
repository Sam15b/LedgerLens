"""The golden set: reviewed questions with known answers, for the retrieval benchmark.

One JSON object per line in ``data/golden/golden_set.jsonl`` (see ``schema.md``)::

    {"id": "q001", "category": "lookup", "question": "What is the total on invoice INV-007324?",
     "ground_truth_answer": "12,656.66 GBP", "relevant_document_ids": ["22bac600-..."],
     "requires_graph": false, "difficulty": "easy",
     "expected": {"amount": "12,656.66", "currency": "GBP"}, "source": "synthetic",
     "reviewed": true}

Validate a file (and, with ``--store``, that every relevant document exists in Postgres)::

    python -m eval.golden_set.schema data/golden/golden_set.jsonl --store
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Category = Literal["lookup", "clause", "multi_hop", "aggregation", "reconciliation", "unanswerable"]
CATEGORIES: tuple[str, ...] = (
    "lookup",
    "clause",
    "multi_hop",
    "aggregation",
    "reconciliation",
    "unanswerable",
)
Difficulty = Literal["easy", "medium", "hard"]

DEFAULT_PATH = Path("data/golden/golden_set.jsonl")
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class GoldenSetError(ValueError):
    """A golden-set file that does not parse or validate."""


class GoldenQuestion(BaseModel):
    """One question with its known answer and the documents that hold the evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^q\d{3,}$")
    category: Category
    question: str = Field(min_length=10)
    ground_truth_answer: str = Field(min_length=1)
    relevant_document_ids: tuple[str, ...] = ()
    """Documents (``documents.id``) whose text or graph records hold the answer, most
    important first. Empty only for ``unanswerable`` questions."""
    requires_graph: bool
    """The answer depends on links between records (or a total over many of them)."""
    difficulty: Difficulty
    expected: dict[str, Any] = Field(default_factory=dict)
    """Key facts a correct answer must state (amounts, numbers, dates), for automatic
    answer checks. Amounts are strings as printed (``"12,656.66"``)."""
    source: str = "synthetic"
    reviewed: bool = False
    check: dict[str, Any] | None = None
    """How to derive the answer again from the stores, independently of the builder
    (``{"kind": "settled_by", "invoice_id": ...}``, see ``verify.py``). Generated
    questions carry one; hand-written ones and golden set v0 do not."""

    @model_validator(mode="after")
    def _evidence(self) -> GoldenQuestion:
        documents = self.relevant_document_ids
        if self.category == "unanswerable" and documents:
            raise ValueError("an unanswerable question has no relevant documents")
        if self.category != "unanswerable" and not documents:
            raise ValueError(f"a {self.category} question needs relevant documents")
        if len(set(documents)) != len(documents):
            raise ValueError("relevant_document_ids repeats a document")
        bad = [doc for doc in documents if not _UUID.match(doc)]
        if bad:
            raise ValueError(f"not document ids (lower-case UUIDs): {bad}")
        if not self.question.strip().endswith("?"):
            raise ValueError("a question ends with '?'")
        return self


def parse_golden_set(lines: Iterable[str], *, source: str = "<golden set>") -> list[GoldenQuestion]:
    """Questions from JSON lines (blank lines skipped); errors name the line."""
    questions: list[GoldenQuestion] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            questions.append(GoldenQuestion.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise GoldenSetError(f"{source}:{number}: {exc}") from exc
    return questions


def load_golden_set(path: Path | str = DEFAULT_PATH) -> list[GoldenQuestion]:
    path = Path(path)
    return parse_golden_set(path.read_text(encoding="utf-8").splitlines(), source=str(path))


def dump_golden_set(questions: Sequence[GoldenQuestion], path: Path | str) -> None:
    """Write one question per line, fields in schema order."""
    lines = [q.model_dump_json(exclude={"check"} if q.check is None else None) for q in questions]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def problems(questions: Sequence[GoldenQuestion]) -> list[str]:
    """Set-level problems: repeated ids or questions. (Each question validates itself.)"""
    found: list[str] = []
    for kind, values in (
        ("id", [q.id for q in questions]),
        ("question", [q.question.strip().lower() for q in questions]),
    ):
        repeated = sorted(value for value, n in Counter(values).items() if n > 1)
        found += [f"{kind} {value!r} appears more than once" for value in repeated]
    return found


def missing_documents(questions: Sequence[GoldenQuestion], known: set[str]) -> dict[str, list[str]]:
    """Question id -> its relevant documents that are not in ``known``."""
    missing = {q.id: [d for d in q.relevant_document_ids if d not in known] for q in questions}
    return {qid: docs for qid, docs in missing.items() if docs}


def store_document_ids() -> set[str]:
    """Ids of the ready documents in Postgres (``DATABASE_URL``)."""
    from sqlalchemy import select

    from ledgerlens.storage.postgres.db import get_session_factory
    from ledgerlens.storage.postgres.models import DocumentRow

    with get_session_factory()() as session:
        rows = session.scalars(select(DocumentRow.id).where(DocumentRow.status == "ready"))
        return set(rows)


def summary(questions: Sequence[GoldenQuestion]) -> str:
    """Counts by category and difficulty, one line each."""
    by_category: Counter[str] = Counter(q.category for q in questions)
    by_difficulty = Counter(q.difficulty for q in questions)
    categories = ", ".join(f"{c} {by_category[c]}" for c in CATEGORIES if by_category[c])
    difficulties = ", ".join(f"{d} {n}" for d, n in sorted(by_difficulty.items()))
    graph = sum(q.requires_graph for q in questions)
    return (
        f"{len(questions)} questions: {categories}\n"
        f"difficulty: {difficulties}; requires_graph: {graph}; "
        f"reviewed: {sum(q.reviewed for q in questions)}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_PATH)
    parser.add_argument(
        "--store", action="store_true", help="also check every document exists in Postgres"
    )
    args = parser.parse_args(argv)
    try:
        questions = load_golden_set(args.path)
    except GoldenSetError as exc:
        print(exc, file=sys.stderr)
        return 1
    errors = problems(questions)
    if args.store:
        missing = missing_documents(questions, store_document_ids())
        errors += [f"{qid}: documents not in the store: {docs}" for qid, docs in missing.items()]
    print(summary(questions))
    for error in errors:
        print(f"ERROR {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
