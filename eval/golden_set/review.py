"""Review drafted golden questions into the golden set: accept, edit or reject each one.

    python -m eval.golden_set.review data/golden/golden_set.draft.jsonl --verify

For each draft not decided yet, the CLI shows the question, the answer, the facts an answer
must state, the evidence documents and, with ``--verify``, whether the graph derives the
same answer (``verify.py``). Then:

* ``a`` accept -- append it to the golden set (``--into``) as ``reviewed: true``, with the
  next free id;
* ``e`` edit -- type a new question and/or answer (Enter keeps the old one), then accept;
* ``r`` reject -- keep it out;
* ``s`` skip -- decide later;
* ``q`` quit -- stop here; the decisions so far are kept.

Decisions are logged next to the drafts (``<drafts>.decisions.jsonl``), so a review can stop
and resume. A draft whose question is already in the golden set is never shown again.

``--accept-verified`` accepts, without asking, every draft whose graph check agrees, and
leaves the others for an interactive pass. It is for drafts made from templates over known
ground truth; never use it for LLM-drafted questions, which a person always reads
(docs/layers/09-evaluation.md).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from eval.golden_set.schema import (
    DEFAULT_PATH,
    GoldenQuestion,
    dump_golden_set,
    load_golden_set,
    problems,
    summary,
)
from eval.golden_set.verify import Verdict

Decision = Literal["accept", "edit", "reject", "skip", "quit"]
KEYS: dict[str, Decision] = {
    "a": "accept",
    "e": "edit",
    "r": "reject",
    "s": "skip",
    "q": "quit",
}
Ask = Callable[[str], str]


def _key(question: GoldenQuestion | str) -> str:
    text = question if isinstance(question, str) else question.question
    return text.strip().lower()


def show(draft: GoldenQuestion, verdict: Verdict | None) -> str:
    lines = [
        f"[{draft.category} / {draft.difficulty}] {draft.question}",
        f"  answer:   {draft.ground_truth_answer}",
        f"  expected: {json.dumps(draft.expected, ensure_ascii=False)}",
        f"  evidence: {len(draft.relevant_document_ids)} document(s)"
        + (f", first {draft.relevant_document_ids[0]}" if draft.relevant_document_ids else ""),
    ]
    if verdict is not None:
        lines.append(
            f"  graph:    {'agrees' if verdict.ok else 'DISAGREES: ' + verdict.detail}"
            f" ({verdict.kind})"
        )
    return "\n".join(lines)


@dataclass
class Review:
    """The golden set as it grows, and every decision made so far (by question text)."""

    golden: list[GoldenQuestion]
    decisions: dict[str, str] = field(default_factory=dict)
    log: list[dict[str, str]] = field(default_factory=list)

    def next_id(self) -> str:
        return f"q{max((int(q.id[1:]) for q in self.golden), default=0) + 1:03d}"

    def pending(self, drafts: Sequence[GoldenQuestion]) -> list[GoldenQuestion]:
        inside = {_key(q) for q in self.golden}
        return [d for d in drafts if _key(d) not in inside and _key(d) not in self.decisions]

    def decide(self, draft: GoldenQuestion, decision: str, note: str = "") -> None:
        self.decisions[_key(draft)] = decision
        self.log.append({"question": draft.question, "decision": decision, "note": note})

    def accept(self, draft: GoldenQuestion, *, note: str = "") -> GoldenQuestion:
        accepted = draft.model_copy(update={"id": self.next_id(), "reviewed": True})
        accepted = GoldenQuestion.model_validate(accepted.model_dump())  # re-validate edits
        self.golden.append(accepted)
        self.decide(draft, "accept", note)
        return accepted


def edited(draft: GoldenQuestion, ask: Ask) -> GoldenQuestion:
    question = ask("  new question (Enter keeps it): ").strip() or draft.question
    answer = ask("  new answer (Enter keeps it): ").strip() or draft.ground_truth_answer
    return draft.model_copy(update={"question": question, "ground_truth_answer": answer})


def run_review(
    review: Review,
    drafts: Sequence[GoldenQuestion],
    *,
    ask: Ask,
    say: Callable[[str], None] = print,
    verdicts: Mapping[str, Verdict] | None = None,
    accept_verified: bool = False,
) -> Review:
    """Go through the pending drafts; ``verdicts`` are graph checks by draft id."""
    verdicts = verdicts or {}
    for draft in review.pending(drafts):
        verdict = verdicts.get(draft.id)
        if accept_verified:
            if verdict is not None and verdict.ok:
                review.accept(draft, note="graph check agrees")
            continue
        say(show(draft, verdict))
        choice = None
        while choice is None:
            choice = KEYS.get(ask("  [a]ccept [e]dit [r]eject [s]kip [q]uit: ").strip().lower())
        if choice == "quit":
            break
        if choice == "skip":
            continue
        if choice == "reject":
            review.decide(draft, "reject", ask("  why (optional): ").strip())
        elif choice == "edit":
            changed = edited(draft, ask)
            if _key(changed) in {_key(q) for q in review.golden}:
                say("  that question is already in the set; not accepted")
                continue
            review.accept(changed, note="edited")
            review.decisions[_key(draft)] = "edit"
        else:
            review.accept(draft)
    return review


# --------------------------------------------------------------------------- files


def decisions_path(drafts: Path) -> Path:
    return drafts.with_name(drafts.stem + ".decisions.jsonl")


def load_decisions(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {_key(row["question"]): row["decision"] for row in rows}


def append_decisions(path: Path, log: Sequence[Mapping[str, str]]) -> None:
    with path.open("a", encoding="utf-8") as out:
        for row in log:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Review drafted golden questions.")
    parser.add_argument("drafts", type=Path)
    parser.add_argument("--into", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--verify", action="store_true", help="check each draft on the graph")
    parser.add_argument(
        "--accept-verified",
        action="store_true",
        help="accept drafts whose graph check agrees, without asking (implies --verify)",
    )
    args = parser.parse_args(argv)
    drafts = load_golden_set(args.drafts)
    review = Review(load_golden_set(args.into), load_decisions(decisions_path(args.drafts)))
    verdicts: dict[str, Verdict] = {}
    if args.verify or args.accept_verified:
        from eval.golden_set.verify import neo4j_cypher, postgres_contract_text, verify

        pending = review.pending(drafts)
        found = verify(pending, neo4j_cypher(), postgres_contract_text())
        verdicts = {v.question_id: v for v in found}
        agree = sum(v.ok for v in found)
        print(f"graph checks: {agree}/{len(found)} agree ({len(pending)} pending drafts)")
    before = len(review.golden)
    try:
        run_review(
            review, drafts, ask=input, verdicts=verdicts, accept_verified=args.accept_verified
        )
    except (EOFError, KeyboardInterrupt):
        print()
    found_problems = problems(review.golden)
    if found_problems:
        for problem in found_problems:
            print(f"PROBLEM {problem}")
        return 1
    dump_golden_set(review.golden, args.into)
    append_decisions(decisions_path(args.drafts), review.log)
    print(f"accepted {len(review.golden) - before}; {len(review.pending(drafts))} still pending")
    print(summary(review.golden))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
