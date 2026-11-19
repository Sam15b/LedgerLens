"""Router intent accuracy on the labelled utterance sets.

    python -m eval.router_accuracy                       # rules only (no model calls)
    python -m eval.router_accuracy --model fast          # rules, then the fast model
    python -m eval.router_accuracy --sets eval/router_utterances.jsonl --show-errors

Each line of a set is ``{"text": ..., "intent": ...}``. For every set it prints accuracy,
how many utterances each layer decided (``rules``, ``llm``, ``fallback``: the rules' best
guess when no model answered) and the rules' precision on the ones they decided.

* ``eval/router_utterances.jsonl`` (100): written together with the rules; the test
  ``tests/unit/agents/test_router.py`` holds it at >= 0.95 with the rules alone.
* ``eval/router_utterances_fresh.jsonl`` (40): written afterwards in new words, scored
  once with the rules frozen (0.50 without a model: 15 decided by rules, 12 of them
  right, 25 left for the model). Two general rule changes followed (a rules-only question
  needs question form and a ledger term), so it is no longer a clean held-out set.

The model path (``--model``) needs a working provider key for the alias.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from ledgerlens.agents.nodes.router import Router

DEFAULT_SETS = (Path("eval/router_utterances.jsonl"), Path("eval/router_utterances_fresh.jsonl"))


def load_set(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def score(router: Router, rows: Sequence[dict[str, str]]) -> dict[str, object]:
    methods: Counter[str] = Counter()
    rules_right = 0
    errors: list[tuple[str, str, str, str]] = []
    for row in rows:
        decision, method = router.decide(row["text"])
        methods[method] += 1
        right = decision.intent == row["intent"]
        rules_right += right and method == "rules"
        if not right:
            errors.append((row["text"], row["intent"], decision.intent, method))
    return {
        "n": len(rows),
        "accuracy": round(1 - len(errors) / len(rows), 3) if rows else None,
        "methods": dict(methods),
        "rules_precision": round(rules_right / methods["rules"], 3) if methods["rules"] else None,
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.router_accuracy", description=__doc__)
    parser.add_argument("--sets", nargs="+", type=Path, default=list(DEFAULT_SETS))
    parser.add_argument("--model", help="model alias for the ambiguous cases (default: none)")
    parser.add_argument("--show-errors", action="store_true")
    args = parser.parse_args(argv)
    llm = None
    if args.model:
        from ledgerlens.llm.router import LLMClient

        llm = LLMClient()
    router = Router(llm, model=args.model or "fast")
    for path in args.sets:
        result = score(router, load_set(path))
        print(
            f"{path}: accuracy {result['accuracy']} on {result['n']} "
            f"(decided by {result['methods']}; rules precision {result['rules_precision']})"
        )
        if args.show_errors:
            for text, expected, got, method in result["errors"]:  # type: ignore[attr-defined]
                print(f"  {expected:>9} -> {got:<9} [{method}] {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
