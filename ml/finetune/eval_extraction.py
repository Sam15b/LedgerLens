"""Score an extraction model on the held-out test set: field F1 and JSON validity.

    python -m ml.finetune.eval_extraction --adapter $DRIVE/ledgerlens/finetune/run-1/adapter \\
        --data data/processed/finetune/test.jsonl

The test set is ``prepare_dataset.py``'s ``test`` split: the official CORD (and SROIE, when
downloaded) test documents plus the synthetic invoice test slice. Every example names the
keys it wants; the model answers with one JSON object.

* **Field F1** -- exact match after normalization (:func:`eval.extraction_f1.normalize_value`:
  ISO dates, amounts to 2 dp, case- and punctuation-folded text), pooled over every field of
  every document (micro F1), and per field. A field the gold leaves ``null`` that the model
  fills is a false positive, so "not shown" has to be learned too. Line items count one per
  item: an item matches a gold item with the same description and amount.
* **JSON validity** -- the reply is one JSON object with exactly the requested keys and
  well-formed values (amounts and dates that parse, line items as objects with an amount).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval.extraction_f1 import DATE_FIELDS, F1Report, FieldScore, normalize_value
from ml.finetune.prepare_dataset import Example
from ml.finetune.train_lora import load_examples

logger = logging.getLogger(__name__)

AMOUNT_KEYS = frozenset({"total", "subtotal", "tax"})
LINE_ITEMS = "line_items"


@dataclass(frozen=True)
class Prediction:
    """One model's answer for one document."""

    fields: dict[str, Any] | None
    """The parsed JSON object (``None``: no object in the reply)."""
    raw: str = ""
    error: str | None = None


Predictor = Callable[[Example], Prediction]


# --------------------------------------------------------------------------- validity


def _amount_ok(value: Any) -> bool:
    return value is None or normalize_value("total", value) is not None


def _date_ok(name: str, value: Any) -> bool:
    return value is None or normalize_value(name, value) is not None


def _item_ok(item: Any) -> bool:
    return (
        isinstance(item, dict)
        and isinstance(item.get("description"), str)
        and item.get("amount") is not None
        and _amount_ok(item.get("amount"))
        and _amount_ok(item.get("unit_price"))
    )


def _items_ok(value: Any) -> bool:
    return value is None or (isinstance(value, list) and all(_item_ok(item) for item in value))


def json_problems(fields: Mapping[str, Any] | None, keys: Sequence[str]) -> list[str]:
    """Why a reply is not valid JSON for the requested ``keys`` (empty: valid)."""
    if fields is None:
        return ["no JSON object"]
    problems = []
    missing = [k for k in keys if k not in fields]
    extra = [k for k in fields if k not in keys]
    if missing:
        problems.append(f"missing {', '.join(missing)}")
    if extra:
        problems.append(f"unexpected {', '.join(extra)}")
    for key in keys:
        value = fields.get(key)
        if key in AMOUNT_KEYS and not _amount_ok(value):
            problems.append(f"{key}: not an amount")
        elif key in DATE_FIELDS and not _date_ok(key, value):
            problems.append(f"{key}: not a date")
        elif key == LINE_ITEMS and not _items_ok(value):
            problems.append("line_items: not a list of items with an amount")
    return problems


# --------------------------------------------------------------------------- scoring


def _item_key(item: Mapping[str, Any]) -> tuple[str | None, str | None]:
    return (
        normalize_value("description", item.get("description")),
        normalize_value("amount", item.get("amount")),
    )


def score_line_items(gold: Sequence[Any], predicted: Any) -> FieldScore:
    """Items matched as a multiset on (description, amount)."""
    wanted = Counter(_item_key(item) for item in gold if isinstance(item, dict))
    got: Counter[tuple[str | None, str | None]] = Counter()
    if isinstance(predicted, list):
        got = Counter(_item_key(item) for item in predicted if isinstance(item, dict))
    matched = sum((wanted & got).values())
    return FieldScore(tp=matched, fp=sum(got.values()) - matched, fn=sum(wanted.values()) - matched)


@dataclass
class ModelScore:
    """Everything one model scored over the test set."""

    model: str
    report: F1Report = field(default_factory=F1Report)
    by_source: dict[str, F1Report] = field(default_factory=dict)
    failures: Counter[str] = field(default_factory=Counter)
    """Why replies were invalid, by reason."""

    def add(self, example: Example, prediction: Prediction) -> None:
        keys = list(example.target)
        problems = json_problems(prediction.fields, keys)
        for problem in problems:
            self.failures[problem.split(":")[0]] += 1
        for report in (self.report, self.by_source.setdefault(example.source, F1Report())):
            scalar = [k for k in keys if k != LINE_ITEMS]
            report.add(example.target, prediction.fields, valid=not problems, fields=scalar)
            if LINE_ITEMS in keys:
                items = score_line_items(
                    example.target.get(LINE_ITEMS) or [],
                    (prediction.fields or {}).get(LINE_ITEMS),
                )
                total = report.fields.setdefault(LINE_ITEMS, FieldScore())
                total.tp += items.tp
                total.fp += items.fp
                total.fn += items.fn


def evaluate(
    examples: Sequence[Example],
    predictor: Predictor,
    model: str,
    *,
    on_result: Callable[[Example, Prediction], None] | None = None,
) -> ModelScore:
    """Run ``predictor`` on every example. A predictor that raises scores nothing on that
    document (invalid, every gold field missed) instead of stopping the run."""
    score = ModelScore(model)
    for example in examples:
        try:
            prediction = predictor(example)
        except Exception as exc:  # one bad document must not end the evaluation
            logger.warning("%s failed on %s", model, example.doc_id, exc_info=True)
            prediction = Prediction(None, error=f"{type(exc).__name__}: {exc}")
        score.add(example, prediction)
        if on_result is not None:
            on_result(example, prediction)
    return score


# --------------------------------------------------------------------------- models


def adapter_predictor(extractor: Any) -> Predictor:
    """Predictions from an :class:`~ml.finetune.infer.AdapterExtractor`, prompted exactly
    as in training."""

    def predict(example: Example) -> Prediction:
        generation = extractor.generate(example)
        return Prediction(generation.fields, raw=generation.text)

    return predict


def summary_line(score: ModelScore) -> str:
    micro = score.report.micro
    return (
        f"{score.model}: F1 {micro.f1:.3f} (P {micro.precision:.3f} R {micro.recall:.3f}), "
        f"JSON valid {score.report.json_validity:.1%} over {score.report.documents} documents"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.finetune.eval_extraction",
        description="Field F1 and JSON validity of an extraction model on the test split.",
    )
    parser.add_argument("--data", type=Path, default=Path("data/processed/finetune/test.jsonl"))
    parser.add_argument("--adapter", required=True, help="LoRA adapter folder")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--no-4bit", action="store_true", help="load unquantized (CPU tests)")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--limit", type=int, default=None, help="first N documents only")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from ml.finetune.infer import AdapterExtractor

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    examples = load_examples(args.data)[: args.limit]
    options = {"base_model": args.base_model} if args.base_model else {}
    extractor = AdapterExtractor(
        args.adapter, load_in_4bit=not args.no_4bit, compute_dtype=args.dtype, **options
    )
    score = evaluate(examples, adapter_predictor(extractor), extractor.name)
    print(summary_line(score))
    print(json.dumps({name: round(s.f1, 3) for name, s in score.report.fields.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
