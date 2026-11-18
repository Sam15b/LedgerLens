"""Score extraction models on the held-out test set: field F1 and JSON validity.

    python -m ml.finetune.eval_extraction --models rules,smart,base,lora \\
        --adapter $DRIVE/ledgerlens/finetune/run-1/adapter

Every model goes through the same harness on the same documents:

* ``rules`` -- the Phase 1 rules extractor (:class:`ledgerlens.extraction.rules.RulesExtractor`);
  it returns fields, not JSON, so it has no JSON-validity score;
* ``base`` / ``lora`` -- Llama 3 8B Instruct without / with the adapter
  (:class:`ml.finetune.infer.AdapterExtractor`), on a GPU;
* any other name -- a model router alias or ``provider/model`` (``smart`` for GPT-4o,
  ``extractor``, ...), asked with the same chat messages the adapter was trained on.

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
* **Cost per document** -- wall-clock latency (p50 / p95), prompt and completion tokens,
  and USD per 1k documents (router price table for API models; 0 for local ones).

``--report`` writes the comparison table of docs/layers/06-fine-tuning.md as markdown,
``--out`` one JSON line per model and document.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any

from eval.extraction_f1 import DATE_FIELDS, F1Report, FieldScore, normalize_value
from ml.finetune.prepare_dataset import Example, chat_messages
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
    json_output: bool = True
    """``False`` for a model that returns fields directly (rules): JSON validity is n/a."""
    seconds: float = 0.0
    """Wall-clock time for the document (set by :func:`evaluate`)."""
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = 0.0
    """``None``: an API model the price table does not know."""


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
    json_output: bool = True
    seconds: list[float] = field(default_factory=list)
    tokens_in: list[int] = field(default_factory=list)
    tokens_out: list[int] = field(default_factory=list)
    costs: list[float | None] = field(default_factory=list)

    @property
    def p50_ms(self) -> float:
        return percentile(self.seconds, 50) * 1000

    @property
    def p95_ms(self) -> float:
        return percentile(self.seconds, 95) * 1000

    @property
    def usd_per_1k(self) -> float | None:
        """USD per 1,000 documents (``None`` if any call had no known price)."""
        if not self.costs or any(cost is None for cost in self.costs):
            return None
        return sum(cost or 0.0 for cost in self.costs) / len(self.costs) * 1000

    def add(self, example: Example, prediction: Prediction) -> None:
        keys = list(example.target)
        self.json_output = self.json_output and prediction.json_output
        self.seconds.append(prediction.seconds)
        self.costs.append(prediction.cost_usd)
        if prediction.tokens_in is not None:
            self.tokens_in.append(prediction.tokens_in)
        if prediction.tokens_out is not None:
            self.tokens_out.append(prediction.tokens_out)
        problems = json_problems(prediction.fields, keys) if prediction.json_output else []
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


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile (0 for no values)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return sum(items) / len(items) if items else None


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
        start = time.perf_counter()
        try:
            prediction = predictor(example)
        except Exception as exc:  # one bad document must not end the evaluation
            logger.warning("%s failed on %s", model, example.doc_id, exc_info=True)
            prediction = Prediction(None, error=f"{type(exc).__name__}: {exc}")
        prediction = replace(prediction, seconds=time.perf_counter() - start)
        score.add(example, prediction)
        if on_result is not None:
            on_result(example, prediction)
    return score


# --------------------------------------------------------------------------- models


def _plain(value: Any) -> Any:
    """Decimals, dates and models from the rules extractor as JSON values."""
    return json.loads(json.dumps(value, default=str))


def rules_predictor() -> Predictor:
    """The rules extractor, its fields cut to the keys each example asks for."""
    from ledgerlens.extraction.rules import RulesExtractor

    extractor = RulesExtractor()

    def predict(example: Example) -> Prediction:
        result = extractor.extract(example.input, example.doc_type)
        found = _plain(result.fields)
        return Prediction({k: found.get(k) for k in example.target}, json_output=False)

    return predict


def llm_predictor(model: str, client: Any = None) -> Predictor:
    """A model behind the router (an alias like ``smart`` or ``provider/model``), prompted
    with the adapter's chat messages, in JSON mode at temperature 0."""
    from ledgerlens.llm.router import LLMClient
    from ledgerlens.llm.types import Message
    from ml.finetune.infer import parse_json_object

    llm = client or LLMClient()

    def predict(example: Example) -> Prediction:
        messages = [Message.model_validate(m) for m in chat_messages(example, with_answer=False)]
        response = llm.complete(messages, model=model, temperature=0.0, json_mode=True)
        return Prediction(
            parse_json_object(response.text),
            raw=response.text,
            tokens_in=response.usage.tokens_in,
            tokens_out=response.usage.tokens_out,
            cost_usd=response.cost_usd,
        )

    return predict


def adapter_predictor(extractor: Any) -> Predictor:
    """Predictions from an :class:`~ml.finetune.infer.AdapterExtractor`, prompted exactly
    as in training."""

    def predict(example: Example) -> Prediction:
        generation = extractor.generate(example)
        return Prediction(
            generation.fields,
            raw=generation.text,
            tokens_in=generation.prompt_tokens,
            tokens_out=generation.completion_tokens,
        )

    return predict


# --------------------------------------------------------------------------- report


def _cell(value: float | None, fmt: str) -> str:
    return "n/a" if value is None else format(value, fmt)


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def render_report(scores: Sequence[ModelScore], *, setup: Mapping[str, str]) -> str:
    """The comparison as markdown: the overall table, F1 by source and by field."""
    lines = ["# Extraction model comparison", ""]
    lines += [f"- **{key}:** {value}" for key, value in setup.items()]
    lines += ["", "## Overall (micro F1 over every field)", ""]
    lines += _table(
        [
            "model",
            "F1",
            "precision",
            "recall",
            "JSON valid",
            "$ / 1k docs",
            "p50 ms",
            "p95 ms",
            "tokens in / out",
        ],
        [
            [
                s.model,
                f"{s.report.micro.f1:.3f}",
                f"{s.report.micro.precision:.3f}",
                f"{s.report.micro.recall:.3f}",
                f"{s.report.json_validity:.1%}" if s.json_output else "n/a",
                _cell(s.usd_per_1k, ".2f"),
                f"{s.p50_ms:,.0f}",
                f"{s.p95_ms:,.0f}",
                f"{_cell(mean(s.tokens_in), ',.0f')} / {_cell(mean(s.tokens_out), ',.0f')}",
            ]
            for s in scores
        ],
    )
    sources = sorted({source for s in scores for source in s.by_source})
    lines += ["", "## F1 by source", ""]
    lines += _table(
        ["model", *(f"{source} (n={_docs(scores, source)})" for source in sources)],
        [[s.model, *(_f1(s.by_source.get(source)) for source in sources)] for s in scores],
    )
    names = list(dict.fromkeys(name for s in scores for name in s.report.fields))
    lines += ["", "## F1 by field", ""]
    lines += _table(
        ["field", "support", *(s.model for s in scores)],
        [
            [
                name,
                str(max(s.report.fields[name].support for s in scores if name in s.report.fields)),
                *(_f1_field(s, name) for s in scores),
            ]
            for name in names
        ],
    )
    failures = [s for s in scores if s.failures]
    if failures:
        lines += ["", "## Why replies were invalid", ""]
        for s in failures:
            reasons = ", ".join(f"{reason} ({n})" for reason, n in s.failures.most_common())
            lines.append(f"- **{s.model}:** {reasons}")
    return "\n".join(lines) + "\n"


def _docs(scores: Sequence[ModelScore], source: str) -> int:
    return max(s.by_source[source].documents for s in scores if source in s.by_source)


def _f1(report: F1Report | None) -> str:
    return "n/a" if report is None else f"{report.micro.f1:.3f}"


def _f1_field(score: ModelScore, name: str) -> str:
    found = score.report.fields.get(name)
    return "n/a" if found is None else f"{found.f1:.3f}"


def write_predictions(path: Path, rows: Sequence[tuple[str, Example, Prediction]]) -> None:
    """One JSON line per model and document: what it said and what it cost."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "model": model,
                "doc_id": example.doc_id,
                "source": example.source,
                "fields": prediction.fields,
                "raw": prediction.raw,
                "error": prediction.error,
                "seconds": round(prediction.seconds, 4),
                "tokens_in": prediction.tokens_in,
                "tokens_out": prediction.tokens_out,
                "cost_usd": prediction.cost_usd,
            },
            ensure_ascii=False,
            default=str,
        )
        for model, example, prediction in rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summary_line(score: ModelScore) -> str:
    micro = score.report.micro
    valid = f"{score.report.json_validity:.1%}" if score.json_output else "n/a"
    return (
        f"{score.model}: F1 {micro.f1:.3f} (P {micro.precision:.3f} R {micro.recall:.3f}), "
        f"JSON valid {valid} over {score.report.documents} documents"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.finetune.eval_extraction",
        description="Field F1 and JSON validity of an extraction model on the test split.",
    )
    parser.add_argument("--data", type=Path, default=Path("data/processed/finetune/test.jsonl"))
    parser.add_argument(
        "--models",
        default="rules",
        help="comma-separated: rules, base, lora, or a router alias / provider/model",
    )
    parser.add_argument("--adapter", default=None, help="LoRA adapter folder (for lora)")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--no-4bit", action="store_true", help="load unquantized (CPU tests)")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--limit", type=int, default=None, help="first N documents only")
    parser.add_argument("--report", type=Path, default=None, help="markdown comparison")
    parser.add_argument("--out", type=Path, default=None, help="per-document JSONL")
    return parser


def make_predictor(name: str, args: argparse.Namespace) -> Predictor:
    """The predictor for one ``--models`` entry."""
    if name == "rules":
        return rules_predictor()
    if name in {"base", "lora"}:
        from ml.finetune.infer import AdapterExtractor

        if name == "lora" and not args.adapter:
            raise SystemExit("--models lora needs --adapter")
        options = {"base_model": args.base_model} if args.base_model else {}
        extractor = AdapterExtractor(
            args.adapter if name == "lora" else None,
            load_in_4bit=not args.no_4bit,
            compute_dtype=args.dtype,
            **options,
        )
        return adapter_predictor(extractor)
    return llm_predictor(name)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    examples = load_examples(args.data)[: args.limit]
    scores, rows = [], []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:

        def keep(example: Example, prediction: Prediction, model: str = name) -> None:
            rows.append((model, example, prediction))

        score = evaluate(examples, make_predictor(name, args), name, on_result=keep)
        scores.append(score)
        print(summary_line(score))
    if args.out:
        write_predictions(args.out, rows)
    sources = Counter(example.source for example in examples)
    setup = {
        "date": date.today().isoformat(),
        "test set": f"`{args.data.as_posix()}` ({len(examples)} documents: "
        + ", ".join(f"{n} {source}" for source, n in sorted(sources.items()))
        + ")",
        "command": "`python -m ml.finetune.eval_extraction "
        + " ".join(argv if argv is not None else sys.argv[1:])
        + "`",
    }
    report = render_report(scores, setup=setup)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
        print(f"wrote {args.report}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
