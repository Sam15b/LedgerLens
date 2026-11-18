"""Extraction comparison harness: validity, F1 with line items, predictors and the report."""

import json
from pathlib import Path
from typing import Any

import pytest

from ml.finetune.eval_extraction import (
    ModelScore,
    Prediction,
    adapter_predictor,
    evaluate,
    json_problems,
    llm_predictor,
    percentile,
    render_report,
    rules_predictor,
    score_line_items,
    write_predictions,
)
from ml.finetune.infer import Generation
from ml.finetune.prepare_dataset import Example

RECEIPT = Example(
    "cord",
    "cord-test-0001",
    "test",
    "receipt",
    "1 EGG TART 13,000\n2 KOPI 46,000\nTOTAL 59,000",
    {
        "line_items": [
            {"description": "EGG TART", "quantity": "1", "unit_price": None, "amount": "13000.00"},
            {"description": "KOPI", "quantity": "2", "unit_price": None, "amount": "46000.00"},
        ],
        "subtotal": None,
        "tax": None,
        "total": "59000.00",
    },
)
INVOICE = Example(
    "synthetic",
    "synthetic-SINV-1",
    "test",
    "invoice",
    "Acme Supplies Ltd\nInvoice INV-1\nTotal 120.00",
    {"invoice_number": "INV-1", "vendor_name": "Acme Supplies Ltd", "total": "120.00"},
)


# --------------------------------------------------------------------------- validity


def test_valid_reply_has_no_problems() -> None:
    assert json_problems(RECEIPT.target, list(RECEIPT.target)) == []


@pytest.mark.parametrize(
    ("fields", "problem"),
    [
        (None, "no JSON object"),
        ({"total": "1.00"}, "missing line_items, subtotal, tax"),
        ({**RECEIPT.target, "store": "x"}, "unexpected store"),
        ({**RECEIPT.target, "total": "lots"}, "total: not an amount"),
        ({**RECEIPT.target, "line_items": [{"description": "x"}]}, "line_items: not a list"),
        ({**RECEIPT.target, "line_items": "EGG TART"}, "line_items: not a list"),
    ],
)
def test_invalid_replies(fields: dict[str, Any] | None, problem: str) -> None:
    assert any(p.startswith(problem) for p in json_problems(fields, list(RECEIPT.target)))


def test_unparseable_date_is_invalid() -> None:
    problems = json_problems({"receipt_date": "someday"}, ["receipt_date"])
    assert problems == ["receipt_date: not a date"]


# --------------------------------------------------------------------------- scoring


def test_line_items_match_on_description_and_amount() -> None:
    gold = RECEIPT.target["line_items"]
    predicted = [
        {"description": "Egg Tart", "amount": "13,000"},  # same after normalization
        {"description": "KOPI", "amount": "23000.00"},  # wrong amount
        {"description": "TEH", "amount": "5000"},
    ]

    score = score_line_items(gold, predicted)

    assert (score.tp, score.fp, score.fn) == (1, 2, 1)
    assert (score_line_items(gold, None).fn, score_line_items([], gold).fp) == (2, 2)


def test_model_score_counts_scalar_fields_line_items_and_validity() -> None:
    score = ModelScore("m")
    perfect = Prediction(json.loads(json.dumps(RECEIPT.target)))
    # fills a field the gold leaves null, misses one item, and adds a key
    sloppy = Prediction(
        {**RECEIPT.target, "tax": "1.00", "line_items": RECEIPT.target["line_items"][:1], "x": 1}
    )

    score.add(RECEIPT, perfect)
    score.add(RECEIPT, sloppy)

    fields = score.report.fields
    assert (fields["total"].tp, fields["total"].fp, fields["total"].fn) == (2, 0, 0)
    assert (fields["tax"].tp, fields["tax"].fp) == (0, 1)
    assert (fields["line_items"].tp, fields["line_items"].fn) == (3, 1)
    assert score.report.json_validity == 0.5
    assert score.failures == {"unexpected x": 1}
    assert score.by_source["cord"].documents == 2


def test_evaluate_survives_a_failing_predictor_and_times_each_document() -> None:
    calls = []

    def predictor(example: Example) -> Prediction:
        calls.append(example.doc_id)
        if example is INVOICE:
            raise RuntimeError("model crashed")
        return Prediction(dict(example.target), tokens_in=100, tokens_out=20, cost_usd=0.001)

    seen: list[Prediction] = []
    score = evaluate([RECEIPT, INVOICE], predictor, "m", on_result=lambda _, p: seen.append(p))

    assert calls == ["cord-test-0001", "synthetic-SINV-1"]
    assert seen[1].error == "RuntimeError: model crashed" and seen[1].fields is None
    assert all(p.seconds >= 0 for p in seen) and len(score.seconds) == 2
    assert score.by_source["synthetic"].fields["total"].fn == 1
    assert score.report.json_validity == 0.5
    assert score.tokens_in == [100]
    assert score.usd_per_1k == pytest.approx(0.5)  # 0.001 + 0 over 2 documents


def test_unknown_price_makes_cost_unknown() -> None:
    score = ModelScore("m")
    score.add(INVOICE, Prediction(dict(INVOICE.target), cost_usd=None))
    assert score.usd_per_1k is None


def test_percentile() -> None:
    assert percentile([], 50) == 0.0
    assert percentile([3.0, 1.0, 2.0], 50) == 2.0
    assert percentile([float(i) for i in range(1, 101)], 95) == 95.0


# --------------------------------------------------------------------------- predictors


def test_rules_predictor_answers_the_examples_keys_without_json() -> None:
    prediction = rules_predictor()(INVOICE)

    assert set(prediction.fields or {}) == set(INVOICE.target)
    assert prediction.json_output is False
    score = ModelScore("rules")
    score.add(INVOICE, prediction)
    assert score.json_output is False and not score.failures


class StubLLM:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: list[Any], **kwargs: Any) -> Any:
        from ledgerlens.llm.types import LLMResponse, Usage

        self.calls.append({"messages": messages, **kwargs})
        return LLMResponse(
            text=self.text,
            provider="openai",
            model="gpt-4o",
            usage=Usage(tokens_in=300, tokens_out=40),
            cost_usd=0.00115,
        )


def test_llm_predictor_uses_the_training_prompt_in_json_mode() -> None:
    llm = StubLLM('{"invoice_number": "INV-1", "vendor_name": "Acme", "total": "120"}')

    prediction = llm_predictor("smart", client=llm)(INVOICE)

    call = llm.calls[0]
    assert (call["model"], call["json_mode"], call["temperature"]) == ("smart", True, 0.0)
    assert [m.role for m in call["messages"]] == ["system", "user"]
    assert call["messages"][1].content.endswith(INVOICE.input)
    assert prediction.fields == {"invoice_number": "INV-1", "vendor_name": "Acme", "total": "120"}
    assert (prediction.tokens_in, prediction.tokens_out, prediction.cost_usd) == (300, 40, 0.00115)


def test_adapter_predictor_reports_generated_tokens() -> None:
    class StubExtractor:
        def generate(self, example: Example) -> Generation:
            return Generation(
                '{"total": "120.00"}', prompt_tokens=250, completion_tokens=9, seconds=1
            )

    prediction = adapter_predictor(StubExtractor())(INVOICE)

    assert prediction.fields == {"total": "120.00"}
    assert (prediction.tokens_in, prediction.tokens_out, prediction.cost_usd) == (250, 9, 0.0)


# --------------------------------------------------------------------------- output


def test_report_and_predictions(tmp_path: Path) -> None:
    rows: list[tuple[str, Example, Prediction]] = []
    perfect = evaluate(
        [RECEIPT, INVOICE],
        lambda e: Prediction(dict(e.target), tokens_in=10, tokens_out=5),
        "lora",
        on_result=lambda e, p: rows.append(("lora", e, p)),
    )
    rules = evaluate([RECEIPT, INVOICE], rules_predictor(), "rules")

    report = render_report([rules, perfect], setup={"test set": "2 documents"})

    assert "| rules | " in report and "| n/a | 0.00 |" in report
    assert "| lora | 1.000 | 1.000 | 1.000 | 100.0% | 0.00 |" in report
    assert "| model | cord (n=1) | synthetic (n=1) |" in report
    assert "| line_items | 2 |" in report
    write_predictions(tmp_path / "p.jsonl", rows)
    lines = (tmp_path / "p.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["doc_id"] for line in lines] == ["cord-test-0001", "synthetic-SINV-1"]
    assert json.loads(lines[0])["tokens_in"] == 10
