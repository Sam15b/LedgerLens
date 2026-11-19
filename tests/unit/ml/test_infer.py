"""Adapter inference helper: the prompt, reply parsing and stop tokens (no model needed)."""

from typing import Any

import pytest

from ledgerlens.extraction.finetuned import (
    CORD_KEYS,
    INVOICE_KEYS,
    KEYS_BY_DOC_TYPE,
    parse_json_object,
)
from ml.finetune.infer import AdapterExtractor, Generation, build_parser, prompt_example
from ml.finetune.prepare_dataset import chat_messages


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"total": "1.00"}', {"total": "1.00"}),
        ('```json\n{"total": "1.00"}\n```', {"total": "1.00"}),
        ('Sure! {"a": {"b": 1}} trailing {"c": 2}', {"a": {"b": 1}}),
        ('{broken {"total": null}', {"total": None}),
        ("[1, 2]", None),
        ("no json", None),
    ],
)
def test_parse_json_object(text: str, expected: dict[str, Any] | None) -> None:
    assert parse_json_object(text) == expected


def test_prompt_example_asks_for_the_doc_types_keys() -> None:
    receipt = prompt_example("TOTAL 5.00", "receipt")

    assert list(receipt.target)[-4:] == list(CORD_KEYS)
    assert all(value is None for value in receipt.target.values())
    assert "receipt_date" in receipt.instruction
    assert (
        tuple(prompt_example("x", "invoice").target) == INVOICE_KEYS == KEYS_BY_DOC_TYPE["invoice"]
    )
    assert list(prompt_example("x", "invoice", ["total"]).target) == ["total"]
    # the prompt is the training prompt, without an answer
    messages = chat_messages(receipt, with_answer=False)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1]["content"].endswith("TOTAL 5.00")


def test_generation_fields() -> None:
    assert Generation('{"total": "1.00"}', 10, 5, 0.1).fields == {"total": "1.00"}
    assert Generation("I cannot help", 10, 3, 0.1).fields is None


class Tokenizer:
    eos_token_id = 128001
    unk_token_id = None

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|eot_id|>": 128009}[token]


def test_reply_stops_at_end_of_text_or_end_of_turn() -> None:
    assert AdapterExtractor().stop_ids(Tokenizer()) == [128001, 128009]


def test_names_and_defaults() -> None:
    assert AdapterExtractor("runs/run-1/adapter").name.endswith("+lora")
    base = AdapterExtractor()
    assert base.name == "meta-llama/Meta-Llama-3-8B-Instruct" and base.adapter is None
    assert base.config.load_in_4bit is True
    args = build_parser().parse_args(["--text-file", "r.txt", "--no-4bit"])
    assert (args.doc_type, args.adapter, args.no_4bit) == ("receipt", None, True)
