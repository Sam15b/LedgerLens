"""Fine-tuning data: source conversion, synthetic mix, leak-free splits, chat encoding."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from ml.finetune.prepare_dataset import (
    IGNORE_INDEX,
    Example,
    assign_splits,
    chat_messages,
    collect,
    cord_examples,
    cord_target,
    cord_text,
    encode,
    fit_length,
    invoice_target,
    leakage,
    mix_synthetic,
    plain_amount,
    plain_quantity,
    split_for,
    sroie_target,
    synthetic_count,
    synthetic_examples,
    write_jsonl,
)

# --------------------------------------------------------------------------- values


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("16,500", "16500.00"),
        ("23.000", "23000.00"),
        ("1.234.500", "1234500.00"),
        ("Rp 1.250,50", "1250.50"),
        ("45,500.00", "45500.00"),
        ("12.5", "12.50"),
        ("-4,500", "-4500.00"),
        ("@ 9.000", "9000.00"),
        ("no amount", None),
        (None, None),
    ],
)
def test_plain_amount(text: str | None, expected: str | None) -> None:
    assert plain_amount(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"), [("2", "2"), ("2 x", "2"), ("1.00", "1"), ("0,5", "0.5"), ("x", None)]
)
def test_plain_quantity(text: str, expected: str | None) -> None:
    assert plain_quantity(text) == expected


# --------------------------------------------------------------------------- CORD


def word(text: str, row: int, x: int, y: int) -> dict[str, Any]:
    quad = {
        "x1": x,
        "y1": y,
        "x2": x + 10,
        "y2": y,
        "x3": x + 10,
        "y3": y + 8,
        "x4": x,
        "y4": y + 8,
    }
    return {"text": text, "row_id": row, "quad": quad}


def test_cord_text_reads_rows_top_to_bottom_and_words_left_to_right() -> None:
    valid_line = [
        {"category": "total.total_price", "words": [word("45,500", 9, 200, 300)]},
        {"category": "menu.nm", "words": [word("TART", 1, 60, 100), word("EGG", 1, 20, 100)]},
        {"category": "total.total_price", "words": [word("TOTAL", 9, 20, 302)]},
    ]

    assert cord_text(valid_line) == "EGG TART\nTOTAL 45,500"


def test_cord_target_maps_menu_and_totals() -> None:
    parse = {
        "menu": [
            {"nm": "EGG TART", "cnt": "2", "price": "26,000", "unitprice": "13,000"},
            {"nm": ["ICE", "TEA"], "cnt": "1 x", "price": "5,000"},
            {"nm": "NOTE ONLY"},  # no price: not a line
        ],
        "sub_total": {"subtotal_price": "31,000", "tax_price": "3,100"},
        "total": {"total_price": "34,100", "cashprice": "50,000"},
    }

    assert cord_target(parse) == {
        "line_items": [
            {
                "description": "EGG TART",
                "quantity": "2",
                "unit_price": "13000.00",
                "amount": "26000.00",
            },
            {"description": "ICE TEA", "quantity": "1", "unit_price": None, "amount": "5000.00"},
        ],
        "subtotal": "31000.00",
        "tax": "3100.00",
        "total": "34100.00",
    }


def test_cord_target_single_item_and_missing_totals() -> None:
    target = cord_target({"menu": {"nm": "Kopi", "cnt": "1", "price": "23.000"}})

    assert target["line_items"][0]["amount"] == "23000.00"
    assert (target["subtotal"], target["tax"], target["total"]) == (None, None, None)


def write_cord(root: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    def gt(split: str, image_id: int, total: str) -> str:
        return json.dumps(
            {
                "gt_parse": {"menu": {"nm": "Kopi", "cnt": "1", "price": total}},
                "meta": {"split": split, "image_id": image_id},
                "valid_line": [{"words": [word("Kopi", 1, 0, 0), word(total, 1, 50, 0)]}],
            }
        )

    (root / "data").mkdir(parents=True)
    for split, rows in {
        "train": [gt("train", 0, "10.000"), gt("train", 1, "11.000")],
        "validation": [gt("valid", 0, "12.000")],
        "test": [gt("test", 0, "13.000")],
    }.items():
        table = pa.table({"ground_truth": rows})
        pq.write_table(table, root / "data" / f"{split}-00000-of-00001.parquet")


def test_cord_examples_keep_the_official_splits(tmp_path: Path) -> None:
    write_cord(tmp_path)

    examples = {e.doc_id: e for e in cord_examples(tmp_path)}

    assert set(examples) == {
        "cord-train-0000",
        "cord-train-0001",
        "cord-val-0000",
        "cord-test-0000",
    }
    test = examples["cord-test-0000"]
    assert (test.split, test.doc_type, test.input) == ("test", "receipt", "Kopi 13.000")
    assert json.loads(test.output)["total"] is None
    assert json.loads(test.output)["line_items"][0]["amount"] == "13000.00"
    assert "line_items, subtotal, tax, total" in test.instruction


def test_missing_cord_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(cord_examples(tmp_path))


# --------------------------------------------------------------------------- SROIE


def test_sroie_target_normalizes_date_and_total() -> None:
    target = sroie_target(
        {"company": "KEDAI ABC", "address": "JALAN 1", "date": "25/12/2018", "total": "9.00"}
    )

    assert target == {
        "vendor_name": "KEDAI ABC",
        "vendor_address": "JALAN 1",
        "receipt_date": "2018-12-25",
        "total": "9.00",
    }
    assert sroie_target({})["receipt_date"] is None


# --------------------------------------------------------------------------- synthetic


INVOICE = {
    "doc_type": "invoice",
    "invoice_number": "INV-1",
    "vendor_name": "Acme Supplies Ltd",
    "invoice_date": "2026-03-14",
    "po_number": "PO-1",
    "currency": "GBP",
    "subtotal": "100",
    "tax": "20.0",
    "total": "120",
    "line_items": [{"description": "Wrap", "quantity": "10", "unit_price": "10", "amount": "100"}],
}


def test_invoice_target_lists_every_key_with_two_decimal_amounts() -> None:
    target = invoice_target(INVOICE)

    assert list(target)[:2] == ["invoice_number", "vendor_name"]
    assert target["due_date"] is None and target["customer_name"] is None
    assert (target["subtotal"], target["tax"], target["total"]) == ("100.00", "20.00", "120.00")
    assert target["line_items"] == [
        {"description": "Wrap", "quantity": "10", "unit_price": "10.00", "amount": "100.00"}
    ]


def write_synthetic(root: Path) -> None:
    import pymupdf

    records = [
        {"record_type": "vendor", "id": "V001", "data": {}},
        {"record_type": "invoice", "id": "SINV-1", "duplicate_of": None, "data": INVOICE},
        {"record_type": "invoice", "id": "SINV-2", "duplicate_of": "SINV-1", "data": INVOICE},
    ]
    (root / "pdf" / "invoice").mkdir(parents=True)
    (root / "records.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    manifest = []
    for record_id in ("SINV-1", "SINV-2", "SINV-404"):
        pdf = f"pdf/invoice/{record_id}.pdf"
        doc = pymupdf.open()
        doc.new_page().insert_text((72, 72), f"Invoice INV-1 total 120.00 ({record_id})")
        doc.save(root / pdf)
        manifest.append({"record_id": record_id, "doc_type": "invoice", "pdf": pdf})
    manifest.append({"record_id": "SPO-1", "doc_type": "purchase_order", "pdf": "x.pdf"})
    (root / "pdf" / "manifest.jsonl").write_text("\n".join(json.dumps(m) for m in manifest))


def test_synthetic_examples_read_the_pdf_and_group_duplicates(tmp_path: Path) -> None:
    write_synthetic(tmp_path)

    examples = list(synthetic_examples(tmp_path))  # SINV-404 has no record: skipped

    assert [e.doc_id for e in examples] == ["synthetic-SINV-1", "synthetic-SINV-2"]
    assert {e.group_key for e in examples} == {"synthetic-SINV-1"}
    assert "Invoice INV-1 total 120.00" in examples[0].input
    assert examples[0].doc_type == "invoice"
    assert json.loads(examples[0].output)["po_number"] == "PO-1"


def example(doc_id: str, split: str = "train", source: str = "cord", **kwargs: Any) -> Example:
    defaults: dict[str, Any] = {"input": f"text of {doc_id}", "target": {"total": "1.00"}}
    return Example(source, doc_id, split, "receipt", **(defaults | kwargs))


def test_synthetic_count() -> None:
    assert synthetic_count(750, 0.25) == 250  # 250 / (750 + 250)
    assert synthetic_count(800, 0.0) == 0
    with pytest.raises(ValueError):
        synthetic_count(10, 1.0)


def test_mix_synthetic_is_seeded_and_sized() -> None:
    real = [example(f"r{i}") for i in range(30)]
    pool = [example(f"s{i:02d}", split="", source="synthetic") for i in range(40)]

    first = mix_synthetic(real, pool, share=0.25, seed=1)
    again = mix_synthetic(real, list(reversed(pool)), share=0.25, seed=1)

    assert [e.doc_id for e in first] == [e.doc_id for e in again]
    assert sum(e.source == "synthetic" for e in first) == 10
    assert len(mix_synthetic(real, pool[:3], share=0.25)) == 33  # fewer than wanted: all


# --------------------------------------------------------------------------- splits


def test_split_for_is_stable_and_follows_the_fractions() -> None:
    fractions = {"train": 0.8, "val": 0.1, "test": 0.1}
    counts = Counter(split_for(f"doc-{i}", fractions) for i in range(5000))

    assert split_for("doc-7", fractions) == split_for("doc-7", fractions)
    assert 0.77 < counts["train"] / 5000 < 0.83
    assert 0.08 < counts["test"] / 5000 < 0.12


def test_assign_splits_keeps_official_splits_and_groups_together() -> None:
    examples = [
        example("cord-test-0001", split="test"),
        example("cord-train-0001", split="train"),
        *[example(f"sroie-{i}", split="train", source="sroie") for i in range(200)],
        example("sroie-t1", split="test", source="sroie"),
        *[
            example(f"synthetic-{i}", split="", source="synthetic", group=f"g{i // 2}")
            for i in range(200)
        ],
    ]

    split = {e.doc_id: e.split for e in assign_splits(examples)}

    assert split["cord-test-0001"] == "test" and split["cord-train-0001"] == "train"
    assert split["sroie-t1"] == "test"
    sroie = Counter(v for k, v in split.items() if k.startswith("sroie-") and k != "sroie-t1")
    assert set(sroie) == {"train", "val"} and sroie["val"] < sroie["train"]
    synthetic = Counter(v for k, v in split.items() if k.startswith("synthetic-"))
    assert set(synthetic) == {"train", "val", "test"}
    for i in range(0, 200, 2):  # duplicates (same group) share a split
        assert split[f"synthetic-{i}"] == split[f"synthetic-{i + 1}"]


def test_same_text_in_two_splits_is_kept_only_in_the_later_one() -> None:
    examples = [
        example("cord-train-0001", split="train", input="Kopi  23.000"),
        example("cord-val-0001", split="val", input="kopi 23.000"),
        example("cord-test-0001", split="test", input="KOPI 23.000"),
        example("cord-train-0002", split="train", input="other"),
    ]

    kept = assign_splits(examples)

    assert [e.doc_id for e in kept] == ["cord-test-0001", "cord-train-0002"]
    assert leakage(kept) == []


def test_leakage_reports_documents_groups_and_text_across_splits() -> None:
    examples = [
        example("a", split="train"),
        example("a", split="test", input="different"),
        example("b", split="train", group="g"),
        example("c", split="val", group="g"),
        example("d", split="train", input="same"),
        example("e", split="test", input="same"),
    ]

    problems = leakage(examples)

    assert any(p.startswith("document 'a'") for p in problems)
    assert any(p.startswith("group 'g'") for p in problems)
    assert any(p.startswith("text 'same'") for p in problems)


def test_no_source_document_is_in_more_than_one_split(tmp_path: Path) -> None:
    write_cord(tmp_path / "cord")
    write_synthetic(tmp_path / "synthetic")

    examples = collect(tmp_path / "cord", None, tmp_path / "synthetic", share=0.3)

    assert leakage(examples) == []
    assert {e.source for e in examples} == {"cord", "synthetic"}
    assert all(e.split in {"train", "val", "test"} for e in examples)


# --------------------------------------------------------------------------- chat format


class FakeTokenizer:
    """A Llama-3-like template over whitespace "tokens": each word is one id."""

    def __init__(self, *, as_mapping: bool = False, broken: bool = False) -> None:
        self.vocab: dict[str, int] = {}
        self.as_mapping = as_mapping
        self.broken = broken

    def ids(self, text: str) -> list[int]:
        return [self.vocab.setdefault(token, len(self.vocab)) for token in text.split()]

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> Any:
        text = "<|begin_of_text|>"
        for message in conversation:
            text += f" <|{message['role']}|> {message['content']} <|eot_id|>"
        if add_generation_prompt:
            text += " <|assistant|>" + (" !" if self.broken else "")
        ids = self.ids(text)
        return {"input_ids": ids} if self.as_mapping else ids


def receipt() -> Example:
    return example("cord-val-0000", split="val", input="1 EGG TART 13,000")


def test_chat_messages() -> None:
    messages = chat_messages(receipt())

    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert messages[1]["content"].endswith("Document text:\n1 EGG TART 13,000")
    assert messages[2]["content"] == '{"total": "1.00"}'
    assert len(chat_messages(receipt(), with_answer=False)) == 2


@pytest.mark.parametrize("as_mapping", [False, True])
def test_encode_puts_the_loss_on_the_response_only(as_mapping: bool) -> None:
    tokenizer = FakeTokenizer(as_mapping=as_mapping)

    encoded = encode(receipt(), tokenizer)

    assert encoded is not None
    ids, labels = encoded["input_ids"], encoded["labels"]
    assert len(ids) == len(labels) == len(encoded["attention_mask"])
    learned = [token for token, label in zip(ids, labels, strict=True) if label != IGNORE_INDEX]
    words = {i: w for w, i in tokenizer.vocab.items()}
    assert " ".join(words[t] for t in learned) == '{"total": "1.00"} <|eot_id|>'
    assert labels[: labels.index(learned[0])] == [IGNORE_INDEX] * labels.index(learned[0])


def test_encode_drops_overlong_examples_and_rejects_a_mismatched_template() -> None:
    assert encode(receipt(), FakeTokenizer(), max_length=10) is None
    kept, dropped = fit_length([receipt()], FakeTokenizer(), max_length=10)
    assert (kept, dropped) == ([], ["cord-val-0000"])
    with pytest.raises(ValueError, match="template"):
        encode(receipt(), FakeTokenizer(broken=True))


def test_write_jsonl_writes_fields_and_chats_per_split(tmp_path: Path) -> None:
    counts = write_jsonl([receipt(), example("cord-train-0001")], tmp_path)

    assert counts == {"train": 1, "val": 1}
    row = json.loads((tmp_path / "val.jsonl").read_text(encoding="utf-8"))
    assert row["output"] == '{"total": "1.00"}' and row["split"] == "val"
    chat = json.loads((tmp_path / "val.chat.jsonl").read_text(encoding="utf-8"))
    assert chat["id"] == "cord-val-0000" and len(chat["messages"]) == 3
