"""Instruction data for fine-tuning the extractor: document text in, JSON fields out.

    python -m ml.finetune.prepare_dataset --out data/processed/finetune

Every example is one source document::

    {"id": "cord-train-0012", "source": "cord", "doc_id": "cord-train-0012",
     "split": "train", "doc_type": "receipt",
     "instruction": "Extract the receipt fields ... keys: line_items, subtotal, tax, total ...",
     "input": "<document text>",
     "output": "{\\"line_items\\": [...], \\"subtotal\\": \\"39000.00\\", ...}"}

Sources (docs/layers/06-fine-tuning.md):

* **CORD v2** (``data/raw/cord/data/*.parquet``, Hugging Face ``naver-clova-ix/cord-v2``):
  Indonesian shop receipts with line items. The text is the annotated words, read row by
  row; the target is the line items, subtotal, tax and total (CORD has no store name or
  date).
* **SROIE** (``data/raw/sroie``, manual download, see :mod:`eval.datasets.sroie`): scanned
  Malaysian receipts; the text is the annotated lines, the target is vendor, address, date
  and total. Skipped with a warning when it is not there.

Targets use the field names of :mod:`ledgerlens.core.schemas` (``Receipt``), so the
fine-tuned model's output drops into the same validation as the API extractor's. Dates are
ISO, amounts plain decimal strings with two places (``"16500.00"``); a field the source does
not label is ``null`` and still listed, so the model learns to say "not shown".
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CORD_SPLITS = {"train": "train", "validation": "val", "valid": "val", "dev": "val", "test": "test"}
SROIE_SPLITS = {"train": "train", "test": "test"}

RECEIPT_KEYS = ("vendor_name", "vendor_address", "receipt_date", "total")
CORD_KEYS = ("line_items", "subtotal", "tax", "total")


# --------------------------------------------------------------------------- examples


def instruction(doc_type: str, keys: Sequence[str]) -> str:
    """The task for one example: which document, which keys, and the value formats."""
    return (
        f"Extract the {doc_type.replace('_', ' ')} fields from the document text as one JSON "
        f"object with exactly these keys: {', '.join(keys)}. Use null for a field the text "
        "does not show. Write dates as YYYY-MM-DD and amounts as plain numbers with two "
        "decimals and no thousands separators. Line items have description, quantity, "
        "unit_price and amount."
    )


@dataclass(frozen=True)
class Example:
    """One source document as an instruction example."""

    source: str
    doc_id: str
    split: str
    doc_type: str
    input: str
    target: dict[str, Any]
    group: str = ""
    """Documents that must stay in one split (duplicates share a group); default: doc_id."""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def instruction(self) -> str:
        return instruction(self.doc_type, list(self.target))

    @property
    def output(self) -> str:
        return json.dumps(self.target, ensure_ascii=False)

    @property
    def group_key(self) -> str:
        return self.group or self.doc_id

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.doc_id,
            "source": self.source,
            "doc_id": self.doc_id,
            "split": self.split,
            "doc_type": self.doc_type,
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
        }


# --------------------------------------------------------------------------- values

_NUMBER = re.compile(r"-?\d[\d.,\s]*")


def plain_amount(text: Any) -> str | None:
    """``"16,500"`` / ``"23.000"`` / ``"Rp 1.250,50"`` -> ``"16500.00"`` / ``"23000.00"`` /
    ``"1250.50"``. A separator followed by exactly one or two final digits is the decimal
    point; any other ``.`` or ``,`` groups thousands (receipts write both ways)."""
    if text is None:
        return None
    match = _NUMBER.search(str(text))
    if match is None:
        return None
    raw = re.sub(r"\s", "", match.group()).rstrip(".,")
    negative = raw.startswith("-") or str(text).strip().startswith("-")
    digits = raw.lstrip("-")
    last = max(digits.rfind("."), digits.rfind(","))
    if last >= 0 and 1 <= len(digits) - last - 1 <= 2:
        whole, fraction = digits[:last], digits[last + 1 :]
    else:
        whole, fraction = digits, ""
    whole = re.sub(r"[.,]", "", whole) or "0"
    try:
        value = Decimal(f"{whole}.{fraction or '0'}")
    except InvalidOperation:
        return None
    return f"{-value if negative else value:.2f}"


def plain_quantity(text: Any) -> str | None:
    """``"2"`` / ``"2 x"`` / ``"1.00"`` -> ``"2"`` / ``"2"`` / ``"1"``."""
    if text is None:
        return None
    match = re.search(r"\d+(?:[.,]\d+)?", str(text))
    if match is None:
        return None
    value = Decimal(match.group().replace(",", "."))
    return f"{value.normalize():f}"


def iso_date(text: str | None) -> str | None:
    from ledgerlens.extraction.normalize import parse_date

    if not text:
        return None
    day = parse_date(text)
    return day.isoformat() if day else None


# --------------------------------------------------------------------------- CORD


def _text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(_text(v) for v in value)
    return str(value or "").strip()


def cord_text(valid_line: Sequence[Mapping[str, Any]]) -> str:
    """The receipt's annotated words, one printed row per line, top to bottom."""
    rows: dict[int, list[tuple[float, float, str]]] = {}
    for line in valid_line:
        for word in line.get("words", []):
            quad = word.get("quad", {})
            y = float(min(quad.get("y1", 0), quad.get("y2", 0)))
            x = float(quad.get("x1", 0))
            rows.setdefault(int(word.get("row_id", 0)), []).append((y, x, str(word["text"])))
    ordered = sorted(rows.values(), key=lambda words: min(y for y, _, _ in words))
    return "\n".join(
        " ".join(text for _, _, text in sorted(words, key=lambda w: w[1])) for words in ordered
    )


def _menu(parse: Mapping[str, Any]) -> list[dict[str, Any]]:
    menu = parse.get("menu") or []
    return [menu] if isinstance(menu, dict) else list(menu)


def cord_target(parse: Mapping[str, Any]) -> dict[str, Any]:
    """Line items and totals from CORD's ``gt_parse``. CORD's ``price`` is the line total."""
    items = []
    for item in _menu(parse):
        description = _text(item.get("nm"))
        amount = plain_amount(_text(item.get("price")) or None)
        if not description or amount is None:
            continue
        items.append(
            {
                "description": description,
                "quantity": plain_quantity(_text(item.get("cnt")) or None),
                "unit_price": plain_amount(_text(item.get("unitprice")) or None),
                "amount": amount,
            }
        )
    sub_total = parse.get("sub_total") or {}
    total = parse.get("total") or {}
    if isinstance(sub_total, list):
        sub_total = sub_total[0] if sub_total else {}
    if isinstance(total, list):
        total = total[0] if total else {}
    return {
        "line_items": items,
        "subtotal": plain_amount(_text(sub_total.get("subtotal_price")) or None),
        "tax": plain_amount(_text(sub_total.get("tax_price")) or None),
        "total": plain_amount(_text(total.get("total_price")) or None),
    }


def cord_examples(root: Path) -> Iterator[Example]:
    """CORD v2 receipts from the Hugging Face parquet files under ``root``."""
    import pyarrow.parquet as pq

    files = sorted((root / "data").glob("*.parquet")) or sorted(root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no CORD parquet files under {root}")
    for path in files:
        table = pq.read_table(path, columns=["ground_truth"])
        for raw in table.column("ground_truth").to_pylist():
            gt = json.loads(raw)
            meta = gt.get("meta", {})
            split = CORD_SPLITS.get(str(meta.get("split", path.stem.split("-")[0])))
            if split is None:
                continue
            doc_id = f"cord-{split}-{int(meta.get('image_id', 0)):04d}"
            yield Example(
                source="cord",
                doc_id=doc_id,
                split=split,
                doc_type="receipt",
                input=cord_text(gt.get("valid_line", [])),
                target=cord_target(gt.get("gt_parse", {})),
            )


# --------------------------------------------------------------------------- SROIE


def sroie_target(fields: Mapping[str, str]) -> dict[str, Any]:
    return {
        "vendor_name": fields.get("company") or None,
        "vendor_address": fields.get("address") or None,
        "receipt_date": iso_date(fields.get("date")),
        "total": plain_amount(fields.get("total")),
    }


def sroie_examples(root: Path) -> Iterator[Example]:
    """SROIE receipts with key-field labels, from the official train and test folders."""
    from eval.datasets.sroie import iter_sroie

    for folder, split in SROIE_SPLITS.items():
        if not (root / folder).is_dir():
            logger.warning("SROIE %s split not found under %s", folder, root)
            continue
        for sample in iter_sroie(root, folder):
            if not sample.fields:
                continue  # no task-3 labels: nothing to learn
            yield Example(
                source="sroie",
                doc_id=f"sroie-{sample.sample_id}",
                split=split,
                doc_type="receipt",
                input=sample.text,
                target=sroie_target(sample.fields),
            )


# --------------------------------------------------------------------------- build


def collect(cord: Path | None, sroie: Path | None) -> list[Example]:
    """Every example from the sources that are present."""
    examples: list[Example] = []
    if cord is not None and cord.exists():
        examples += cord_examples(cord)
    elif cord is not None:
        logger.warning("CORD not found at %s; skipping it", cord)
    if sroie is not None and sroie.is_dir():
        examples += sroie_examples(sroie)
    elif sroie is not None:
        logger.warning("SROIE not found at %s (manual download); skipping it", sroie)
    return examples


def write_jsonl(examples: Iterable[Example], out: Path) -> dict[str, int]:
    """``out/<split>.jsonl`` for each split; returns the count per split."""
    out.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[str]] = {}
    for example in examples:
        line = json.dumps(example.to_json(), ensure_ascii=False)
        by_split.setdefault(example.split, []).append(line)
    for split, lines in by_split.items():
        (out / f"{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {split: len(lines) for split, lines in sorted(by_split.items())}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.finetune.prepare_dataset",
        description="Build instruction JSONL for fine-tuning the extractor.",
    )
    parser.add_argument("--cord", type=Path, default=Path("data/raw/cord"))
    parser.add_argument("--sroie", type=Path, default=Path("data/raw/sroie"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/finetune"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    examples = collect(args.cord, args.sroie)
    if not examples:
        logger.error("no source data found")
        return 1
    counts = write_jsonl(examples, args.out)
    print(json.dumps({"examples": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
