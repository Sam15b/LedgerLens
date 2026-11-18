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
* **Synthetic invoices** (``data/synthetic``, ``ml.synthetic``): the rendered invoice PDFs,
  read through the ingestion layout (the text the pipeline extracts from), with the record
  they were rendered from as the target -- invoice number, PO, dates, tax and line items,
  the fields receipts lack. They make up ``--synthetic-share`` (25 %) of the examples;
  exact and near duplicates (``duplicate_of``) share a group with their original.

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
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CORD_SPLITS = {"train": "train", "validation": "val", "valid": "val", "dev": "val", "test": "test"}
SROIE_SPLITS = {"train": "train", "test": "test"}

RECEIPT_KEYS = ("vendor_name", "vendor_address", "receipt_date", "total")
CORD_KEYS = ("line_items", "subtotal", "tax", "total")
INVOICE_KEYS = (
    "invoice_number",
    "vendor_name",
    "vendor_tax_id",
    "customer_name",
    "invoice_date",
    "due_date",
    "po_number",
    "currency",
    "subtotal",
    "tax",
    "total",
    "line_items",
)
DEFAULT_SYNTHETIC_SHARE = 0.25
DEFAULT_SEED = 13


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


# --------------------------------------------------------------------------- synthetic


def _amount_2dp(value: Any) -> str | None:
    return None if value in (None, "") else f"{Decimal(str(value)):.2f}"


def invoice_target(data: Mapping[str, Any]) -> dict[str, Any]:
    """The rendered invoice's record, in :data:`INVOICE_KEYS` order, amounts to 2 places."""
    target = {key: data.get(key) for key in INVOICE_KEYS}
    for key in ("subtotal", "tax", "total"):
        target[key] = _amount_2dp(target[key])
    target["line_items"] = [
        {
            "description": line.get("description"),
            "quantity": None if line.get("quantity") is None else str(line["quantity"]),
            "unit_price": _amount_2dp(line.get("unit_price")),
            "amount": _amount_2dp(line.get("amount")),
        }
        for line in data.get("line_items") or []
    ]
    return target


def pdf_text(path: Path) -> str:
    """A PDF's text as the ingestion pipeline lays it out (text layer, blocks in order)."""
    import pymupdf

    from ledgerlens.ingestion.layout import read_text_page

    with pymupdf.open(path) as doc:
        return "\n\n".join(read_text_page(page).text for page in doc)


def synthetic_examples(root: Path) -> Iterator[Example]:
    """Every rendered synthetic invoice, not yet assigned a split (``split`` is empty)."""
    records: dict[str, dict[str, Any]] = {}
    with (root / "records.jsonl").open(encoding="utf-8") as lines:
        for line in lines:
            record = json.loads(line)
            if record.get("record_type") == "invoice":
                records[record["id"]] = record
    manifest = root / "pdf" / "manifest.jsonl"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line) if line.strip() else None
        if not entry or entry.get("doc_type") != "invoice":
            continue
        record = records.get(entry["record_id"])
        if record is None:
            logger.warning("no record for rendered invoice %s", entry["record_id"])
            continue
        yield Example(
            source="synthetic",
            doc_id=f"synthetic-{record['id']}",
            split="",
            doc_type="invoice",
            input=pdf_text(root / entry["pdf"]),
            target=invoice_target(record["data"]),
            group=f"synthetic-{record.get('duplicate_of') or record['id']}",
            meta={"template": entry.get("template"), "anomalies": entry.get("anomalies", [])},
        )


def synthetic_count(real: int, share: float) -> int:
    """How many synthetic examples make them ``share`` of ``real`` + synthetic."""
    if not 0 <= share < 1:
        raise ValueError(f"synthetic share must be in [0, 1), got {share}")
    return round(real * share / (1 - share))


def mix_synthetic(
    real: Sequence[Example],
    synthetic: Sequence[Example],
    *,
    share: float = DEFAULT_SYNTHETIC_SHARE,
    seed: int = DEFAULT_SEED,
) -> list[Example]:
    """``real`` plus a seeded sample of ``synthetic`` sized to ``share`` of the result. All
    of ``synthetic`` is used, with a warning, if there is less than that."""
    import random

    wanted = synthetic_count(len(real), share)
    pool = sorted(synthetic, key=lambda example: example.doc_id)
    if wanted > len(pool):
        logger.warning("wanted %d synthetic examples, have %d; using all", wanted, len(pool))
    picked = random.Random(seed).sample(pool, min(wanted, len(pool)))
    return [*real, *sorted(picked, key=lambda example: example.doc_id)]


# --------------------------------------------------------------------------- splits

SPLIT_ORDER = ("train", "val", "test")
DEFAULT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}
SROIE_VAL_SHARE = 0.1


def stable_fraction(key: str, seed: int = DEFAULT_SEED) -> float:
    """A number in [0, 1) fixed by ``key`` and ``seed`` (the same on every machine)."""
    import hashlib

    digest = hashlib.blake2b(f"{seed}:{key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def split_for(key: str, fractions: Mapping[str, float], seed: int = DEFAULT_SEED) -> str:
    """The split a group lands in, by its stable fraction against cumulative ``fractions``."""
    point, edge = stable_fraction(key, seed), 0.0
    for split, share in fractions.items():
        edge += share
        if point < edge:
            return split
    return list(fractions)[-1]


def assign_splits(examples: Sequence[Example], *, seed: int = DEFAULT_SEED) -> list[Example]:
    """Split by source document (by group, so duplicates stay together):

    * CORD keeps its official train / validation / test split;
    * SROIE keeps its official test split and carves val from train (no official val);
    * synthetic invoices go 80 / 10 / 10 by group.

    The held-out test set is therefore the official CORD and SROIE test documents, plus
    a synthetic test slice for the invoice-only fields.
    """
    assigned = []
    for example in examples:
        split = example.split
        if example.source == "synthetic":
            split = split_for(example.group_key, DEFAULT_FRACTIONS, seed)
        elif (
            example.source == "sroie"
            and split == "train"
            and stable_fraction(example.group_key, seed) < SROIE_VAL_SHARE
        ):
            split = "val"
        assigned.append(replace(example, split=split))
    return drop_leaks(assigned)


def text_key(text: str) -> str:
    """Whitespace- and case-insensitive identity of a document's text."""
    return " ".join(text.split()).casefold()


def drop_leaks(examples: Sequence[Example]) -> list[Example]:
    """Drop examples whose text already sits in a later split (test before val before
    train): two documents with the same text are the same document, whatever their ids."""
    rank = {split: i for i, split in enumerate(SPLIT_ORDER)}
    owner: dict[str, int] = {}
    for example in examples:
        key = text_key(example.input)
        owner[key] = max(owner.get(key, -1), rank.get(example.split, -1))
    kept = [e for e in examples if rank.get(e.split, -1) == owner[text_key(e.input)]]
    if len(kept) < len(examples):
        logger.warning(
            "dropped %d examples whose text is in a later split", len(examples) - len(kept)
        )
    return kept


def leakage(examples: Sequence[Example]) -> list[str]:
    """Every document id, group or text found in more than one split (empty: no leaks)."""
    problems = []
    for name, key in (
        ("document", lambda e: e.doc_id),
        ("group", lambda e: e.group_key),
        ("text", lambda e: text_key(e.input)),
    ):
        splits: dict[str, set[str]] = {}
        for example in examples:
            splits.setdefault(key(example), set()).add(example.split)
        problems += [
            f"{name} {value[:60]!r} is in {sorted(where)}"
            for value, where in splits.items()
            if len(where) > 1
        ]
    return problems


# --------------------------------------------------------------------------- build


def collect(
    cord: Path | None,
    sroie: Path | None,
    synthetic: Path | None = None,
    *,
    share: float = DEFAULT_SYNTHETIC_SHARE,
    seed: int = DEFAULT_SEED,
) -> list[Example]:
    """Every example from the sources that are present, synthetic mixed in at ``share``."""
    examples: list[Example] = []
    if cord is not None and cord.exists():
        examples += cord_examples(cord)
    elif cord is not None:
        logger.warning("CORD not found at %s; skipping it", cord)
    if sroie is not None and sroie.is_dir():
        examples += sroie_examples(sroie)
    elif sroie is not None:
        logger.warning("SROIE not found at %s (manual download); skipping it", sroie)
    if synthetic is not None and (synthetic / "pdf" / "manifest.jsonl").is_file():
        rendered = list(synthetic_examples(synthetic))
        examples = mix_synthetic(examples, rendered, share=share, seed=seed)
    elif synthetic is not None:
        logger.warning("no rendered synthetic documents under %s; skipping them", synthetic)
    return assign_splits(examples, seed=seed)


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
    parser.add_argument("--synthetic", type=Path, default=Path("data/synthetic"))
    parser.add_argument(
        "--synthetic-share",
        type=float,
        default=DEFAULT_SYNTHETIC_SHARE,
        help="synthetic share of all examples (0.2-0.3 recommended)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=Path("data/processed/finetune"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    examples = collect(
        args.cord, args.sroie, args.synthetic, share=args.synthetic_share, seed=args.seed
    )
    if not examples:
        logger.error("no source data found")
        return 1
    problems = leakage(examples)
    if problems:
        for problem in problems[:20]:
            logger.error("leak: %s", problem)
        return 1
    counts = write_jsonl(examples, args.out)
    sources: dict[str, dict[str, int]] = {}
    for example in examples:
        per_split = sources.setdefault(example.source, {})
        per_split[example.split] = per_split.get(example.split, 0) + 1
    print(json.dumps({"examples": counts, "by_source": sources}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
