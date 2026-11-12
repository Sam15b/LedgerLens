"""Field-level extraction F1, JSON-validity rate and a per-field breakdown.

For every document and field, the predicted and gold values are normalized the same way
(:func:`normalize_value`) and compared:

* both present and equal -> true positive
* predicted but wrong, or predicted where gold has nothing -> false positive
* gold present but missed or wrong -> false negative (a wrong value counts as both)

Precision, recall and F1 are reported per field and micro-averaged over all fields.
**JSON validity** is the share of documents for which the extractor produced a record that
validated against the schema (``ExtractionResult.ok``).

Run on SROIE (key fields company / date / address / total, extracted from the
ground-truth OCR lines so that only extraction is measured)::

    python -m eval.extraction_f1 data/raw/sroie --split test --limit 100 \\
        --extractors rules llm --report eval/reports/2026-11-12/extraction.md
"""

from __future__ import annotations

import argparse
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from eval.datasets.sroie import iter_sroie
from ledgerlens.core.schemas import DocType
from ledgerlens.extraction.base import ExtractionError, ExtractionResult, Extractor
from ledgerlens.extraction.normalize import canonical_vendor, parse_amount, parse_date

AMOUNT_FIELDS = frozenset({"total", "subtotal", "tax", "amount"})
DATE_FIELDS = frozenset(
    {"invoice_date", "due_date", "receipt_date", "order_date", "delivery_date", "paid_at"}
    | {"start_date", "end_date"}
)
PARTY_FIELDS = frozenset({"vendor_name", "customer_name", "buyer_name", "payer_name"})
# SROIE key -> our Receipt field
SROIE_FIELDS = {
    "company": "vendor_name",
    "date": "receipt_date",
    "address": "vendor_address",
    "total": "total",
}
_PUNCT = re.compile(r"[^\w\s]")
_SPACES = re.compile(r"\s+")


def normalize_value(name: str, value: Any) -> str | None:
    """Comparison form of a field value, or ``None`` if it is empty or unparseable."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if name in AMOUNT_FIELDS:
        amount = value if isinstance(value, Decimal) else _amount(value)
        return None if amount is None else f"{amount.quantize(Decimal('0.01'))}"
    if name in DATE_FIELDS:
        day = value if isinstance(value, date) else parse_date(str(value))
        return None if day is None else day.isoformat()
    if name in PARTY_FIELDS:
        return canonical_vendor(str(value)) or None
    text = _SPACES.sub(" ", _PUNCT.sub(" ", str(value).casefold())).strip()
    return text or None


def _amount(value: Any) -> Decimal | None:
    if isinstance(value, int | float):
        return Decimal(str(value))
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return parse_amount(str(value))


@dataclass
class FieldScore:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def support(self) -> int:
        """Documents where gold has this field (each is either a TP or an FN)."""
        return self.tp + self.fn


@dataclass
class F1Report:
    fields: dict[str, FieldScore] = field(default_factory=dict)
    documents: int = 0
    valid: int = 0  # documents with a schema-valid record

    @property
    def micro(self) -> FieldScore:
        total = FieldScore()
        for score in self.fields.values():
            total.tp += score.tp
            total.fp += score.fp
            total.fn += score.fn
        return total

    @property
    def json_validity(self) -> float:
        return self.valid / self.documents if self.documents else 0.0

    def add(
        self,
        gold: Mapping[str, Any],
        predicted: Mapping[str, Any] | None,
        *,
        valid: bool,
        fields: Iterable[str] | None = None,
    ) -> None:
        """Score one document. ``fields`` defaults to every scalar field in ``gold``."""
        self.documents += 1
        self.valid += int(valid)
        names = list(fields) if fields is not None else _scalar_fields(gold)
        predicted = predicted or {}
        for name in names:
            g = normalize_value(name, gold.get(name))
            p = normalize_value(name, predicted.get(name))
            score = self.fields.setdefault(name, FieldScore())
            if g is not None and p == g:
                score.tp += 1
                continue
            if p is not None:
                score.fp += 1
            if g is not None:
                score.fn += 1


def _scalar_fields(record: Mapping[str, Any]) -> list[str]:
    return [k for k, v in record.items() if k != "doc_type" and not isinstance(v, list | dict)]


def render_markdown(
    reports: Mapping[str, F1Report], *, title: str, notes: Sequence[str] = ()
) -> str:
    names = list(reports)
    lines = [f"# {title}", "", *notes, ""] if notes else [f"# {title}", ""]
    lines += [
        "| Extractor | Docs | JSON valid | Micro P | Micro R | Micro F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        r = reports[name]
        m = r.micro
        lines.append(
            f"| {name} | {r.documents} | {r.json_validity:.1%} | {m.precision:.3f} "
            f"| {m.recall:.3f} | {m.f1:.3f} |"
        )
    all_fields = sorted({f for r in reports.values() for f in r.fields})
    lines += ["", "## F1 per field", "", "| Field | " + " | ".join(names) + " |"]
    lines.append("|---|" + "---:|" * len(names))
    for f in all_fields:
        cells = [f"{reports[n].fields[f].f1:.3f}" if f in reports[n].fields else "—" for n in names]
        lines.append(f"| {f} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- SROIE run


def sroie_gold(fields: Mapping[str, str]) -> dict[str, Any]:
    return {ours: fields.get(theirs) for theirs, ours in SROIE_FIELDS.items()}


@dataclass
class RunStats:
    seconds: list[float] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


def evaluate_sroie(
    extractor: Extractor, root: Path, *, split: str = "test", limit: int | None = 100
) -> tuple[F1Report, RunStats]:
    report, stats = F1Report(), RunStats()
    for sample in iter_sroie(root, split, limit):
        if not sample.fields:
            continue  # no task-3 labels for this image
        start = time.perf_counter()
        try:
            result: ExtractionResult | None = extractor.extract(sample.text, DocType.RECEIPT)
        except ExtractionError as exc:
            stats.failures.append(f"{sample.sample_id}: {exc}")
            result = None
        stats.seconds.append(time.perf_counter() - start)
        report.add(
            sroie_gold(sample.fields),
            result.fields if result else None,
            valid=bool(result and result.ok),
            fields=SROIE_FIELDS.values(),
        )
    return report, stats


def _make_extractor(name: str) -> Extractor:
    if name == "rules":
        from ledgerlens.extraction.rules import RulesExtractor

        return RulesExtractor()
    if name == "llm":
        from ledgerlens.extraction.llm_extractor import LLMExtractor

        return LLMExtractor()
    raise SystemExit(f"unknown extractor {name!r} (rules, llm)")


def main(
    argv: Sequence[str] | None = None,
    *,
    make_extractor: Callable[[str], Extractor] = _make_extractor,
) -> int:
    parser = argparse.ArgumentParser(description="Extraction F1 on SROIE")
    parser.add_argument("root", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--extractors", nargs="+", default=["rules", "llm"])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    reports: dict[str, F1Report] = {}
    notes: list[str] = []
    for name in args.extractors:
        report, stats = evaluate_sroie(
            make_extractor(name), args.root, split=args.split, limit=args.limit
        )
        reports[name] = report
        median = sorted(stats.seconds)[len(stats.seconds) // 2] if stats.seconds else 0.0
        notes.append(
            f"- `{name}`: median {median:.2f} s/receipt, {len(stats.failures)} failed calls"
        )
    text = render_markdown(
        reports,
        title=f"Extraction F1 — SROIE {args.split} (first {args.limit}), {date.today()}",
        notes=[
            "Input: SROIE ground-truth OCR lines; fields company, date, address, total, "
            "compared after normalization (`eval/extraction_f1.py`).",
            *notes,
        ],
    )
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
