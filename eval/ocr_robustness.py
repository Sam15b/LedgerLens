"""OCR robustness: CER and extraction F1 on clean vs degraded scans, per OCR engine.

    python -m eval.ocr_robustness data/synthetic \\
        --variants data/synthetic/degraded/classical/manifest.jsonl \\
        --engines tesseract paddle --limit 100 \\
        --report eval/reports/2026-11-13/ocr_degraded.md

Inputs are the rendered synthetic documents (``ml.synthetic.render``) and one or more
degraded-variant manifests (``ml.synthetic.diffusion.augment`` for classical
degradations, ``degrade_scans`` for Stable Diffusion img2img). Every image -- the clean
page and each of its variants -- goes through the same OCR path as ingestion
(:func:`eval.ocr_benchmark.ocr_text`), and is scored three ways:

* **CER** against the text layer of the PDF the page was rendered from -- exactly the text
  on the page -- read through ingestion's own line grouping, so both sides share a
  reading order, and separator lines ("-----") dropped from both (``eval.ocr_cer``;
  corpus CER weights long pages more);
* **extraction F1**: the rules extractor on the OCR text, with the page's true document
  type, against the ledger record (``eval.extraction_f1``) -- so the number moves with OCR
  quality only, not with classification;
* **key fields readable**: the share of pages whose document number, PO reference and
  total are all read exactly (the validity filter of ``degrade_scans``).

Results are grouped by ``(engine, method, strength)``; the clean pages are method
``clean``, strength 0. An engine that is not installed is reported as skipped.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from eval.extraction_f1 import F1Report
from eval.extraction_synthetic import gold_fields
from eval.ocr_benchmark import ocr_text
from eval.ocr_cer import cer_counts
from ledgerlens.extraction.base import Extractor
from ledgerlens.ingestion.ocr import OCREngine, OCRUnavailableError, get_engine

RECORD_TYPES = {"invoice": "invoice", "purchase_order": "purchase_order", "payment": "payment"}
_RULE_LINE = re.compile(r"^[\s\-_=*.~·]*$")


def without_rule_lines(text: str) -> str:
    """``text`` minus separator lines ("-------"): they carry no information, and OCR
    engines often drop thin light-gray rules, which would otherwise dominate CER on
    layouts that use them."""
    return "\n".join(line for line in text.splitlines() if not _RULE_LINE.match(line))


@dataclass(frozen=True)
class Item:
    """One image to OCR, and what it shows."""

    image: Path
    record_id: str
    doc_type: str  # record type (folder name under pages/)
    method: str  # "clean" | "classical" | "sd_img2img"
    strength: float


@dataclass
class Condition:
    """Scores for one ``(engine, method, strength)`` group."""

    engine: str
    method: str
    strength: float
    errors: int = 0
    reference_chars: int = 0
    per_page_cer: list[float] = field(default_factory=list)
    f1: F1Report = field(default_factory=F1Report)
    readable: int = 0
    seconds: list[float] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    @property
    def pages(self) -> int:
        return len(self.per_page_cer)

    @property
    def cer(self) -> float:
        return self.errors / self.reference_chars if self.reference_chars else float("nan")

    @property
    def readable_rate(self) -> float:
        return self.readable / self.pages if self.pages else float("nan")


@dataclass
class EngineRun:
    engine: str
    conditions: list[Condition] = field(default_factory=list)
    skipped_reason: str | None = None

    def condition(self, method: str, strength: float) -> Condition:
        return next(c for c in self.conditions if (c.method, c.strength) == (method, strength))


def record_id_of(page: Path) -> str:
    """``SINV-000001-p1.png`` -> ``SINV-000001``."""
    return page.stem.rsplit("-p", 1)[0]


def load_items(
    variant_manifests: Sequence[Path], *, limit: int | None = None, kept_only: bool = False
) -> list[Item]:
    """Clean pages and their variants from the manifests.

    ``limit`` caps the number of distinct clean pages (their variants come with them);
    ``kept_only`` drops variants a validity filter marked ``kept: false``.
    """
    variants: list[dict[str, Any]] = []
    for path in variant_manifests:
        lines = path.read_text(encoding="utf-8").splitlines()
        variants += [json.loads(line) for line in lines if line.strip()]
    sources = list(dict.fromkeys(v["source"] for v in variants))[:limit]
    chosen = set(sources)
    items = [
        Item(Path(s), record_id_of(Path(s)), Path(s).parent.name, "clean", 0.0) for s in sources
    ]
    for v in variants:
        if v["source"] not in chosen or (kept_only and v.get("kept") is False):
            continue
        source = Path(v["source"])
        items.append(
            Item(
                Path(v["image"]),
                record_id_of(source),
                source.parent.name,
                str(v["method"]),
                float(v["strength"]),
            )
        )
    return items


def evaluate_engine(
    engine_name: str,
    items: Sequence[Item],
    records: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, str],
    *,
    make_engine: Callable[[str], OCREngine] = lambda name: get_engine(name=name),
    extractor: Extractor | None = None,
    workers: int = 4,
) -> EngineRun:
    """OCR and score every item with one engine (``workers`` pages at a time)."""
    from ledgerlens.extraction.rules import RulesExtractor
    from ml.synthetic.diffusion.degrade_scans import key_fields, missing_fields

    run = EngineRun(engine_name)
    try:
        engine = make_engine(engine_name)
    except OCRUnavailableError as exc:
        run.skipped_reason = str(exc)
        return run
    extractor = extractor or RulesExtractor()

    def ocr(item: Item) -> tuple[Item, str | None, float, str | None]:
        start = time.perf_counter()
        try:
            return item, ocr_text(engine, item.image), time.perf_counter() - start, None
        except Exception as exc:  # one unreadable image must not end the benchmark
            return item, None, time.perf_counter() - start, str(exc)

    groups: dict[tuple[str, float], Condition] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(ocr, items))
    for item, text, seconds, error in results:
        key = (item.method, item.strength)
        condition = groups.setdefault(key, Condition(engine_name, *key))
        if text is None:
            condition.failed.append(f"{item.image.name}: {error}")
            continue
        record = records[item.record_id]
        counts = cer_counts(
            without_rule_lines(references[item.record_id]), without_rule_lines(text)
        )
        condition.errors += counts.errors
        condition.reference_chars += counts.reference_chars
        condition.per_page_cer.append(counts.cer)
        condition.seconds.append(seconds)
        extraction = extractor.extract(text, RECORD_TYPES.get(item.doc_type, item.doc_type))
        condition.f1.add(gold_fields(record), extraction.fields, valid=extraction.ok)
        condition.readable += not missing_fields(key_fields(record), text)
    run.conditions = sorted(
        groups.values(), key=lambda c: (c.method != "clean", c.method, c.strength)
    )
    return run


def reference_text(pdf: Path) -> str:
    """The PDF's text layer read the way ingestion reads it.

    Lines come from the same geometric grouping (``layout.group_lines``) that OCR words go
    through, so reference and OCR text share a reading order and CER measures recognition
    errors -- not the PDF's internal order, which lists table cells one per line.
    """
    from ledgerlens.ingestion.layout import PageContent, layout_pages

    pages = layout_pages(pdf, "application/pdf")
    return "\n".join(
        line.text for page in pages if isinstance(page, PageContent) for line in page.lines
    )


def reference_texts(items: Sequence[Item], root: Path) -> dict[str, str]:
    """:func:`reference_text` of each item's rendered PDF, by record id."""
    out: dict[str, str] = {}
    for item in items:
        if item.record_id not in out:
            out[item.record_id] = reference_text(
                root / "pdf" / item.doc_type / f"{item.record_id}.pdf"
            )
    return out


# --------------------------------------------------------------------------- report


def _median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else float("nan")


def render_markdown(runs: Sequence[EngineRun], *, run_date: date, notes: Sequence[str] = ()) -> str:
    lines = [
        f"# OCR robustness: clean vs degraded scans — {run_date.isoformat()}",
        "",
        *notes,
        "",
        "| Engine | Method | Strength | Pages | Corpus CER | Median CER | Extraction F1 "
        "| F1 vs clean | Key fields readable | Median s/page | Failed |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        if run.skipped_reason is not None:
            lines.append(
                f"| {run.engine} | — | — | — | — | — | — | — | — | — | skipped: "
                f"{run.skipped_reason} |"
            )
            continue
        clean = next((c for c in run.conditions if c.method == "clean"), None)
        clean_f1 = clean.f1.micro.f1 if clean else float("nan")
        for c in run.conditions:
            f1 = c.f1.micro.f1
            relative = f1 / clean_f1 if clean_f1 else float("nan")
            lines.append(
                f"| {run.engine} | {c.method} | {c.strength:.2f} | {c.pages} | {c.cer:.3f} "
                f"| {_median(c.per_page_cer):.3f} | {f1:.3f} | {relative:.0%} "
                f"| {c.readable_rate:.0%} | {_median(c.seconds):.2f} | {len(c.failed)} |"
            )
    failures = [f for run in runs for c in run.conditions for f in c.failed]
    if failures:
        lines += ["", "## Failures", "", *(f"- {f}" for f in failures[:20])]
    return "\n".join(lines) + "\n"


def summary_json(runs: Sequence[EngineRun]) -> list[dict[str, Any]]:
    return [
        {
            "engine": run.engine,
            "method": c.method,
            "strength": c.strength,
            "pages": c.pages,
            "cer": c.cer,
            "median_cer": _median(c.per_page_cer),
            "f1": c.f1.micro.f1,
            "readable": c.readable_rate,
        }
        for run in runs
        for c in run.conditions
    ]


def main(argv: Sequence[str] | None = None) -> int:
    from ml.synthetic.build import read_records

    parser = argparse.ArgumentParser(description="OCR CER and extraction F1, clean vs degraded")
    parser.add_argument("root", type=Path, nargs="?", default=Path("data/synthetic"))
    parser.add_argument("--variants", type=Path, nargs="+", required=True, help="manifests")
    parser.add_argument("--engines", nargs="+", default=["tesseract", "paddle"])
    parser.add_argument("--limit", type=int, help="at most this many clean pages")
    parser.add_argument("--kept-only", action="store_true", help="skip filtered-out variants")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--json", type=Path, help="also write the per-condition numbers")
    args = parser.parse_args(argv)

    items = load_items(args.variants, limit=args.limit, kept_only=args.kept_only)
    records = {r["id"]: r for r in read_records(args.root / "records.jsonl")}
    references = reference_texts(items, args.root)
    runs = [
        evaluate_engine(name, items, records, references, workers=args.workers)
        for name in args.engines
    ]
    pages = sum(1 for i in items if i.method == "clean")
    report = render_markdown(
        runs,
        run_date=date.today(),
        notes=[f"{pages} clean pages and {len(items) - pages} degraded variants."],
    )
    print(report)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary_json(runs), indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
