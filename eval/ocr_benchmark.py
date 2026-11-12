"""Compare OCR engines by character error rate on SROIE receipts.

    python -m eval.ocr_benchmark data/raw/sroie --split test --limit 100 \\
        --engines tesseract paddle --report eval/reports/2026-11-12/ocr.md

Each receipt goes through the same path as ingestion (``layout_pages`` -> engine ->
``PageContent``), and its text is scored against the SROIE ground-truth lines with
:func:`eval.ocr_cer.corpus_cer`. An engine that is not installed is listed in the report
as skipped rather than failing the run.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from eval.datasets.sroie import SroieSample, iter_sroie
from eval.ocr_cer import cer_counts
from ledgerlens.ingestion.layout import ScannedPage, layout_pages
from ledgerlens.ingestion.loaders import sniff_mime_type
from ledgerlens.ingestion.ocr import OCREngine, OCRUnavailableError, get_engine


@dataclass
class EngineResult:
    engine: str
    errors: int = 0
    reference_chars: int = 0
    per_doc_cer: list[float] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    skipped_reason: str | None = None

    @property
    def cer(self) -> float:
        return self.errors / self.reference_chars if self.reference_chars else float("nan")


def ocr_text(engine: OCREngine, image_path: Path) -> str:
    """OCR every page of ``image_path`` the way ingestion does; lines joined by newlines."""
    mime_type = sniff_mime_type(image_path.read_bytes()[:16]) or "image/jpeg"
    lines: list[str] = []
    for page in layout_pages(image_path, mime_type):
        if isinstance(page, ScannedPage):
            content = page.to_page_content(engine.run(page.image))
            lines.extend(line.text for line in content.lines)
    return "\n".join(lines)


def evaluate(
    engine_name: str,
    samples: Sequence[SroieSample],
    *,
    make_engine: Callable[[str], OCREngine] = lambda name: get_engine(name=name),
) -> EngineResult:
    result = EngineResult(engine_name)
    try:
        engine = make_engine(engine_name)
    except OCRUnavailableError as exc:
        result.skipped_reason = str(exc)
        return result
    for sample in samples:
        start = time.perf_counter()
        try:
            text = ocr_text(engine, sample.image_path)
        except Exception as exc:  # keep going; one bad scan should not end the benchmark
            result.failed.append(f"{sample.sample_id}: {exc}")
            continue
        result.seconds.append(time.perf_counter() - start)
        counts = cer_counts(sample.text, text)
        result.errors += counts.errors
        result.reference_chars += counts.reference_chars
        result.per_doc_cer.append(counts.cer)
    return result


def render_markdown(
    results: Sequence[EngineResult], *, dataset: str, n_samples: int, run_date: date
) -> str:
    lines = [
        f"# OCR benchmark — {run_date.isoformat()}",
        "",
        f"Dataset: {dataset}, {n_samples} receipts. CER = edit distance / reference "
        "characters, after collapsing whitespace and ignoring case (`eval/ocr_cer.py`). "
        "Corpus CER weights long receipts more; median is per receipt.",
        "",
        "| Engine | Corpus CER | Median CER | p90 CER | Median s/receipt | Failed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        if r.skipped_reason is not None:
            lines.append(f"| {r.engine} | — | — | — | — | skipped: {r.skipped_reason} |")
            continue
        p90 = _quantile(r.per_doc_cer, 0.9)
        lines.append(
            f"| {r.engine} | {r.cer:.3f} | {_median(r.per_doc_cer):.3f} | {p90:.3f} "
            f"| {_median(r.seconds):.2f} | {len(r.failed)} |"
        )
    failures = [f for r in results for f in r.failed]
    if failures:
        lines += ["", "## Failures", "", *(f"- {f}" for f in failures)]
    return "\n".join(lines) + "\n"


def _median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else float("nan")


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("root", type=Path, help="SROIE folder (e.g. data/raw/sroie)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--engines", nargs="+", default=["tesseract", "paddle"])
    parser.add_argument("--report", type=Path, help="write the markdown report here")
    args = parser.parse_args(argv)

    samples = list(iter_sroie(args.root, args.split, args.limit))
    results = [evaluate(name, samples) for name in args.engines]
    report = render_markdown(
        results,
        dataset=f"SROIE {args.split} (first {len(samples)} by id)",
        n_samples=len(samples),
        run_date=date.today(),
    )
    print(report)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
