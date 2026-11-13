"""Ingest rendered synthetic documents and score extraction against their ground truth.

    python -m eval.extraction_synthetic data/synthetic \\
        --report eval/reports/2026-11-13/extraction_synthetic.md

Every PDF listed in ``<root>/pdf/manifest.jsonl`` (``ml.synthetic.render``) goes through
the normal :class:`~ledgerlens.ingestion.pipeline.IngestionPipeline` -- load (with sha256
dedupe), layout, classify, extract, chunk; no embedding -- and the extracted fields are
compared with the record the PDF was rendered from (``records.jsonl``) using the
field-level F1 of :mod:`eval.extraction_f1`.

Reported:

* **ingestion**: documents read, failed, and caught as byte-identical duplicates;
* **classification**: predicted ``doc_type`` vs the rendered type;
* **extraction F1** per document type and per invoice layout, and per field. A misclassified
  document scores as "nothing extracted" for its true type. Only scalar fields are scored
  (line items, allocations and rate cards are tables the rules extractor leaves to the
  LLM).
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from eval.extraction_f1 import F1Report
from ledgerlens.core.schemas import DocType
from ledgerlens.extraction.base import Extractor
from ledgerlens.ingestion.pipeline import IngestionPipeline

RECORD_DOC_TYPE = {
    "invoice": DocType.INVOICE,
    "purchase_order": DocType.PURCHASE_ORDER,
    "payment": DocType.PAYMENT,
    "contract": DocType.CONTRACT,
}


@dataclass
class SyntheticRun:
    documents: int = 0
    ingested: int = 0
    duplicates: int = 0
    failed: list[str] = field(default_factory=list)
    confusion: Counter[tuple[str, str]] = field(default_factory=Counter)  # (true, predicted)
    by_type: dict[str, F1Report] = field(default_factory=dict)
    by_layout: dict[str, F1Report] = field(default_factory=dict)
    total_ms: list[float] = field(default_factory=list)
    anomalies_seen: Counter[str] = field(default_factory=Counter)

    @property
    def classification_accuracy(self) -> float:
        total = sum(self.confusion.values())
        correct = sum(n for (true, predicted), n in self.confusion.items() if true == predicted)
        return correct / total if total else 0.0


def gold_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    """The record's scalar fields (what a document of its type is scored on)."""
    return {
        k: v
        for k, v in record["data"].items()
        if k != "doc_type" and not isinstance(v, list | dict)
    }


def read_manifest(root: Path) -> list[dict[str, Any]]:
    path = root / "pdf" / "manifest.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def evaluate(
    root: Path,
    records: Mapping[str, Mapping[str, Any]],
    *,
    extractor: Extractor | None = None,
    pipeline: IngestionPipeline | None = None,
    limit: int | None = None,
) -> SyntheticRun:
    """Ingest the manifest's PDFs under ``root`` and score them against ``records`` (by id)."""
    if pipeline is None:
        from ledgerlens.extraction.rules import RulesExtractor

        pipeline = IngestionPipeline(extractor=extractor or RulesExtractor(), embedder=None)
    run = SyntheticRun()
    for entry in read_manifest(root)[:limit]:
        run.documents += 1
        record = records[entry["record_id"]]
        true_type = RECORD_DOC_TYPE[entry["doc_type"]]
        try:
            result = pipeline.process(root / entry["pdf"], embed=False)
        except ValueError as exc:  # the loader rejected the file
            run.failed.append(f"{entry['record_id']}: {exc}")
            continue
        if result.duplicate:
            run.duplicates += 1  # same bytes as a document already ingested
            run.anomalies_seen.update(entry.get("anomalies") or [])
            continue
        if result.document.error:
            run.failed.append(f"{entry['record_id']}: {result.document.error}")
            continue
        run.ingested += 1
        run.total_ms.append(result.total_ms)
        predicted_type = result.document.doc_type
        run.confusion[(true_type.value, predicted_type.value)] += 1
        extraction = result.extraction
        predicted = extraction.fields if extraction and predicted_type is true_type else None
        valid = bool(extraction and extraction.ok and predicted_type is true_type)
        gold = gold_fields(record)
        run.by_type.setdefault(true_type.value, F1Report()).add(gold, predicted, valid=valid)
        if true_type is DocType.INVOICE:
            layout = run.by_layout.setdefault(entry["template"], F1Report())
            layout.add(gold, predicted, valid=valid)
    return run


def _ms(values: Sequence[float]) -> str:
    return f"{statistics.median(values):.0f}" if values else "—"


def render_markdown(run: SyntheticRun, *, run_date: date, notes: Sequence[str] = ()) -> str:
    lines = [
        f"# Extraction on rendered synthetic PDFs — {run_date.isoformat()}",
        "",
        *notes,
        "",
        "## Ingestion",
        "",
        "| Documents | Ingested | Byte-identical duplicates (deduped) | Failed "
        "| Median ms / document |",
        "|---:|---:|---:|---:|---:|",
        f"| {run.documents} | {run.ingested} | {run.duplicates} | {len(run.failed)} "
        f"| {_ms(run.total_ms)} |",
        "",
        f"## Classification — accuracy {run.classification_accuracy:.1%}",
        "",
    ]
    predicted_types = sorted({p for _, p in run.confusion})
    lines.append("| True \\ predicted | " + " | ".join(predicted_types) + " |")
    lines.append("|---|" + "---:|" * len(predicted_types))
    for true in sorted({t for t, _ in run.confusion}):
        cells = [str(run.confusion[(true, p)]) for p in predicted_types]
        lines.append(f"| {true} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Extraction (rules extractor)",
        "",
        "| Group | Docs | Schema-valid | Micro P | Micro R | Micro F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    groups = [(t, r) for t, r in sorted(run.by_type.items())]
    groups += [(f"invoice · {name}", r) for name, r in sorted(run.by_layout.items())]
    for name, report in groups:
        m = report.micro
        lines.append(
            f"| {name} | {report.documents} | {report.json_validity:.1%} | {m.precision:.3f} "
            f"| {m.recall:.3f} | {m.f1:.3f} |"
        )
    for doc_type, report in sorted(run.by_type.items()):
        columns = [doc_type]
        reports = [report]
        if doc_type == "invoice":
            columns += sorted(run.by_layout)
            reports += [run.by_layout[name] for name in sorted(run.by_layout)]
        lines += ["", f"### F1 per field: {doc_type}", ""]
        lines.append("| Field | Support | " + " | ".join(columns) + " |")
        lines.append("|---|---:|" + "---:|" * len(columns))
        for name in sorted(report.fields):
            cells = [f"{r.fields[name].f1:.3f}" if name in r.fields else "—" for r in reports]
            support = report.fields[name].support
            lines.append(f"| {name} | {support} | " + " | ".join(cells) + " |")
    if run.failed:
        lines += ["", "## Failures", "", *(f"- {f}" for f in run.failed[:20])]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    from ml.synthetic.build import read_records

    parser = argparse.ArgumentParser(description="Score extraction on rendered synthetic PDFs")
    parser.add_argument("root", type=Path, nargs="?", default=Path("data/synthetic"))
    parser.add_argument("--records", type=Path, help="default: <root>/records.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    records = {r["id"]: r for r in read_records(args.records or args.root / "records.jsonl")}
    run = evaluate(args.root, records, limit=args.limit)
    templates = Counter(e["template"] for e in read_manifest(args.root)[: args.limit])
    report = render_markdown(
        run,
        run_date=date.today(),
        notes=[f"Rendered documents by template: {dict(sorted(templates.items()))}."],
    )
    print(report)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
    return 0 if not run.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
