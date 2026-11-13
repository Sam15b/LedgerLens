import json
from collections.abc import Mapping
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from eval import extraction_synthetic as es
from eval.extraction_synthetic import evaluate, gold_fields, render_markdown
from ledgerlens.core.schemas import DocType
from ledgerlens.extraction.base import ExtractionResult
from ml.synthetic.build import build, write_records
from ml.synthetic.render.render_documents import render_record, select, vendors_by_id
from ml.synthetic.seed import load_vendor_master, po_invoice_records, seed_purchase_orders


@pytest.fixture(scope="module")
def rendered(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    """A small ledger rendered to PDFs (2 of each type plus an exact duplicate pair)."""
    root = tmp_path_factory.mktemp("synthetic")
    vendors = load_vendor_master()
    rows = [r.features() for r in po_invoice_records(seed_purchase_orders(vendors), vendors)]
    ledger = build(rows, invoices=200, seed=5).ledger
    write_records(ledger, root / "records.jsonl")
    records = ledger.records()
    chosen = list(select(records, ["invoice", "purchase_order", "payment", "contract"], 2))
    duplicate = next(r for r in records if r.get("anomalies") == ["exact_duplicate"])
    source = next(r for r in records if r["id"] == duplicate["duplicate_of"])
    chosen += [source, duplicate]
    by_vendor = vendors_by_id(records)
    manifest = root / "pdf" / "manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    with manifest.open("w", encoding="utf-8") as fh:
        for record in chosen:
            rendered = render_record(record, by_vendor, root, dpi=None, backend="pymupdf")
            fh.write(json.dumps(asdict(rendered)) + "\n")
    return root, {r["id"]: r for r in records}


def test_rendered_pdfs_are_ingested_classified_and_scored(
    rendered: tuple[Path, dict[str, Any]],
) -> None:
    root, records = rendered

    run = evaluate(root, records)

    assert run.documents == 10
    assert run.failed == []
    assert run.duplicates == 1  # the exact duplicate is the same bytes as its source
    assert run.anomalies_seen == {"exact_duplicate": 1}
    assert run.ingested == 9
    assert set(run.by_type) == {"invoice", "purchase_order", "payment", "contract"}
    assert run.classification_accuracy > 0.5
    assert run.by_type["invoice"].micro.f1 > 0.3
    assert sum(r.documents for r in run.by_layout.values()) == run.by_type["invoice"].documents


def test_misclassified_documents_score_as_nothing_extracted(
    rendered: tuple[Path, dict[str, Any]],
) -> None:
    root, records = rendered

    class Always:
        name = "fake"

        def extract(self, text: str, doc_type: DocType | str) -> ExtractionResult:
            return ExtractionResult.from_fields("fake", doc_type, {"total": "1.00"})

    run = evaluate(root, records, extractor=Always(), limit=4)

    report = run.by_type["invoice"]
    assert report.documents == 2 and report.valid == 0
    assert report.fields["total"].tp == 0


def test_gold_fields_keep_scalars_only() -> None:
    record: Mapping[str, Any] = {
        "data": {"doc_type": "invoice", "total": "1.00", "line_items": [{}], "po_number": None}
    }
    assert gold_fields(record) == {"total": "1.00", "po_number": None}


def test_report_has_ingestion_classification_and_per_layout_tables(
    rendered: tuple[Path, dict[str, Any]], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, records = rendered
    text = render_markdown(evaluate(root, records), run_date=date(2026, 11, 13))

    assert text.startswith("# Extraction on rendered synthetic PDFs — 2026-11-13")
    assert "| 10 | 9 | 1 | 0 |" in text
    assert "## Classification" in text and "### F1 per field: invoice" in text
    assert "invoice · invoice_" in text

    code = es.main([str(root), "--report", str(tmp_path / "r.md")])
    assert code == 0
    assert (tmp_path / "r.md").read_text(encoding="utf-8").startswith("# Extraction")
    capsys.readouterr()
