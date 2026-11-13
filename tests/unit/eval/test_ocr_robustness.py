import json
from datetime import date
from pathlib import Path
from typing import Any, cast

import pytest
from PIL import Image

from eval import ocr_robustness as rob
from eval.ocr_robustness import (
    evaluate_engine,
    load_items,
    reference_text,
    reference_texts,
    render_markdown,
    without_rule_lines,
)
from ledgerlens.ingestion.ocr import OCREngine, OCRUnavailableError
from ml.synthetic.build import build, write_records
from ml.synthetic.render.render_documents import render_record, vendors_by_id
from ml.synthetic.seed import load_vendor_master, po_invoice_records, seed_purchase_orders


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any], Path]:
    """Three rendered invoices, a classical-variant manifest for them (2 severities)."""
    root = tmp_path_factory.mktemp("synthetic")
    vendors = load_vendor_master()
    rows = [r.features() for r in po_invoice_records(seed_purchase_orders(vendors), vendors)]
    ledger = build(rows, invoices=100, seed=2).ledger
    write_records(ledger, root / "records.jsonl")
    records = ledger.records()
    invoices = [r for r in records if r["record_type"] == "invoice"][:3]
    manifest = root / "degraded" / "manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    lines = []
    for record in invoices:
        rendered = render_record(record, vendors_by_id(records), root, dpi=20)
        source = root / rendered.pages[0]
        for strength in (0.5, 1.0):
            image = root / "degraded" / f"{source.stem}-c{strength:.2f}.png"
            Image.new("L", (10, 10), 255).save(image)
            lines.append(
                {
                    "source": source.as_posix(),
                    "image": image.as_posix(),
                    "method": "classical",
                    "strength": strength,
                    "seed": 0,
                }
            )
    manifest.write_text("".join(json.dumps(v) + "\n" for v in lines), encoding="utf-8")
    return root, {r["id"]: r for r in records}, manifest


def no_engine(name: str) -> OCREngine:
    """OCR is faked at ``ocr_text``; the engine object is never used."""
    return cast(OCREngine, object())


def fake_ocr(references: dict[str, str]) -> Any:
    """Clean pages read perfectly; severity 0.5 loses digits 5-9; severity 1 reads nothing."""

    def ocr(engine: OCREngine, path: Path) -> str:
        text = references[rob.record_id_of(path)]
        if path.stem.endswith("-c0.50"):
            return text.translate(str.maketrans("56789", "     "))
        if path.stem.endswith("-c1.00"):
            return ""
        return text

    return ocr


def test_items_are_clean_pages_plus_their_variants(synthetic: tuple[Path, Any, Path]) -> None:
    _, _, manifest = synthetic

    items = load_items([manifest])
    limited = load_items([manifest], limit=1)

    assert [i.method for i in items].count("clean") == 3 and len(items) == 9
    assert {i.strength for i in items} == {0.0, 0.5, 1.0}
    assert all(i.doc_type == "invoice" and i.record_id.startswith("SINV-") for i in items)
    assert len(limited) == 3 and len({i.record_id for i in limited}) == 1


def test_kept_only_skips_variants_the_filter_dropped(tmp_path: Path) -> None:
    manifest = tmp_path / "validity.jsonl"
    rows = [
        {
            "source": "pages/invoice/A-p1.png",
            "image": "a.png",
            "method": "sd_img2img",
            "strength": 0.2,
            "kept": True,
        },
        {
            "source": "pages/invoice/A-p1.png",
            "image": "b.png",
            "method": "sd_img2img",
            "strength": 0.45,
            "kept": False,
        },
    ]
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    assert len(load_items([manifest])) == 3
    assert [i.image.name for i in load_items([manifest], kept_only=True)] == ["A-p1.png", "a.png"]


def test_reference_uses_ingestion_line_grouping(synthetic: tuple[Path, Any, Path]) -> None:
    root, records, _ = synthetic
    record = next(r for r in records.values() if r["record_type"] == "invoice")

    text = reference_text(root / "pdf" / "invoice" / f"{record['id']}.pdf")

    assert record["data"]["invoice_number"] in text
    assert len(text.splitlines()) > 5


def test_rule_lines_do_not_count() -> None:
    assert (
        without_rule_lines("BILL NO. 1\n------------\nTOTAL 5\n ===== \n") == "BILL NO. 1\nTOTAL 5"
    )


def test_scores_fall_with_degradation(
    synthetic: tuple[Path, Any, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, records, manifest = synthetic
    items = load_items([manifest])
    references = reference_texts(items, root)
    monkeypatch.setattr(rob, "ocr_text", fake_ocr(references))

    run = evaluate_engine("fake", items, records, references, make_engine=no_engine, workers=2)

    clean, half, full = (
        run.condition("clean", 0.0),
        run.condition("classical", 0.5),
        run.condition("classical", 1.0),
    )
    assert clean.pages == half.pages == full.pages == 3
    assert clean.cer == 0.0 and 0 < half.cer < full.cer == 1.0
    assert clean.readable_rate == 1.0 and full.readable_rate == 0.0
    assert clean.f1.micro.f1 > half.f1.micro.f1 > full.f1.micro.f1 == 0.0

    report = render_markdown([run], run_date=date(2026, 11, 13))
    assert "| fake | clean | 0.00 | 3 | 0.000 |" in report
    assert "| 100% |" in report  # clean F1 relative to itself
    assert rob.summary_json([run])[0]["method"] == "clean"


def test_unavailable_engine_is_skipped_and_failures_are_counted(
    synthetic: tuple[Path, Any, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, records, manifest = synthetic
    items = load_items([manifest], limit=1)

    def unavailable(name: str) -> OCREngine:
        raise OCRUnavailableError("not installed")

    skipped = evaluate_engine("paddle", items, records, {}, make_engine=unavailable)
    assert skipped.skipped_reason == "not installed"
    assert "skipped: not installed" in render_markdown([skipped], run_date=date(2026, 1, 1))

    def broken(engine: OCREngine, path: Path) -> str:
        raise RuntimeError("bad image")

    monkeypatch.setattr(rob, "ocr_text", broken)
    run = evaluate_engine("fake", items, records, {}, make_engine=no_engine)
    assert all(c.pages == 0 and len(c.failed) == 1 for c in run.conditions)


def test_cli_writes_report_and_json(
    synthetic: tuple[Path, Any, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _, manifest = synthetic
    items = load_items([manifest])
    monkeypatch.setattr(rob, "ocr_text", fake_ocr(reference_texts(items, root)))
    monkeypatch.setattr(rob, "get_engine", lambda **kw: no_engine(""))

    code = rob.main(
        [
            str(root),
            "--variants",
            str(manifest),
            "--engines",
            "tesseract",
            "--report",
            str(tmp_path / "r.md"),
            "--json",
            str(tmp_path / "r.json"),
        ]
    )

    assert code == 0
    assert "3 clean pages and 6 degraded variants" in (tmp_path / "r.md").read_text("utf-8")
    rows = json.loads((tmp_path / "r.json").read_text("utf-8"))
    assert [r["strength"] for r in rows] == [0.0, 0.5, 1.0]
    capsys.readouterr()
