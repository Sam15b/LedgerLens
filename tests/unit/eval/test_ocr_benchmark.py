import shutil
from datetime import date
from pathlib import Path

import pytest
from PIL import Image

from eval import ocr_benchmark
from eval.datasets.sroie import SroieLine, SroieSample
from eval.ocr_benchmark import EngineResult, evaluate, main, render_markdown
from ledgerlens.ingestion.layout import BBox
from ledgerlens.ingestion.ocr import OCREngine, OCRUnavailableError
from ledgerlens.ingestion.ocr.base import OCRWord

RECEIPT = Path(__file__).parents[2] / "fixtures" / "documents" / "receipt_scanned.png"


class ScriptedEngine:
    """Returns the same words for every image."""

    name = "scripted"

    def __init__(self, *texts: str) -> None:
        self.words = [
            OCRWord(text=t, bbox=BBox(10, 20 * i, 60, 20 * i + 12), conf=90.0)
            for i, t in enumerate(texts)
        ]

    def run(self, image: Image.Image) -> list[OCRWord]:
        return self.words


def sample(sample_id: str, *lines: str, image: Path = RECEIPT) -> SroieSample:
    return SroieSample(sample_id, image, [SroieLine(t, ((0, 0),) * 4) for t in lines])


def test_evaluate_scores_engine_text_against_ground_truth() -> None:
    samples = [sample("a", "TOTAL", "9.50"), sample("b", "TOTAL", "9.SO")]

    result = evaluate("scripted", samples, make_engine=lambda name: ScriptedEngine("TOTAL", "9.50"))

    # "total\n9.50" vs "total 9.50": whitespace is normalized, so only b's "SO" counts.
    assert (result.errors, result.reference_chars) == (2, 20)
    assert result.cer == 0.1
    assert result.per_doc_cer == [0.0, 0.2] and len(result.seconds) == 2


def test_unavailable_engines_and_unreadable_files_do_not_stop_the_run(tmp_path: Path) -> None:
    def missing(name: str) -> OCREngine:
        raise OCRUnavailableError("paddleocr is not installed")

    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"\xff\xd8\xff not really a jpeg")

    skipped = evaluate("paddle", [sample("a", "x")], make_engine=missing)
    partial = evaluate(
        "scripted",
        [sample("bad", "x", image=broken), sample("ok", "TOTAL")],
        make_engine=lambda name: ScriptedEngine("TOTAL"),
    )

    assert skipped.skipped_reason == "paddleocr is not installed"
    assert partial.failed[0].startswith("bad: ") and partial.per_doc_cer == [0.0]


def test_report_table() -> None:
    done = EngineResult(
        "tesseract", errors=5, reference_chars=100, per_doc_cer=[0.0, 0.1], seconds=[1.0, 3.0]
    )
    skipped = EngineResult("paddle", skipped_reason="not installed")

    report = render_markdown(
        [done, skipped], dataset="SROIE test", n_samples=2, run_date=date(2026, 11, 12)
    )

    assert report.startswith("# OCR benchmark — 2026-11-12")
    assert "| tesseract | 0.050 | 0.050 | 0.100 | 2.00 | 0 |" in report
    assert "| paddle | — | — | — | — | skipped: not installed |" in report


def test_cli_writes_the_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "sroie" / "test"
    (root / "img").mkdir(parents=True)
    (root / "box").mkdir()
    shutil.copy(RECEIPT, root / "img" / "R1.png")
    (root / "box" / "R1.txt").write_text("0,0,1,0,1,1,0,1,KEDAI KOPI\n", encoding="utf-8")
    monkeypatch.setattr(ocr_benchmark, "get_engine", lambda name: ScriptedEngine("KEDAI", "KOPI"))
    report = tmp_path / "reports" / "ocr.md"

    assert main([str(tmp_path / "sroie"), "--engines", "tesseract", "--report", str(report)]) == 0
    assert "| tesseract | 0.000 | 0.000 | 0.000 |" in report.read_text(encoding="utf-8")
