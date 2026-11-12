from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from eval.extraction_f1 import (
    F1Report,
    FieldScore,
    evaluate_sroie,
    main,
    normalize_value,
    render_markdown,
)
from ledgerlens.core.schemas import DocType
from ledgerlens.extraction.base import ExtractionError, ExtractionResult, Extractor
from ledgerlens.extraction.rules import RulesExtractor

BOX = """\
72,25,326,25,326,64,72,64,BOOK TA .K(TAMAN DAYA) SDN BHD
50,82,440,82,440,121,50,121,NO.53 55,57 & 59, JALAN SAGU 18, TAMAN DAYA
50,130,440,130,440,150,50,150,DATE: 25/12/2018 8:13:39 PM
110,312,214,312,214,334,110,334,TOTAL: 9.00
"""
ENTITIES = (
    '{"company": "BOOK TA .K(TAMAN DAYA) SDN BHD", "date": "25/12/2018",'
    ' "address": "NO.53 55,57 & 59, JALAN SAGU 18, TAMAN DAYA", "total": "9.00"}'
)


@pytest.mark.parametrize(
    "name,value,expected",
    [
        ("total", "9.00", "9.00"),
        ("total", "RM 1,234.5", "1234.50"),
        ("total", Decimal("9"), "9.00"),
        ("total", 9.5, "9.50"),
        ("receipt_date", "25/12/2018", "2018-12-25"),
        ("receipt_date", date(2018, 12, 25), "2018-12-25"),
        ("vendor_name", "BOOK TA .K(TAMAN DAYA) SDN BHD", "book ta k taman daya"),
        ("vendor_address", "NO.53, Jalan  Sagu 18.", "no 53 jalan sagu 18"),
        ("total", "", None),
        ("total", None, None),
        ("receipt_date", "not a date", None),
    ],
)
def test_normalize_value(name: str, value: object, expected: str | None) -> None:
    assert normalize_value(name, value) == expected


def test_counts_precision_recall_and_f1() -> None:
    report = F1Report()
    gold = {"total": "9.00", "receipt_date": "25/12/2018", "vendor_name": "Book Ta K Sdn Bhd"}

    report.add(
        gold, {"total": "9.0", "receipt_date": "2018-12-25", "vendor_name": "BOOK TA K"}, valid=True
    )
    report.add(gold, {"total": "7.50", "vendor_name": None}, valid=False)  # wrong, missed, missed
    report.add({"total": None}, {"total": "1.00"}, valid=True, fields=["total"])  # spurious

    total = report.fields["total"]
    assert (total.tp, total.fp, total.fn) == (1, 2, 1)
    assert (total.precision, total.recall, total.support) == (1 / 3, 1 / 2, 2)
    assert report.fields["vendor_name"].f1 == pytest.approx(2 / 3)
    assert report.documents == 3
    assert report.json_validity == pytest.approx(2 / 3)
    micro = report.micro
    assert (micro.tp, micro.fp, micro.fn) == (3, 2, 3)


def test_empty_scores_are_zero_not_errors() -> None:
    assert FieldScore().f1 == 0.0 and F1Report().json_validity == 0.0


def test_report_markdown() -> None:
    report = F1Report()
    report.add({"total": "9.00"}, {"total": "9.00"}, valid=True)

    text = render_markdown({"rules": report, "llm": F1Report()}, title="Extraction F1")

    assert "| rules | 1 | 100.0% | 1.000 | 1.000 | 1.000 |" in text
    assert "| total | 1.000 | — |" in text


def sroie(tmp_path: Path) -> Path:
    base = tmp_path / "sroie" / "test"
    for sub in ("img", "box", "entities"):
        (base / sub).mkdir(parents=True)
    (base / "img" / "X1.jpg").write_bytes(b"")
    (base / "box" / "X1.txt").write_text(BOX, encoding="utf-8")
    (base / "entities" / "X1.txt").write_text(ENTITIES, encoding="utf-8")
    return tmp_path / "sroie"


def test_rules_on_a_sroie_receipt(tmp_path: Path) -> None:
    report, stats = evaluate_sroie(RulesExtractor(), sroie(tmp_path))

    assert report.documents == 1 and report.json_validity == 1.0
    assert {name: s.f1 for name, s in report.fields.items()} == {
        "vendor_name": 1.0,
        "receipt_date": 1.0,
        "vendor_address": 1.0,
        "total": 1.0,
    }
    assert len(stats.seconds) == 1 and stats.failures == []


class Broken:
    name = "broken"

    def extract(self, text: str, doc_type: DocType | str) -> ExtractionResult:
        raise ExtractionError("no models reachable")


def test_failed_calls_count_as_misses(tmp_path: Path) -> None:
    report, stats = evaluate_sroie(Broken(), sroie(tmp_path))

    assert report.micro.f1 == 0.0 and report.json_validity == 0.0
    assert stats.failures == ["X1: no models reachable"]


def test_cli_writes_a_report(tmp_path: Path) -> None:
    out = tmp_path / "reports" / "extraction.md"

    def make(name: str) -> Extractor:
        return RulesExtractor() if name == "rules" else Broken()

    code = main(
        [str(sroie(tmp_path)), "--extractors", "rules", "llm", "--report", str(out)],
        make_extractor=make,
    )

    text = out.read_text(encoding="utf-8")
    assert code == 0
    assert "| rules | 1 | 100.0% | 1.000 | 1.000 | 1.000 |" in text
    assert "`llm`: median" in text and "1 failed calls" in text
