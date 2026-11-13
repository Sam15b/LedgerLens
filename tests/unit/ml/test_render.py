import importlib
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import pymupdf
import pytest

from ml.synthetic.build import build
from ml.synthetic.render import render_documents as rd
from ml.synthetic.render.render_documents import (
    INVOICE_LAYOUTS,
    address_for,
    format_amount,
    format_date,
    format_qty,
    html_to_pdf,
    page_images,
    pdf_text,
    render_html,
    render_record,
    template_for,
    vendors_by_id,
)
from ml.synthetic.seed import load_vendor_master, po_invoice_records, seed_purchase_orders

Record = dict[str, Any]


@pytest.fixture(scope="module")
def records() -> list[Record]:
    vendors = load_vendor_master()
    rows = [r.features() for r in po_invoice_records(seed_purchase_orders(vendors), vendors)]
    return build(rows, invoices=400, seed=11).ledger.records()


@pytest.fixture(scope="module")
def vendors(records: list[Record]) -> dict[str, Any]:
    return vendors_by_id(records)


def of_type(records: list[Record], kind: str) -> list[Record]:
    return [r for r in records if r["record_type"] == kind]


def text_of(record: Record, vendors: dict[str, Any]) -> str:
    html = render_html(record, vendors, template=template_for(record))
    return pdf_text(html_to_pdf(html, title=record["id"], backend="pymupdf"))


def flat(text: str) -> str:
    return " ".join(text.split())


# --------------------------------------------------------------------------- filters


def test_filters_format_amounts_dates_and_quantities() -> None:
    assert format_amount("1234.5", "GBP") == "1,234.50"
    assert format_amount("1234500", "IDR") == "1,234,500"
    assert format_amount(None) == ""
    day = date(2025, 3, 3)
    assert format_date("2025-03-03") == "03 Mar 2025"
    assert format_date(day, "us") == "March 3, 2025"
    assert format_date(day, "iso") == "2025-03-03"
    assert format_qty("3") == "3" and format_qty("2.50") == "2.5" and format_qty("1200") == "1,200"


def test_addresses_are_stable_per_vendor_and_exist_for_unknown_vendors() -> None:
    vendor = {"vendor_id": "V001", "country": "GB"}
    assert address_for(vendor, "x") == address_for(vendor, "y")
    assert address_for(vendor, "x")[-1] == "United Kingdom"
    assert address_for(None, "Apex Global Solutions LLC")


# --------------------------------------------------------------------------- key fields


def test_every_invoice_layout_shows_the_key_fields(
    records: list[Record], vendors: dict[str, Any]
) -> None:
    clean = [r for r in of_type(records, "invoice") if not r["anomalies"]]
    by_layout: dict[str, Record] = {}
    for record in clean:
        by_layout.setdefault(template_for(record), record)
    assert set(by_layout) == set(INVOICE_LAYOUTS)

    styles = {"invoice_classic": "long", "invoice_modern": "us", "invoice_compact": "iso"}
    for layout, record in by_layout.items():
        data = record["data"]
        text = flat(text_of(record, vendors))
        currency = data["currency"]
        assert data["invoice_number"] in text, layout
        assert data["vendor_name"].upper() in text.upper(), layout
        assert data["po_number"] in text, layout
        assert format_date(data["invoice_date"], styles[layout]) in text, layout
        assert format_date(data["due_date"], styles[layout]) in text, layout
        assert format_amount(data["total"], currency) in text, layout
        assert format_amount(data["subtotal"], currency) in text, layout
        assert format_amount(data["tax"], currency) in text, layout
        assert currency in text, layout
        for line in data["line_items"]:
            assert line["description"] in text, layout


def test_invoices_without_lines_or_po_still_render(
    records: list[Record], vendors: dict[str, Any]
) -> None:
    rounded = next(r for r in of_type(records, "invoice") if "round_amount" in r["anomalies"])
    unknown = next(r for r in of_type(records, "invoice") if "unknown_vendor" in r["anomalies"])

    for record in (rounded, unknown):
        text = flat(text_of(record, vendors))
        assert record["data"]["invoice_number"] in text
        assert format_amount(record["data"]["total"], record["data"]["currency"]) in text
    assert unknown["data"]["vendor_name"].upper() in text_of(unknown, vendors).upper()


def test_po_payment_and_contract_show_their_key_fields(
    records: list[Record], vendors: dict[str, Any]
) -> None:
    po = of_type(records, "purchase_order")[0]
    text = flat(text_of(po, vendors))
    data = po["data"]
    for value in (data["po_number"], data["vendor_name"], data["contract_number"]):
        assert value in text
    assert "PURCHASE ORDER" in text and format_date(data["order_date"]) in text
    assert format_amount(data["total"], data["currency"]) in text

    payment = of_type(records, "payment")[0]
    text = flat(text_of(payment, vendors))
    data = payment["data"]
    assert data["payment_reference"] in text and data["vendor_name"] in text
    assert format_date(data["paid_at"]) in text
    assert data["allocations"][0]["invoice_number"] in text
    assert (
        f"Total Paid: {data['currency']} {format_amount(data['amount'], data['currency'])}" in text
    )

    contract = of_type(records, "contract")[0]
    text = flat(text_of(contract, vendors))
    data = contract["data"]
    assert data["contract_number"] in text
    assert f"between {data['vendor_name']} (the Supplier) and" in text
    assert f"within {data['payment_terms_days']} days" in text
    assert all(entry["item"] in text for entry in data["rate_card"])


# --------------------------------------------------------------------------- files


def test_rendering_is_deterministic_and_exact_duplicates_are_the_same_pdf(
    records: list[Record], vendors: dict[str, Any], tmp_path: Path
) -> None:
    invoices = {r["id"]: r for r in of_type(records, "invoice")}
    duplicate = next(r for r in invoices.values() if "exact_duplicate" in r["anomalies"])
    source = invoices[duplicate["duplicate_of"]]

    a = render_record(source, vendors, tmp_path / "a", dpi=None)
    b = render_record(source, vendors, tmp_path / "b", dpi=None)
    c = render_record(duplicate, vendors, tmp_path / "c", dpi=None)

    first = (tmp_path / "a" / a.pdf).read_bytes()
    assert first == (tmp_path / "b" / b.pdf).read_bytes()
    assert first == (tmp_path / "c" / c.pdf).read_bytes()
    assert c.record_id == duplicate["id"] and c.anomalies == ["exact_duplicate"]


def test_render_record_writes_pdf_and_grayscale_page_images(
    records: list[Record], vendors: dict[str, Any], tmp_path: Path
) -> None:
    contract = of_type(records, "contract")[0]

    rendered = render_record(contract, vendors, tmp_path, dpi=72)

    assert rendered.pdf == f"pdf/contract/{contract['id']}.pdf"
    with pymupdf.open(tmp_path / rendered.pdf) as doc:
        assert doc.page_count == len(rendered.pages) >= 1
        assert doc.metadata is not None and doc.metadata["title"] == contract["id"]
    image = pymupdf.Pixmap(str(tmp_path / rendered.pages[0]))
    assert image.n == 1  # grayscale
    assert image.width == round(pymupdf.paper_rect("a4").width)  # 72 dpi = 1 px per point
    assert len(page_images((tmp_path / rendered.pdf).read_bytes(), dpi=50)) == len(rendered.pages)


def test_backend_falls_back_to_pymupdf_without_weasyprint(monkeypatch: pytest.MonkeyPatch) -> None:
    rd._weasyprint.cache_clear()
    monkeypatch.setattr(importlib, "import_module", _missing)
    try:
        assert rd.available_backend() == "pymupdf"
        with pytest.raises(RuntimeError, match="WeasyPrint"):
            html_to_pdf("<p>x</p>", title="t", backend="weasyprint")
    finally:
        rd._weasyprint.cache_clear()


def _missing(name: str) -> Any:
    raise ModuleNotFoundError(name)


def test_cli_renders_a_limited_set_with_manifest(
    records: list[Record], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    code = rd.main(
        [
            str(path),
            "--out",
            str(tmp_path / "out"),
            "--limit",
            "2",
            "--no-images",
            "--backend",
            "pymupdf",
        ]
    )

    assert code == 0
    assert "8 documents" in capsys.readouterr().out
    lines = (tmp_path / "out" / "pdf" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    manifest = [json.loads(line) for line in lines]
    assert Counter(m["doc_type"] for m in manifest) == {
        "invoice": 2,
        "purchase_order": 2,
        "payment": 2,
        "contract": 2,
    }
    assert all((tmp_path / "out" / m["pdf"]).is_file() and m["pages"] == [] for m in manifest)


def test_a_limited_selection_spreads_over_the_whole_file(records: list[Record]) -> None:
    invoices = of_type(records, "invoice")

    chosen = rd.select(records, ["invoice"], 40)

    assert len(chosen) == 40 and len({r["id"] for r in chosen}) == 40
    assert chosen[0] is invoices[0]
    assert chosen[-1]["id"] in {r["id"] for r in invoices[-len(invoices) // 40 :]}
    assert any(r["anomalies"] == ["exact_duplicate"] for r in invoices[-20:])
    assert rd.select(records, ["contract"], None) == of_type(records, "contract")
