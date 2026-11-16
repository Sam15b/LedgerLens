"""Reading a ledger and deciding vendors before anything is written (no stores needed)."""

import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from ledgerlens.core.schemas import Invoice, Payment
from ledgerlens.storage.bulk_load import (
    LedgerLine,
    StoreCounts,
    parse_line,
    plan_vendors,
    read_ledger,
    read_manifest,
)
from ledgerlens.storage.neo4j.linker import new_vendor_id

VENDOR = {
    "record_type": "vendor",
    "id": "V001",
    "data": {"vendor_id": "V001", "name": "Acme Supplies Ltd", "tax_id": "GB123456789"},
}


def invoice_line(
    id: str, vendor: str, vendor_id: str | None, tax_id: str | None = None
) -> dict[str, Any]:
    return {
        "record_type": "invoice",
        "id": id,
        "vendor_id": vendor_id,
        "payment_status": "unpaid",
        "anomalies": [],
        "data": {
            "doc_type": "invoice",
            "invoice_number": f"INV-{id}",
            "vendor_name": vendor,
            "vendor_tax_id": tax_id,
            "invoice_date": "2026-03-01",
            "total": "10.00",
        },
    }


def lines(*objs: dict[str, Any]) -> list[LedgerLine]:
    return [parse_line(obj) for obj in objs]


def test_lines_parse_into_vendors_and_records(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        "\n".join(json.dumps(o) for o in (VENDOR, invoice_line("S1", "Acme", "V001"))) + "\n\n",
        encoding="utf-8",
    )

    vendor, invoice = read_ledger(path)

    assert vendor.vendor is not None and vendor.vendor.tax_id == "GB123456789"
    assert vendor.record is None
    assert isinstance(invoice.record, Invoice)
    assert (invoice.id, invoice.vendor_id, invoice.record.total) == ("S1", "V001", Decimal("10.00"))


def test_unknown_record_types_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown record_type"):
        parse_line({"record_type": "receipt", "id": "x", "data": {}})


def test_known_vendor_ids_are_kept_and_other_spellings_become_aliases() -> None:
    vendors, assignments = plan_vendors(
        lines(
            VENDOR,
            invoice_line("S1", "Acme Supplies Ltd", "V001"),
            invoice_line("S2", "ACME Supplies", "V001"),
            invoice_line("S3", "ACME Supplies", "V001"),
        )
    )

    assert [a.vendor_id for a in assignments] == ["V001"] * 3
    assert vendors["V001"].aliases == ["ACME Supplies"]
    assert not vendors["V001"].new


def test_records_without_a_vendor_are_resolved_like_ingestion_does() -> None:
    vendors, assignments = plan_vendors(
        lines(
            VENDOR,
            invoice_line("S1", "Some Other Name", None, tax_id="gb 123-456-789"),  # tax id
            invoice_line("S2", "Acme Supplies Limited", None),  # canonical name
            invoice_line("S3", "Apex Global Solutions LLC", None),  # nobody: a new vendor
            invoice_line("S4", "Apex Global Solutions", None),  # ...found again by name
            invoice_line("S5", "Ghost Co", "V999"),  # an id not in the master: resolved too
        )
    )

    apex = new_vendor_id("Apex Global Solutions LLC")
    assert [a.vendor_id for a in assignments] == [
        "V001",
        "V001",
        apex,
        apex,
        new_vendor_id("Ghost Co"),
    ]
    assert vendors[apex].new and vendors[apex].name == "Apex Global Solutions LLC"
    assert vendors[apex].aliases == ["Apex Global Solutions"]
    assert vendors["V001"].aliases == ["Some Other Name", "Acme Supplies Limited"]


def test_manifest_paths_resolve_against_the_data_root(tmp_path: Path) -> None:
    manifest = tmp_path / "pdf" / "manifest.jsonl"
    manifest.parent.mkdir()
    entry = {"record_id": "S1", "doc_type": "invoice", "pdf": "pdf/invoice/S1.pdf"}
    manifest.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    (rendered,) = read_manifest(manifest)

    assert rendered.path == tmp_path / "pdf" / "invoice" / "S1.pdf"


def test_store_counts_report_every_disagreement() -> None:
    counts = StoreCounts(
        postgres={"vendors": 2, "invoices": 3, "chunks": 5},
        pinecone=4,
        neo4j={"Vendor": 2, "Invoice": 2},
    )

    assert counts.mismatches() == [
        "invoices: postgres 3 != neo4j 2",
        "chunks: postgres 5 != pinecone 4",
    ]
    assert StoreCounts({"chunks": 1}, 1, {}).mismatches() == []


def test_payments_parse_with_their_allocations() -> None:
    (line,) = lines(
        {
            "record_type": "payment",
            "id": "P1",
            "vendor_id": "V001",
            "data": {
                "doc_type": "payment",
                "payment_reference": "P1",
                "vendor_name": "Acme",
                "paid_at": "2026-03-05",
                "amount": "10.00",
                "allocations": [{"invoice_number": "INV-S1", "amount": "10.00"}],
            },
        }
    )
    assert isinstance(line.record, Payment)
    assert line.record.paid_at == date(2026, 3, 5)
    assert line.record.allocations[0].invoice_number == "INV-S1"
