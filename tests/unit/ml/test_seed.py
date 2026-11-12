import random
from decimal import Decimal
from pathlib import Path

import pytest

from ml.synthetic.seed import (
    CATALOG,
    FIELDS,
    SeedRecord,
    cord_record,
    cord_records,
    is_plausible,
    load_vendor_master,
    po_invoice_records,
    read_csv,
    seed_purchase_orders,
    sroie_record,
    write_csv,
    write_purchase_orders,
)

# Shapes of CORD's ground-truth "gt_parse" (naver-clova-ix/cord-v2).
CORD_LIST = {
    "menu": [
        {"nm": "REAL GANACHE", "cnt": "1", "price": "16,500"},
        {"nm": "EGG TART", "cnt": "2", "price": "26,000"},
    ],
    "total": {"total_price": "42,500", "cashprice": "50,000", "changeprice": "7,500"},
}
CORD_SINGLE = {
    "menu": {"nm": "S-Ovaltine 50%", "unitprice": "20,000", "cnt": "1 x", "price": "20,000"},
    "sub_total": {"subtotal_price": "18,181", "tax_price": "1,818"},
    "total": {"total_price": "20,000"},
}


def test_cord_receipt_with_item_list() -> None:
    record = cord_record(CORD_LIST, "train/0", random.Random(0))

    assert record is not None
    assert (record.line_count, record.qty_mean) == (2, 1.5)
    assert record.unit_price_mean == pytest.approx((16_500 + 13_000) / 2)
    assert (record.subtotal, record.total, record.tax_rate) == (42_500, 42_500, 0.0)
    assert (record.currency, record.vendor_category, record.days_to_due) == (
        "IDR",
        "food_beverage",
        0,
    )
    assert 1 <= record.day_of_month <= 28
    assert record.imputed == ["days_to_due", "day_of_month", "vendor_category", "subtotal"]


def test_cord_receipt_with_single_item_and_tax() -> None:
    record = cord_record(CORD_SINGLE, "train/1", random.Random(0))

    assert record is not None
    assert (record.line_count, record.unit_price_mean, record.subtotal) == (1, 20_000, 18_181)
    assert record.tax_rate == pytest.approx(0.1, abs=1e-3)
    assert "subtotal" not in record.imputed


@pytest.mark.parametrize(
    "gt",
    [
        {"menu": [{"nm": "x", "price": "1,000"}]},  # no total
        {"menu": [], "total": {"total_price": "1,000"}},  # no items
        {"menu": [{"nm": "x", "cnt": "1"}], "total": {"total_price": "1,000"}},  # no prices
    ],
)
def test_unusable_cord_receipts_are_skipped(gt: dict[str, object]) -> None:
    assert cord_record(gt, "x", random.Random(0)) is None


def test_inconsistent_annotations_are_dropped() -> None:
    misread = {**CORD_SINGLE, "sub_total": {"subtotal_price": "18,181", "tax_price": "181,818"}}

    records = cord_records([("a", CORD_LIST), ("b", misread)])

    assert [r.source_id for r in records] == ["a"]
    assert not is_plausible(
        SeedRecord(1, 1, 1, 100, 0.0, 400, 0, 1, "retail", "USD", "x", "x")  # total 4x subtotal
    )


def test_imputed_day_of_month_is_reproducible() -> None:
    first = cord_records([(str(i), CORD_LIST) for i in range(5)], seed=1)
    second = cord_records([(str(i), CORD_LIST) for i in range(5)], seed=1)

    assert [r.day_of_month for r in first] == [r.day_of_month for r in second]


def test_sroie_receipt() -> None:
    record = sroie_record(
        "X1",
        {"company": "SYARIKAT PERNIAGAAN GIN KEE HARDWARE", "date": "04/01/2018", "total": "7.95"},
    )

    assert record is not None
    assert (record.total, record.day_of_month, record.currency) == (7.95, 4, "MYR")
    assert record.vendor_category == "hardware"
    assert sroie_record("X2", {"company": "A", "total": "7.95"}) is None  # no date


def test_vendor_master_has_fifty_vendors_with_tax_ids() -> None:
    vendors = load_vendor_master()

    assert len(vendors) == 50 and len({v.vendor_id for v in vendors}) == 50
    assert len({v.tax_id for v in vendors}) == 50
    assert {v.category for v in vendors} == set(CATALOG)
    assert {v.currency for v in vendors} == {"GBP", "EUR", "USD", "MYR", "SGD", "IDR"}
    assert vendors[0].name == "Acme Supplies Ltd"  # the fixtures' supplier


def test_seed_purchase_orders_are_valid_and_reproducible() -> None:
    vendors = load_vendor_master()

    orders = seed_purchase_orders(vendors, per_vendor=2)

    assert len(orders) == 100
    assert orders == seed_purchase_orders(vendors, per_vendor=2)
    assert len({po.po_number for po in orders}) == 100
    for po in orders:
        assert po.consistency_issues() == []  # lines add up, subtotal + tax = total
        assert po.order_date.weekday() < 5
        assert po.buyer_name == "Northwind Traders"
    idr = next(po for po in orders if po.currency == "IDR")
    assert all(
        line.unit_price is not None and line.unit_price % 100 == 0 for line in idr.line_items
    )


def test_each_po_becomes_a_seed_invoice_on_vendor_terms() -> None:
    vendors = load_vendor_master()
    orders = seed_purchase_orders(vendors, per_vendor=1)

    records = po_invoice_records(orders, vendors)

    first = records[0]
    assert len(records) == 50 and first.source == "seed_po"
    assert first.days_to_due == vendors[0].payment_terms_days
    assert first.total == float(orders[0].total) and first.tax_rate == 0.2
    assert first.line_count == len(orders[0].line_items)


def test_csv_and_json_round_trip(tmp_path: Path) -> None:
    vendors = load_vendor_master()
    orders = seed_purchase_orders(vendors[:3], per_vendor=1)
    records = po_invoice_records(orders, vendors) + cord_records([("a", CORD_LIST)])

    write_csv(records, tmp_path / "seed_invoices.csv")
    write_purchase_orders(orders, tmp_path / "seed_pos.json")

    assert read_csv(tmp_path / "seed_invoices.csv") == records
    assert list(records[0].features()) == list(FIELDS)
    assert '"total": "' in (tmp_path / "seed_pos.json").read_text(encoding="utf-8")
    assert Decimal(str(orders[0].total)) > 0
