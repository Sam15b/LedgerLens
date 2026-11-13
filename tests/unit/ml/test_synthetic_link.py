from collections import Counter
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from ml.synthetic import build as build_cli
from ml.synthetic.anomalies import ANOMALY_RATES, plan_for_total
from ml.synthetic.build import BuildResult, build, read_records
from ml.synthetic.link import AS_OF, Ledger, LinkedInvoice, make_contracts, money
from ml.synthetic.seed import CATALOG, load_vendor_master, po_invoice_records, seed_purchase_orders

INVOICES = 1000


def seed_rows() -> list[dict[str, Any]]:
    vendors = load_vendor_master()
    return [r.features() for r in po_invoice_records(seed_purchase_orders(vendors), vendors)]


@pytest.fixture(scope="module")
def result() -> BuildResult:
    return build(seed_rows(), invoices=INVOICES, seed=7)


@pytest.fixture(scope="module")
def ledger(result: BuildResult) -> Ledger:
    return result.ledger


def clean(ledger: Ledger) -> list[LinkedInvoice]:
    return [i for i in ledger.invoices if not i.anomalies]


def labelled(ledger: Ledger, kind: str) -> list[LinkedInvoice]:
    return [i for i in ledger.invoices if kind in i.anomalies]


# --------------------------------------------------------------------------- contracts


def test_every_vendor_has_a_contract_pricing_its_catalog() -> None:
    vendors = load_vendor_master()
    contracts = make_contracts(vendors)

    assert set(contracts) == {v.vendor_id for v in vendors}
    for vendor in vendors:
        contract = contracts[vendor.vendor_id]
        assert contract.vendor_name == vendor.name
        assert contract.currency == vendor.currency
        assert contract.payment_terms_days == vendor.payment_terms_days
        assert [e.item for e in contract.rate_card] == [i[0] for i in CATALOG[vendor.category]]
        assert all(e.unit_price > 0 for e in contract.rate_card)
        assert contract.consistency_issues() == []
        assert contract.is_active(date(2025, 1, 6)) and contract.is_active(AS_OF)


# --------------------------------------------------------------------------- link integrity


def test_every_link_resolves(ledger: Ledger) -> None:
    vendor_ids = {v.vendor_id for v in ledger.vendors}
    pos = {p.id: p for p in ledger.purchase_orders}
    invoice_ids = [i.id for i in ledger.invoices]
    by_id = dict(zip(invoice_ids, ledger.invoices, strict=True))

    assert len(set(invoice_ids)) == len(invoice_ids)  # record ids are unique
    for linked in ledger.invoices:
        if "unknown_vendor" in linked.anomalies:
            assert linked.vendor_id is None and linked.po_number is None
            assert linked.invoice.vendor_name not in {v.name for v in ledger.vendors}
            continue
        assert linked.vendor_id in vendor_ids
        vendor = ledger.vendor(linked.vendor_id)
        assert linked.invoice.vendor_name == vendor.name
        assert linked.contract_number == ledger.contracts[vendor.vendor_id].contract_number
        po = pos[linked.po_number or ""]
        assert linked.id in po.invoice_ids
        assert po.vendor_id == linked.vendor_id
        assert po.po.order_date <= linked.invoice.invoice_date
        if linked.duplicate_of:
            assert by_id[linked.duplicate_of].po_number == linked.po_number
    for po in ledger.purchase_orders:
        assert po.invoice_ids and all(i in by_id for i in po.invoice_ids)
    for payment in ledger.payments:
        (invoice_id,) = payment.invoice_ids
        invoice = by_id[invoice_id].invoice
        assert payment.payment.vendor_name == invoice.vendor_name
        assert payment.payment.invoice_numbers == [invoice.invoice_number]
        assert payment.payment.currency == invoice.currency
        assert payment.payment.consistency_issues() == []


def test_clean_invoices_match_their_po_and_contract(ledger: Ledger) -> None:
    for linked in clean(ledger):
        invoice = linked.invoice
        po = ledger.purchase_order(linked.po_number or "").po
        contract = ledger.contracts[linked.vendor_id or ""]
        rates = {e.item: e.unit_price for e in contract.rate_card}

        assert invoice.consistency_issues() == []
        assert invoice.line_items == po.line_items
        assert invoice.total == po.total
        assert all(line.unit_price == rates[line.description] for line in invoice.line_items)
        assert invoice.invoice_date.weekday() < 5
        assert invoice.due_date is not None
        terms = contract.payment_terms_days or 0
        assert (invoice.due_date - invoice.invoice_date).days == terms


def test_invoice_numbers_increase_per_vendor_with_room_for_near_duplicates(
    ledger: Ledger,
) -> None:
    per_vendor: dict[str, list[tuple[date, str]]] = {}
    for linked in clean(ledger):
        per_vendor.setdefault(linked.vendor_id or "", []).append(
            (linked.invoice.invoice_date, linked.invoice.invoice_number)
        )
    for rows in per_vendor.values():
        numbers = [int(n.removeprefix("INV-")) for _, n in sorted(rows)]
        assert numbers == sorted(numbers)
        assert len(set(numbers)) == len(numbers)


# --------------------------------------------------------------------------- anomalies


def test_anomaly_rates_are_within_half_a_percent_of_target(result: BuildResult) -> None:
    ledger = result.ledger
    counts = Counter(a for i in ledger.invoices for a in i.anomalies)

    assert len(ledger.invoices) == INVOICES
    for kind, target in ANOMALY_RATES.items():
        assert counts[kind] / INVOICES == pytest.approx(target, abs=0.005)
        assert result.summary["anomalies"][kind]["count"] == counts[kind]
    assert all(len(i.anomalies) <= 1 for i in ledger.invoices)


def test_plan_sizes_the_clean_set_so_duplicates_reach_the_total() -> None:
    plan = plan_for_total(5000)

    assert plan.base == 4800 and plan.total == 5000
    assert plan.counts == {
        "exact_duplicate": 100,
        "near_duplicate": 100,
        "price_inflation": 100,
        "round_amount": 50,
        "unknown_vendor": 50,
        "rush_terms": 50,
    }
    with pytest.raises(ValueError, match="too few"):
        plan_for_total(10, {"exact_duplicate": 0.6, "price_inflation": 0.6})


def test_duplicates(ledger: Ledger) -> None:
    by_id = {i.id: i for i in ledger.invoices}
    for dup in labelled(ledger, "exact_duplicate"):
        source = by_id[dup.duplicate_of or ""]
        assert dup.id != source.id
        assert dup.invoice == source.invoice

    vendor_numbers = Counter((i.vendor_id, i.invoice.invoice_number) for i in ledger.invoices)
    for near in labelled(ledger, "near_duplicate"):
        source = by_id[near.duplicate_of or ""]
        a, b = near.invoice, source.invoice
        assert (a.vendor_name, a.total, a.line_items) == (b.vendor_name, b.total, b.line_items)
        assert 1 <= abs((a.invoice_date - b.invoice_date).days) <= 3
        number = a.invoice_number.removeprefix("INV-")
        assert number.endswith("A") or abs(int(number) - int(b.invoice_number[4:])) == 1
        assert vendor_numbers[(near.vendor_id, a.invoice_number)] == 1  # not a real number


def test_price_inflation_bills_10_to_40_percent_over_contract(ledger: Ledger) -> None:
    for linked in labelled(ledger, "price_inflation"):
        rates = {e.item: e.unit_price for e in ledger.contracts[linked.vendor_id or ""].rate_card}
        over = [
            line.unit_price / rates[line.description] - 1
            for line in linked.invoice.line_items
            if line.unit_price is not None and line.unit_price != rates[line.description]
        ]
        assert over and all(Decimal("0.09") <= o <= Decimal("0.41") for o in over)
        assert linked.invoice.consistency_issues() == []  # adds up; only the price is wrong
        po = ledger.purchase_order(linked.po_number or "").po
        assert linked.invoice.total > po.total


def test_round_amount_unknown_vendor_and_rush_terms(ledger: Ledger) -> None:
    for linked in labelled(ledger, "round_amount"):
        invoice = linked.invoice
        step = Decimal(1_000_000) if invoice.currency == "IDR" else Decimal(1000)
        assert invoice.total % step == 0 and invoice.total > 0
        assert invoice.line_items == [] and invoice.subtotal is None
        assert str(invoice.total) != linked.details["original_total"]

    remaining = {p.id for p in ledger.purchase_orders}
    for linked in labelled(ledger, "unknown_vendor"):
        assert linked.details["removed_po"] not in remaining

    for linked in labelled(ledger, "rush_terms"):
        invoice = linked.invoice
        assert invoice.invoice_date.weekday() >= 5
        assert invoice.due_date is not None
        assert 0 <= (invoice.due_date - invoice.invoice_date).days <= 3


# --------------------------------------------------------------------------- payments


def test_payment_states_and_amounts(ledger: Ledger) -> None:
    paid: dict[str, Decimal] = {}
    for payment in ledger.payments:
        paid[payment.invoice_ids[0]] = payment.payment.amount
        assert payment.payment.paid_at <= AS_OF
    states = Counter(i.payment_status for i in ledger.invoices)

    assert set(states) == {"paid", "partial", "unpaid", "overpaid"}
    assert states["paid"] > states["partial"] + states["unpaid"] + states["overpaid"]
    for linked in ledger.invoices:
        total = linked.invoice.total
        amount = paid.get(linked.id)
        match linked.payment_status:
            case "unpaid":
                assert amount is None
            case "paid":
                assert amount == total
            case "partial":
                assert amount is not None and amount < total
            case "overpaid":
                assert amount is not None and amount > total


def test_build_is_deterministic_per_seed() -> None:
    rows = seed_rows()

    first = build(rows, invoices=200, seed=3).ledger.records()

    assert build(rows, invoices=200, seed=3).ledger.records() == first
    assert build(rows, invoices=200, seed=4).ledger.records() != first


def test_money_rounds_to_the_currency_step() -> None:
    assert money(Decimal("10.005"), "USD") == Decimal("10.01")
    assert money(15_849, "IDR") == 15_800
    assert money(0, "GBP") == 0


# --------------------------------------------------------------------------- CLI


def test_cli_writes_records_and_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "records.jsonl"

    code = build_cli.main(
        ["--invoices", "300", "--seed-csv", str(tmp_path / "missing.csv"), "--out", str(out)]
    )

    assert code == 0
    assert "rush_terms" in capsys.readouterr().out
    types = Counter(r["record_type"] for r in read_records(out))
    assert types["invoice"] == 300 and types["vendor"] == 50 and types["contract"] == 50
    invoice = next(read_records(out, "invoice"))
    assert {"vendor_id", "po_number", "payment_status", "anomalies", "data"} <= set(invoice)
    summary = (tmp_path / "records.summary.json").read_text(encoding="utf-8")
    assert '"invoices": 300' in summary
