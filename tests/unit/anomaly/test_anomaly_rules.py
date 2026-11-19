"""Anomaly rules: one test per rule, then every injected anomaly type on a synthetic ledger.

(Named ``test_anomaly_rules`` because ``tests/unit/extraction/test_rules.py`` exists and the
test folders are not packages.)
"""

from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from ledgerlens.anomaly.rules import (
    InvoiceRef,
    RuleContext,
    RuleHit,
    VendorMaster,
    check_invoice,
    contract_rate_breaches,
    exact_duplicates,
    near_duplicates,
    number_key,
    number_similarity,
    over_po,
    rate_for,
    unknown_vendor,
)
from ledgerlens.core.schemas import Contract, Invoice, PurchaseOrder
from ml.synthetic.build import build
from ml.synthetic.seed import load_vendor_master, po_invoice_records, seed_purchase_orders

D = Decimal


def invoice(**overrides: Any) -> Invoice:
    data: dict[str, Any] = {
        "invoice_number": "INV-005275",
        "vendor_name": "Acme Supplies Ltd",
        "vendor_tax_id": "GB123456789",
        "invoice_date": "2026-03-04",
        "po_number": "PO-1",
        "currency": "GBP",
        "total": "1200.00",
        "line_items": [
            {
                "description": "Pallet wrap 500m",
                "quantity": "10",
                "unit_price": "42.41",
                "amount": "424.10",
            },
            {"description": "Packing tape 48mm", "quantity": "2", "amount": "118.30"},
        ],
    }
    return Invoice.model_validate(data | overrides)


def ref(id: str = "I2", vendor_id: str | None = "V001", **overrides: Any) -> InvoiceRef:
    sha = overrides.pop("sha256", None)
    return InvoiceRef(id, invoice(**overrides), vendor_id, sha)


STORED = ref("I1")


# --------------------------------------------------------------------------- exact duplicate


def test_exact_duplicate_same_vendor_and_number() -> None:
    (hit,) = exact_duplicates(ref(invoice_number="inv 005275"), [STORED])

    assert (hit.rule, hit.invoice_id, hit.related_ids) == ("exact_duplicate", "I2", ["I1"])
    assert "inv 005275" in hit.reason and "I1" in hit.reason
    assert exact_duplicates(STORED, [STORED]) == []  # not a duplicate of itself
    assert exact_duplicates(ref(vendor_id="V002"), [STORED]) == []  # another vendor's number
    assert exact_duplicates(ref(invoice_number="INV-005276"), [STORED]) == []


def test_exact_duplicate_same_file_and_unresolved_vendor() -> None:
    first = ref("I1", sha256="abc", invoice_number="A-1")
    (hit,) = exact_duplicates(ref(sha256="abc", invoice_number="B-9"), [first])
    assert hit.details["same_file"] == ["I1"] and "same file" in hit.reason

    # vendors with no master id compare on the canonical printed name
    printed = ref("I1", vendor_id=None, vendor_name="ACME SUPPLIES LIMITED")
    assert exact_duplicates(ref(vendor_id=None), [printed])
    assert number_key(" inv-005275/a ") == "INV005275A"


# --------------------------------------------------------------------------- near duplicate


@pytest.mark.parametrize(
    ("a", "b", "alike"),
    [
        ("INV-005275", "INV-005276", True),  # the next number
        ("INV-010359", "INV-010360", True),  # ... across a carry
        ("INV-005275", "INV-005275A", True),  # a suffix
        ("INV-005275", "INV-005725", True),  # digits transposed
        ("INV-005275", "INV-005285", False),  # another number in the sequence
        ("INV-005380", "INV-005348", False),
        ("A-77", "B-77", False),  # short numbers: one edit is a lot
    ],
)
def test_number_similarity(a: str, b: str, alike: bool) -> None:
    assert (number_similarity(a, b) > 0.85) is alike


def test_near_duplicate_hit_and_its_limits() -> None:
    resent = ref(invoice_number="INV-005276", invoice_date="2026-03-07", total="1205.00")
    (hit,) = near_duplicates(resent, [STORED])

    assert hit.rule == "near_duplicate" and hit.related_ids == ["I1"]
    assert hit.details["days_apart"] == 3 and hit.details["number_similarity"] == 1.0
    assert hit.details["amount_gap"] == pytest.approx(5 / 1205)
    assert "INV-005275" in hit.reason

    assert near_duplicates(ref(invoice_number="INV-005276", total="1213.00"), [STORED]) == []
    late = ref(invoice_number="INV-005276", invoice_date="2026-03-12")
    assert near_duplicates(late, [STORED]) == []  # 8 days apart
    assert near_duplicates(ref(), [STORED]) == []  # same number: exact, not near
    assert near_duplicates(ref(invoice_number="INV-005276", vendor_id="V002"), [STORED]) == []


# --------------------------------------------------------------------------- over PO


def purchase_order(total: str = "1200.00", currency: str = "GBP") -> PurchaseOrder:
    return PurchaseOrder(
        po_number="PO-1",
        vendor_name="Acme Supplies Ltd",
        order_date=date(2026, 2, 1),
        currency=currency,
        total=D(total),
    )


def test_over_po() -> None:
    po = purchase_order()
    assert over_po(ref(), po) == []
    assert over_po(ref(total="1205.00"), po) == []  # within 0.5 %

    (hit,) = over_po(ref(total="1300.00"), po)
    assert hit.rule == "over_po" and hit.related_ids == ["PO-1"]
    assert hit.details["over_by"] == "100.00" and "PO-1" in hit.reason

    (billed,) = over_po(ref(total="300.00"), po, already_billed=D("1000.00"))
    assert billed.details["remaining"] == "200.00" and "already billed" in billed.reason
    assert over_po(ref(total="9999"), None) == []
    assert over_po(ref(total="9999"), purchase_order(currency="EUR")) == []


# --------------------------------------------------------------------------- contract rates


CONTRACT = Contract.model_validate(
    {
        "contract_number": "CT-V001",
        "vendor_name": "Acme Supplies Ltd",
        "start_date": "2024-08-23",
        "currency": "GBP",
        "rate_card": [
            {"item": "Pallet wrap 500m", "unit": "roll", "unit_price": "42.41"},
            {"item": "Packing tape 48mm", "unit": "case", "unit_price": "59.15"},
        ],
    }
)


def test_rate_lookup() -> None:
    assert rate_for("PALLET WRAP 500M", CONTRACT) == ("Pallet wrap 500m", D("42.41"))
    assert rate_for("500m pallet wrap", CONTRACT) == ("Pallet wrap 500m", D("42.41"))
    assert rate_for("Bubble wrap", CONTRACT) is None


def test_contract_rate_breach() -> None:
    assert contract_rate_breaches(ref(), CONTRACT) == []  # at the rate
    within = [
        {
            "description": "Pallet wrap 500m",
            "quantity": "1",
            "unit_price": "43.25",
            "amount": "43.25",
        }
    ]
    assert contract_rate_breaches(ref(line_items=within), CONTRACT) == []  # +2.0 %

    inflated = [
        {
            "description": "Pallet wrap 500m",
            "quantity": "10",
            "unit_price": "50.89",
            "amount": "508.90",
        },
        {
            "description": "Packing tape 48mm",
            "quantity": "1",
            "unit_price": "70.98",
            "amount": "70.98",
        },
        {"description": "Delivery", "quantity": "1", "unit_price": "999.00", "amount": "999.00"},
    ]
    (hit,) = contract_rate_breaches(ref(line_items=inflated), CONTRACT)
    assert hit.rule == "contract_rate_breach" and hit.related_ids == ["CT-V001"]
    assert [line["item"] for line in hit.details["lines"]] == [
        "Pallet wrap 500m",
        "Packing tape 48mm",
    ]
    assert "Pallet wrap 500m at 50.89 vs 42.41 (+20%)" in hit.reason  # the worst line
    assert contract_rate_breaches(ref(line_items=inflated, currency="EUR"), CONTRACT) == []
    assert contract_rate_breaches(ref(line_items=inflated), None) == []


# --------------------------------------------------------------------------- unknown vendor


MASTER = VendorMaster.from_vendors(
    [
        {"vendor_id": "V001", "name": "Acme Supplies Ltd", "tax_id": "GB123456789"},
        {"vendor_id": "V002", "name": "Brightline Packaging Ltd", "aliases": ["Brightline"]},
    ]
)


def test_unknown_vendor() -> None:
    assert unknown_vendor(ref(), MASTER) == []  # master id
    assert (
        unknown_vendor(
            ref(vendor_id=None, vendor_name="Someone", vendor_tax_id="gb 123 456 789"), MASTER
        )
        == []
    )
    assert (
        unknown_vendor(
            ref(vendor_id=None, vendor_name="ACME SUPPLES LTD", vendor_tax_id=None), MASTER
        )
        == []
    )
    assert (
        unknown_vendor(ref(vendor_id=None, vendor_name="Brightline", vendor_tax_id=None), MASTER)
        == []
    )

    stranger = ref(vendor_id=None, vendor_name="Apex Global Solutions LLC", vendor_tax_id=None)
    (hit,) = unknown_vendor(stranger, MASTER)
    assert (
        hit.rule == "unknown_vendor" and "Apex Global" in hit.reason and "no tax id" in hit.reason
    )
    assert unknown_vendor(stranger, None) == []


# --------------------------------------------------------------------------- all rules


def test_check_invoice_runs_every_rule() -> None:
    clean = check_invoice(ref(), RuleContext(stored=[ref("I0", invoice_number="INV-1")]))
    assert clean == []

    context = RuleContext(
        stored=[STORED, ref("X1", vendor_id="V002")],
        purchase_order=purchase_order(),
        already_billed=D("1200.00"),
        contract=CONTRACT,
        master=MASTER,
    )
    hits = check_invoice(ref(), context)
    assert [h.rule for h in hits] == ["exact_duplicate", "over_po"]
    assert all(isinstance(h, RuleHit) and h.reason for h in hits)
    assert hits[0].related_ids == ["I1"]  # not V002's invoice with the same number


# --------------------------------------------------------------------------- synthetic ledger


RULE_FOR = {
    "exact_duplicate": "exact_duplicate",
    "near_duplicate": "near_duplicate",
    "price_inflation": "contract_rate_breach",
    "unknown_vendor": "unknown_vendor",
}


def test_each_injected_anomaly_is_caught_by_its_rule() -> None:
    vendors = load_vendor_master()
    rows = [r.features() for r in po_invoice_records(seed_purchase_orders(vendors), vendors)]
    ledger = build(rows, invoices=600, seed=11).ledger
    master = VendorMaster.from_vendors(vars(v) for v in ledger.vendors)
    pos = {p.id: p.po for p in ledger.purchase_orders}

    stored: list[InvoiceRef] = []
    billed: dict[str, Decimal] = defaultdict(Decimal)
    caught: Counter[str] = Counter()
    labelled: Counter[str] = Counter()
    false_alarms: Counter[str] = Counter()
    for item in ledger.invoices:  # the order they were added: copies after their source
        candidate = InvoiceRef(item.id, item.invoice, item.vendor_id)
        po_number = item.invoice.po_number or ""
        context = RuleContext(
            stored=stored,
            purchase_order=pos.get(po_number),
            already_billed=billed[po_number],
            contract=ledger.contracts.get(item.vendor_id or ""),
            master=master,
        )
        rules = {hit.rule for hit in check_invoice(candidate, context)}
        for label in item.anomalies:
            labelled[label] += 1
            caught[label] += RULE_FOR.get(label) in rules
        if not item.anomalies:
            false_alarms.update(rules)
        stored.append(candidate)
        billed[po_number] += item.invoice.total

    for label in RULE_FOR:
        assert labelled[label] > 0 and caught[label] == labelled[label], label
    clean = len(ledger.invoices) - sum(labelled.values())
    assert sum(false_alarms.values()) <= 0.01 * clean, false_alarms
