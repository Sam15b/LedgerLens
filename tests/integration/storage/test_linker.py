"""GraphLinker against the compose Neo4j: MERGE idempotency, edges, vendor resolution.

Every id and vendor name carries a per-test tag, and everything tagged is deleted after
the test, so other data in the development graph is left alone.
"""

import uuid
from collections.abc import Iterator
from datetime import date
from decimal import Decimal
from typing import Any

import pytest
from neo4j import Driver

from ledgerlens.core.schemas import (
    Contract,
    Invoice,
    LineItem,
    Payment,
    PaymentAllocation,
    PurchaseOrder,
    RateCardEntry,
    Receipt,
)
from ledgerlens.storage.neo4j.driver import apply_schema, get_driver, run_read, run_write
from ledgerlens.storage.neo4j.linker import GraphLinker, VendorResolver

pytestmark = pytest.mark.integration

D = Decimal


@pytest.fixture(scope="module")
def driver() -> Driver:
    driver = get_driver()
    apply_schema(driver)
    return driver


@pytest.fixture
def tag(driver: Driver) -> Iterator[str]:
    tag = f"t{uuid.uuid4().hex[:10]}"
    yield tag
    run_write(driver, "MATCH (n) WHERE n.id CONTAINS $tag DETACH DELETE n", {"tag": tag})


@pytest.fixture
def linker(driver: Driver) -> GraphLinker:
    return GraphLinker(driver)


def vendor_name(tag: str) -> str:
    return f"Acme {tag} Supplies Ltd"


def invoice(tag: str, number: str = "INV-1", po: str | None = "PO-7", lines: int = 2) -> Invoice:
    return Invoice(
        invoice_number=f"{number}-{tag}",
        vendor_name=vendor_name(tag),
        vendor_tax_id=f"GB{tag}",
        invoice_date=date(2026, 9, 1),
        due_date=date(2026, 10, 1),
        po_number=f"{po}-{tag}" if po else None,
        total=D(50) * lines,
        line_items=[
            LineItem(description=f"Item {n}", quantity=D(10), unit_price=D(5), amount=D(50))
            for n in range(lines)
        ],
    )


def purchase_order(tag: str) -> PurchaseOrder:
    return PurchaseOrder(
        po_number=f"PO-7-{tag}",
        vendor_name=vendor_name(tag),
        order_date=date(2026, 8, 20),
        contract_number=f"MSA-{tag}",
        total=D(100),
        line_items=[LineItem(description="Item 0", quantity=D(20), unit_price=D(5), amount=D(100))],
    )


def contract(tag: str) -> Contract:
    return Contract(
        contract_number=f"MSA-{tag}", vendor_name=vendor_name(tag), start_date=date(2026, 1, 1)
    )


def payment(tag: str, *numbers: str) -> Payment:
    allocations = [PaymentAllocation(invoice_number=f"{n}-{tag}", amount=D(60)) for n in numbers]
    return Payment(
        payment_reference=f"PAY-{tag}",
        vendor_name=vendor_name(tag),
        paid_at=date(2026, 10, 15),
        amount=D(60) * len(numbers),
        allocations=allocations,
    )


def snapshot(driver: Driver, tag: str) -> dict[str, Any]:
    """Every tagged node (by label) and edge (by type and endpoints)."""
    nodes = run_read(
        driver,
        "MATCH (n) WHERE n.id CONTAINS $tag RETURN labels(n)[0] AS label, n.id AS id",
        {"tag": tag},
    )
    edges = run_read(
        driver,
        "MATCH (a)-[r]->(b) WHERE a.id CONTAINS $tag OR b.id CONTAINS $tag "
        "RETURN a.id AS a, type(r) AS type, b.id AS b, properties(r) AS props",
        {"tag": tag},
    )
    return {
        "nodes": sorted((n["label"], n["id"]) for n in nodes),
        "edges": sorted(
            (e["a"], e["type"], e["b"], str(sorted(e["props"].items()))) for e in edges
        ),
    }


def edges_of(driver: Driver, tag: str, kind: str) -> list[tuple[str, str]]:
    return [(a, b) for a, t, b, _ in snapshot(driver, tag)["edges"] if t == kind]


def link_all(linker: GraphLinker, tag: str) -> None:
    linker.link(contract(tag), id=f"c-{tag}")
    linker.link(purchase_order(tag), id=f"po-{tag}")
    linker.link(invoice(tag), id=f"inv-{tag}")
    linker.link(payment(tag, "INV-1"), id=f"pay-{tag}")


def test_linking_twice_creates_no_duplicate_nodes_or_edges(
    driver: Driver, linker: GraphLinker, tag: str
) -> None:
    link_all(linker, tag)
    first = snapshot(driver, tag)
    link_all(linker, tag)

    assert snapshot(driver, tag) == first
    labels = [label for label, _ in first["nodes"]]
    assert sorted(set(labels)) == [
        "Contract",
        "Invoice",
        "LineItem",
        "Payment",
        "PurchaseOrder",
        "Vendor",
    ]
    assert labels.count("Vendor") == 1  # every record resolved to the same vendor
    assert len(first["edges"]) == len(set(first["edges"]))


def test_every_edge_type_is_created(driver: Driver, linker: GraphLinker, tag: str) -> None:
    link_all(linker, tag)
    vendor = run_read(
        driver, "MATCH (v:Vendor) WHERE v.id CONTAINS $tag RETURN v.id AS id", {"tag": tag}
    )[0]["id"]

    assert edges_of(driver, tag, "ISSUED") == [(vendor, f"inv-{tag}")]
    assert edges_of(driver, tag, "PARTY_TO") == [(vendor, f"c-{tag}")]
    assert edges_of(driver, tag, "GOVERNS") == [(f"c-{tag}", f"po-{tag}")]
    assert edges_of(driver, tag, "REFERENCES") == [(f"inv-{tag}", f"po-{tag}")]
    assert edges_of(driver, tag, "SETTLES") == [(f"pay-{tag}", f"inv-{tag}")]
    assert sorted(edges_of(driver, tag, "HAS_LINE")) == [
        (f"inv-{tag}", f"inv-{tag}#l0"),
        (f"inv-{tag}", f"inv-{tag}#l1"),
        (f"po-{tag}", f"po-{tag}#l0"),
    ]
    settles = run_read(
        driver,
        "MATCH (:Payment {id: $id})-[s:SETTLES]->() RETURN s.amount AS amount, "
        "s.invoice_number AS number",
        {"id": f"pay-{tag}"},
    )
    assert settles == [{"amount": 60.0, "number": f"INV-1-{tag}"}]


def test_linking_order_does_not_matter(driver: Driver, linker: GraphLinker, tag: str) -> None:
    linker.link(payment(tag, "INV-1"), id=f"pay-{tag}")  # nothing to settle yet
    linker.link(invoice(tag), id=f"inv-{tag}")  # picks up the payment
    linker.link(purchase_order(tag), id=f"po-{tag}")  # picks up the invoice
    result = linker.link(contract(tag), id=f"c-{tag}")  # picks up the PO
    reverse = snapshot(driver, tag)

    run_write(driver, "MATCH (n) WHERE n.id CONTAINS $tag DETACH DELETE n", {"tag": tag})
    link_all(linker, tag)

    assert snapshot(driver, tag) == reverse
    assert result.links["GOVERNS"] == [f"po-{tag}"]


def test_relinking_a_corrected_record_drops_stale_edges_and_lines(
    driver: Driver, linker: GraphLinker, tag: str
) -> None:
    link_all(linker, tag)

    linker.link(invoice(tag, po=None, lines=1), id=f"inv-{tag}")  # PO number and a line removed
    linker.link(payment(tag), id=f"pay-{tag}")  # allocation removed

    assert edges_of(driver, tag, "REFERENCES") == []
    assert edges_of(driver, tag, "SETTLES") == []
    assert [b for a, b in edges_of(driver, tag, "HAS_LINE") if a == f"inv-{tag}"] == [
        f"inv-{tag}#l0"
    ]
    assert ("LineItem", f"inv-{tag}#l1") not in snapshot(driver, tag)["nodes"]


def test_vendors_resolve_by_tax_id_then_fuzzy_name(
    driver: Driver, linker: GraphLinker, tag: str
) -> None:
    resolver = VendorResolver(driver)
    resolver.upsert(f"V-{tag}", vendor_name(tag), tax_id=f"GB {tag}", aliases=[f"ACME {tag}"])

    by_tax = linker.link(invoice(tag), id=f"inv-{tag}")
    slip = Invoice.model_validate(
        invoice(tag, number="INV-2").model_dump()
        | {"vendor_name": f"ACME {tag} SUPPLES LTD", "vendor_tax_id": None}
    )
    by_name = linker.link(slip, id=f"inv2-{tag}")

    assert (by_tax.vendor.vendor_id, by_tax.vendor.method) == (f"V-{tag}", "tax_id")
    assert (by_name.vendor.vendor_id, by_name.vendor.method) == (f"V-{tag}", "fuzzy")
    aliases = run_read(
        driver, "MATCH (v:Vendor {id: $id}) RETURN v.aliases AS a", {"id": f"V-{tag}"}
    )
    assert aliases[0]["a"] == [f"ACME {tag}", f"ACME {tag} SUPPLES LTD"]
    assert edges_of(driver, tag, "ISSUED") == [
        (f"V-{tag}", f"inv-{tag}"),
        (f"V-{tag}", f"inv2-{tag}"),
    ]


def test_given_vendor_ids_skip_resolution(driver: Driver, linker: GraphLinker, tag: str) -> None:
    result = linker.link(invoice(tag), id=f"inv-{tag}", vendor_id=f"V-{tag}")

    assert result.vendor.method == "given"
    names = run_read(driver, "MATCH (v:Vendor {id: $id}) RETURN v.name AS n", {"id": f"V-{tag}"})
    assert names == [{"n": vendor_name(tag)}]


def test_contract_rate_card_is_stored_as_parallel_lists(
    driver: Driver, linker: GraphLinker, tag: str
) -> None:
    record = contract(tag).model_copy(
        update={
            "rate_card": [
                RateCardEntry(item="Pallet wrap 500m", unit="roll", unit_price=D("44.67")),
                RateCardEntry(item="Consulting", unit_price=D(120)),
            ]
        }
    )

    linker.link(record, id=f"c-{tag}")

    rows = run_read(
        driver,
        "MATCH (c:Contract {id: $id}) RETURN c.rate_items AS items, c.rate_units AS units, "
        "c.rate_prices AS prices",
        {"id": f"c-{tag}"},
    )
    assert rows == [
        {
            "items": ["Pallet wrap 500m", "Consulting"],
            "units": ["roll", ""],
            "prices": [44.67, 120.0],
        }
    ]


def test_receipts_become_invoice_nodes(driver: Driver, linker: GraphLinker, tag: str) -> None:
    receipt = Receipt(vendor_name=f"Kedai {tag}", receipt_date=date(2026, 3, 1), total=D("9.50"))

    record_id = f"r-{tag}"
    linker.link(receipt, id=record_id)

    rows = run_read(
        driver,
        "MATCH (i:Invoice {id: $id}) RETURN i.doc_type AS t, i.invoice_number AS n, "
        "toString(i.invoice_date) AS d, i.total AS total",
        {"id": record_id},
    )
    number = f"RECEIPT-{record_id[:8]}"  # the same number Postgres stores
    assert rows == [{"t": "receipt", "n": number, "d": "2026-03-01", "total": 9.5}]


# --------------------------------------------------------------------------- unresolved links


def unresolved(linker: GraphLinker, tag: str) -> list[tuple[str, str, str]]:
    return [(u.source_id, u.rel, u.value) for u in linker.unresolved() if tag in u.source_id]


def test_missing_targets_are_recorded_and_cleared_when_they_arrive(
    linker: GraphLinker, tag: str
) -> None:
    paid = linker.link(payment(tag, "INV-1", "INV-404"), id=f"pay-{tag}")
    billed = linker.link(invoice(tag), id=f"inv-{tag}")
    ordered = linker.link(purchase_order(tag), id=f"po-{tag}")

    assert [u.value for u in paid.unresolved] == [f"INV-1-{tag}", f"INV-404-{tag}"]
    assert [(u.rel, u.value) for u in billed.unresolved] == [("REFERENCES", f"PO-7-{tag}")]
    assert [(u.rel, u.target_label) for u in ordered.unresolved] == [("GOVERNS", "Contract")]
    assert unresolved(linker, tag) == [
        (f"pay-{tag}", "SETTLES", f"INV-404-{tag}"),  # INV-1 arrived; the PO cleared REFERENCES
        (f"po-{tag}", "GOVERNS", f"MSA-{tag}"),
    ]

    linker.link(contract(tag), id=f"c-{tag}")

    assert unresolved(linker, tag) == [(f"pay-{tag}", "SETTLES", f"INV-404-{tag}")]
    assert linker.unresolved("REFERENCES") == [
        u for u in linker.unresolved("REFERENCES") if tag not in u.source_id
    ]


def test_invoice_waiting_for_its_po(linker: GraphLinker, tag: str) -> None:
    result = linker.link(invoice(tag), id=f"inv-{tag}")

    assert [(u.rel, u.key, u.value, u.target_label) for u in result.unresolved] == [
        ("REFERENCES", "po_number", f"PO-7-{tag}", "PurchaseOrder")
    ]
    assert unresolved(linker, tag) == [(f"inv-{tag}", "REFERENCES", f"PO-7-{tag}")]

    linker.link(purchase_order(tag), id=f"po-{tag}")

    assert (f"inv-{tag}", "REFERENCES", f"PO-7-{tag}") not in unresolved(linker, tag)


def test_relinking_keeps_one_record_per_unresolved_link(
    driver: Driver, linker: GraphLinker, tag: str
) -> None:
    for _ in range(2):
        linker.link(payment(tag, "INV-404"), id=f"pay-{tag}")

    count = run_read(
        driver,
        "MATCH (u:UnresolvedLink {source_id: $id}) RETURN count(u) AS n",
        {"id": f"pay-{tag}"},
    )
    assert count == [{"n": 1}]

    linker.link(payment(tag), id=f"pay-{tag}")  # the allocation was a mistake: removed
    assert unresolved(linker, tag) == []
