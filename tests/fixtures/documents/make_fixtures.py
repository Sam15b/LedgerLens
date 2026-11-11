"""Regenerate the sample documents in this folder and their expected extraction JSON.

    python tests/fixtures/documents/make_fixtures.py

Every document is rendered from the record it should extract to, so ``<name>.json`` is
always exactly what ``<name>.pdf`` / ``.png`` says. The records tell one story: a contract
between Acme Supplies and Northwind Traders, a PO under it, the invoice billing that PO,
and the payment settling the invoice, plus an unrelated cafe receipt that has been
"scanned" (rasterised, tilted and speckled) so it has no text layer.

Output is deterministic: fixed PDF metadata and ids, seeded noise.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from pathlib import Path

import pymupdf
from PIL import Image

from ledgerlens.core.schemas import (
    Contract,
    ExtractedModel,
    Invoice,
    LineItem,
    Payment,
    PaymentAllocation,
    PurchaseOrder,
    RateCardEntry,
    Receipt,
)

HERE = Path(__file__).resolve().parent
A4 = pymupdf.paper_rect("a4")
MARGIN = 56.0
FONT, BOLD = "helv", "hebo"
FIXED_DATE = "D:20261111090000+00'00'"

WRAP = LineItem(
    description="Pallet wrap 500m",
    quantity=Decimal(2),
    unit_price=Decimal("45.00"),
    amount=Decimal("90.00"),
)
LABELS = LineItem(
    description="Shipping labels (box of 500)",
    quantity=Decimal(10),
    unit_price=Decimal("32.50"),
    amount=Decimal("325.00"),
)

CONTRACT = Contract(
    contract_number="MSA-2026-07",
    title="Master Supply Agreement",
    vendor_name="Acme Supplies Ltd",
    customer_name="Northwind Traders",
    start_date=date(2026, 1, 1),
    end_date=date(2026, 12, 31),
    payment_terms_days=30,
    currency="GBP",
    rate_card=[
        RateCardEntry(item="Pallet wrap 500m", unit="roll", unit_price=Decimal("45.00")),
        RateCardEntry(item="Shipping labels (box of 500)", unit="box", unit_price=Decimal("32.50")),
        RateCardEntry(item="Courier", unit="hour", unit_price=Decimal("38.00")),
    ],
)
PURCHASE_ORDER = PurchaseOrder(
    po_number="PO-88231",
    vendor_name="Acme Supplies Ltd",
    buyer_name="Northwind Traders",
    order_date=date(2026, 9, 20),
    delivery_date=date(2026, 9, 30),
    contract_number="MSA-2026-07",
    currency="GBP",
    subtotal=Decimal("415.00"),
    tax=Decimal("83.00"),
    total=Decimal("498.00"),
    line_items=[WRAP, LABELS],
)
INVOICE = Invoice(
    invoice_number="INV-2026-0417",
    vendor_name="Acme Supplies Ltd",
    vendor_tax_id="GB123456789",
    customer_name="Northwind Traders",
    invoice_date=date(2026, 10, 2),
    due_date=date(2026, 11, 1),
    po_number="PO-88231",
    currency="GBP",
    subtotal=Decimal("415.00"),
    tax=Decimal("83.00"),
    total=Decimal("498.00"),
    line_items=[WRAP, LABELS],
)
PAYMENT = Payment(
    payment_reference="RA-5531",
    payer_name="Northwind Traders",
    vendor_name="Acme Supplies Ltd",
    paid_at=date(2026, 10, 30),
    method="bank_transfer",
    currency="GBP",
    amount=Decimal("498.00"),
    allocations=[PaymentAllocation(invoice_number="INV-2026-0417", amount=Decimal("498.00"))],
)
RECEIPT = Receipt(
    vendor_name="Kedai Kopi Sentosa",
    vendor_address="12 Jalan Bunga, 50450 Kuala Lumpur",
    receipt_number="R-000732",
    receipt_date=date(2026, 3, 14),
    currency="MYR",
    tax=Decimal("0.54"),
    total=Decimal("9.50"),
    line_items=[
        LineItem(
            description="Kopi O",
            quantity=Decimal(2),
            unit_price=Decimal("2.20"),
            amount=Decimal("4.40"),
        ),
        LineItem(
            description="Roti bakar",
            quantity=Decimal(1),
            unit_price=Decimal("5.10"),
            amount=Decimal("5.10"),
        ),
    ],
)


# --------------------------------------------------------------------------- drawing


class Writer:
    """Writes lines top to bottom, starting a new page when one is full."""

    def __init__(self, doc: pymupdf.Document, rect: pymupdf.Rect = A4, size: float = 10):
        self.doc, self.rect, self.size = doc, rect, size
        self.page = doc.new_page(width=rect.width, height=rect.height)
        self.y = MARGIN

    def line(
        self, text: str = "", *, bold: bool = False, size: float | None = None, x: float = MARGIN
    ) -> None:
        size = size or self.size
        if self.y + size > self.rect.height - MARGIN:
            self.page = self.doc.new_page(width=self.rect.width, height=self.rect.height)
            self.y = MARGIN
        if text:
            self.page.insert_text(
                (x, self.y + size), text, fontname=BOLD if bold else FONT, fontsize=size
            )
        self.y += size * 1.6

    def row(self, cells: Sequence[str], xs: Sequence[float], *, bold: bool = False) -> None:
        for text, x in zip(cells, xs, strict=True):
            self.page.insert_text(
                (x, self.y + self.size), text, fontname=BOLD if bold else FONT, fontsize=self.size
            )
        self.y += self.size * 1.6


def money(value: Decimal | None) -> str:
    return "" if value is None else f"{value:,.2f}"


def day(value: date | None) -> str:
    return "" if value is None else f"{value.day} {value:%B %Y}"


def line_table(w: Writer, items: Sequence[LineItem]) -> None:
    xs = (MARGIN, 330, 390, 470)
    w.row(("Description", "Qty", "Unit price", "Amount"), xs, bold=True)
    for item in items:
        w.row(
            (item.description, f"{item.quantity}", money(item.unit_price), money(item.amount)), xs
        )
    w.line()


def totals(
    w: Writer, subtotal: Decimal | None, tax: Decimal | None, total: Decimal, currency: str
) -> None:
    w.row(("Subtotal", money(subtotal)), (390, 470))
    w.row(("VAT 20%", money(tax)), (390, 470))
    w.row((f"Total ({currency})", money(total)), (390, 470), bold=True)


def save_pdf(doc: pymupdf.Document, name: str) -> None:
    doc.set_metadata(
        {
            "producer": "LedgerLens fixtures",
            "creator": "make_fixtures.py",
            "creationDate": FIXED_DATE,
            "modDate": FIXED_DATE,
            "title": name,
        }
    )
    doc.save(HERE / f"{name}.pdf", garbage=4, deflate=True, no_new_id=True)


def save_json(record: ExtractedModel, name: str) -> None:
    (HERE / f"{name}.json").write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- documents


def invoice_pdf() -> pymupdf.Document:
    inv, doc = INVOICE, pymupdf.open()
    w = Writer(doc)
    w.line("INVOICE", bold=True, size=20)
    w.line(f"{inv.vendor_name}, 4 Canal Wharf, Leeds LS11 5PS", bold=True)
    w.line(f"VAT reg. no. {inv.vendor_tax_id}")
    w.line()
    w.line(f"Invoice number: {inv.invoice_number}")
    w.line(f"Invoice date: {day(inv.invoice_date)}")
    w.line(f"Due date: {day(inv.due_date)} (Net 30)")
    w.line(f"Your PO number: {inv.po_number}")
    w.line(f"Bill to: {inv.customer_name}, 80 Market Street, Manchester M1 1PW")
    w.line()
    line_table(w, inv.line_items)
    totals(w, inv.subtotal, inv.tax, inv.total, inv.currency)
    w.line()
    w.line("Please pay by bank transfer quoting the invoice number. Sort code 20-00-00.")
    return doc


def purchase_order_pdf() -> pymupdf.Document:
    po, doc = PURCHASE_ORDER, pymupdf.open()
    w = Writer(doc)
    w.line("PURCHASE ORDER", bold=True, size=20)
    w.line(f"{po.buyer_name}, 80 Market Street, Manchester M1 1PW", bold=True)
    w.line()
    w.line(f"PO number: {po.po_number}")
    w.line(f"Order date: {day(po.order_date)}")
    w.line(f"Deliver by: {day(po.delivery_date)}")
    w.line(f"Supplier: {po.vendor_name}")
    w.line(f"Under contract: {po.contract_number}")
    w.line()
    line_table(w, po.line_items)
    totals(w, po.subtotal, po.tax, po.total, po.currency)
    w.line()
    w.line("Quote the PO number on every invoice. Goods not on this PO will not be paid.")
    return doc


def contract_pdf() -> pymupdf.Document:
    c, doc = CONTRACT, pymupdf.open()
    w = Writer(doc)
    w.line(f"{c.title.upper() if c.title else 'AGREEMENT'}", bold=True, size=18)
    w.line(f"Contract number: {c.contract_number}")
    w.line(f"Between {c.vendor_name} (the Supplier) and {c.customer_name} (the Customer).")
    w.line()
    clauses = [
        ("1. Term", f"This agreement runs from {day(c.start_date)} to {day(c.end_date)}."),
        (
            "2. Supply",
            "The Supplier will supply packaging and courier services on request by purchase order.",
        ),
        (
            "3. Prices",
            f"Prices are fixed for the term at the rates in Schedule 1, in "
            f"{c.currency}, excluding VAT.",
        ),
        (
            "4. Payment",
            f"The Customer will pay each valid invoice within "
            f"{c.payment_terms_days} days of the invoice date.",
        ),
        ("5. Invoices", "Every invoice must quote a valid purchase order number."),
    ]
    for heading, text in clauses:
        w.line(heading, bold=True)
        w.line(text)
        w.line()
    # Schedule on its own page: contracts are multi-page.
    w.page = doc.new_page(width=A4.width, height=A4.height)
    w.y = MARGIN
    w.line("Schedule 1 - Rate card", bold=True, size=14)
    xs = (MARGIN, 330, 430)
    w.row(("Item", "Unit", "Price"), xs, bold=True)
    for entry in c.rate_card:
        w.row((entry.item, entry.unit or "", money(entry.unit_price)), xs)
    return doc


def payment_pdf() -> pymupdf.Document:
    p, doc = PAYMENT, pymupdf.open()
    w = Writer(doc)
    w.line("REMITTANCE ADVICE", bold=True, size=20)
    w.line(f"From: {p.payer_name}")
    w.line(f"To: {p.vendor_name}")
    w.line()
    w.line(f"Payment reference: {p.payment_reference}")
    w.line(f"Payment date: {day(p.paid_at)}")
    w.line("Method: Bank transfer (BACS)")
    w.line()
    xs = (MARGIN, 390)
    w.row(("Invoice", f"Amount paid ({p.currency})"), xs, bold=True)
    for a in p.allocations:
        w.row((a.invoice_number, money(a.amount)), xs)
    w.line()
    w.row(("Total paid", money(p.amount)), xs, bold=True)
    return doc


def receipt_png() -> None:
    r = RECEIPT
    rect = pymupdf.Rect(0, 0, 230, 340)
    doc = pymupdf.open()
    w = Writer(doc, rect, size=8)
    w.y = 20
    w.line(r.vendor_name.upper(), bold=True, x=20)
    w.line(r.vendor_address or "", x=20)
    w.line(f"Receipt: {r.receipt_number}   Date: {r.receipt_date:%d/%m/%Y}", x=20)
    w.line("-" * 44, x=20)
    for item in r.line_items:
        w.row((f"{item.quantity} x {item.description}", money(item.amount)), (20, 170))
    w.line("-" * 44, x=20)
    w.row((f"TOTAL {r.currency}", money(r.total)), (20, 170), bold=True)
    w.row(("incl. SST 6%", money(r.tax)), (20, 170))
    w.line("Terima kasih!", x=20)

    pix = doc[0].get_pixmap(dpi=150, colorspace=pymupdf.csGRAY)
    image = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    image = image.rotate(1.5, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255)
    rng = random.Random(42)
    pixels = image.load()
    assert pixels is not None
    for _ in range(image.width * image.height // 150):  # scanner speckle
        x, y = rng.randrange(image.width), rng.randrange(image.height)
        pixels[x, y] = rng.choice((0, 90, 160))
    image.save(HERE / "receipt_scanned.png", optimize=True)


def main() -> None:
    for name, build, record in [
        ("invoice_digital", invoice_pdf, INVOICE),
        ("purchase_order", purchase_order_pdf, PURCHASE_ORDER),
        ("contract", contract_pdf, CONTRACT),
        ("payment_advice", payment_pdf, PAYMENT),
    ]:
        save_pdf(build(), name)
        save_json(record, name)
    receipt_png()
    save_json(RECEIPT, "receipt_scanned")
    for path in sorted(HERE.glob("*.*")):
        if path.suffix in {".pdf", ".png", ".json"}:
            print(f"{path.name:28} {path.stat().st_size:>7,} bytes")


if __name__ == "__main__":
    main()
