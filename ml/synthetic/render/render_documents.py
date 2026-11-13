"""Render ledger records to PDFs and page images.

    python -m ml.synthetic.render.render_documents data/synthetic/records.jsonl \\
        --out data/synthetic --limit 200

reads ``records.jsonl`` (:mod:`ml.synthetic.build`) and writes, per record:

* ``pdf/<doc_type>/<record id>.pdf`` -- a digital PDF with a text layer, and
* ``pages/<doc_type>/<record id>-p<n>.png`` -- one image per page (the clean scans that
  the degradation step, Day 24, starts from),

plus ``pdf/manifest.jsonl`` linking each file back to its record id and template.

Templates are Jinja2 HTML (``templates/``): three invoice layouts -- picked per record id,
so a vendor's invoices vary the way real inboxes do -- plus purchase order, payment
advice and contract. HTML becomes PDF through **WeasyPrint** when it is installed and
working, else through **PyMuPDF's Story** (WeasyPrint needs GTK, which Windows does not
ship); the stylesheet keeps to the CSS both understand.

Output is deterministic: the same record renders to the same bytes (fixed metadata, no
random file id). An exact-duplicate invoice renders with its source's layout and title,
so it is the same PDF byte for byte -- what a resent PDF looks like to ingestion's
sha256 dedupe. Near duplicates are re-issued documents and may get another layout.
"""

from __future__ import annotations

import argparse
import importlib
import io
import json
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from decimal import Decimal
from functools import cache
from pathlib import Path
from typing import Any, Literal

import pymupdf
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from ml.synthetic.seed import BUYER

TEMPLATES = Path(__file__).with_name("templates")
INVOICE_LAYOUTS = ("invoice_classic", "invoice_modern", "invoice_compact")
TEMPLATE_FOR = {
    "purchase_order": "purchase_order",
    "payment": "payment_advice",
    "contract": "contract",
}
DOC_TYPES = ("invoice", "purchase_order", "payment", "contract")
PAGE = pymupdf.paper_rect("a4")
MARGIN = 42  # points
Backend = Literal["weasyprint", "pymupdf"]

BUYER_ADDRESS = ["1 Market Street", "London EC2A 4BX", "United Kingdom"]
# Fictional street addresses by country (vendor addresses are not in the vendor master).
ADDRESSES: dict[str, list[list[str]]] = {
    "GB": [
        ["14 Harbour Road", "Bristol BS1 4AA", "United Kingdom"],
        ["Unit 7, Kingsway Park", "Manchester M19 2LL", "United Kingdom"],
        ["220 Station Road", "Leeds LS1 5DY", "United Kingdom"],
    ],
    "DE": [
        ["Industriestraße 12", "70565 Stuttgart", "Germany"],
        ["Hafenweg 3", "20457 Hamburg", "Germany"],
    ],
    "FR": [
        ["18 rue des Entrepôts", "69007 Lyon", "France"],
        ["5 avenue Foch", "75116 Paris", "France"],
    ],
    "NL": [["Havenstraat 41", "3011 Rotterdam", "Netherlands"]],
    "MY": [
        ["Lot 12, Jalan Perusahaan 2", "40150 Shah Alam, Selangor", "Malaysia"],
        ["No. 8, Jalan Tun Razak", "50400 Kuala Lumpur", "Malaysia"],
    ],
    "SG": [["30 Tuas Avenue 2", "Singapore 639461", "Singapore"]],
    "ID": [
        ["Jl. Gatot Subroto No. 21", "Jakarta 12930", "Indonesia"],
        ["Jl. Rungkut Industri III/7", "Surabaya 60293", "Indonesia"],
    ],
    "US": [
        ["4100 Commerce Drive", "Columbus, OH 43219", "United States"],
        ["77 Mission Street, Suite 300", "San Francisco, CA 94105", "United States"],
    ],
}
UNKNOWN_ADDRESSES = [["PO Box 4410", "Dubai", "United Arab Emirates"], ["Suite 9, 1 Main Road"]]


def _stable(text: str) -> int:
    return zlib.crc32(text.encode("utf-8"))


def address_for(vendor: Mapping[str, Any] | None, name: str) -> list[str]:
    """A fictional, stable street address for a vendor (or an unknown one, by name)."""
    if vendor is None:
        return UNKNOWN_ADDRESSES[_stable(name) % len(UNKNOWN_ADDRESSES)]
    options = ADDRESSES.get(str(vendor["country"]), UNKNOWN_ADDRESSES)
    return options[_stable(str(vendor["vendor_id"])) % len(options)]


def document_key(record: Mapping[str, Any]) -> str:
    """What the rendered file is keyed on: the record id, except that an exact duplicate
    uses its source's id -- a resent invoice is the same PDF, byte for byte."""
    if "exact_duplicate" in (record.get("anomalies") or []) and record.get("duplicate_of"):
        return str(record["duplicate_of"])
    return str(record["id"])


def template_for(record: Mapping[str, Any]) -> str:
    kind = str(record["record_type"])
    if kind == "invoice":
        return INVOICE_LAYOUTS[_stable(document_key(record)) % len(INVOICE_LAYOUTS)]
    return TEMPLATE_FOR[kind]


# --------------------------------------------------------------------------- Jinja filters


def format_amount(value: Any, currency: str = "") -> str:
    """``1234.5`` -> ``1,234.50``; IDR without decimals (``1,234,500``)."""
    if value is None or value == "":
        return ""
    amount = Decimal(str(value))
    if currency == "IDR":
        return f"{amount:,.0f}"
    return f"{amount:,.2f}"


def format_date(value: Any, style: str = "long") -> str:
    day = value if isinstance(value, date) else date.fromisoformat(str(value))
    if style == "iso":
        return day.isoformat()
    if style == "us":
        return f"{day:%B} {day.day}, {day.year}"
    return f"{day.day:02d} {day:%b} {day.year}"


def format_qty(value: Any) -> str:
    if value is None:
        return ""
    qty = Decimal(str(value))
    return f"{qty:,.0f}" if qty == qty.to_integral_value() else f"{qty.normalize():,f}"


def format_method(value: Any) -> str:
    return str(value or "other").replace("_", " ").capitalize()


@cache
def environment() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES),
        autoescape=select_autoescape(["j2", "html"]),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters.update(amount=format_amount, date=format_date, qty=format_qty, method=format_method)
    return env


@cache
def stylesheet() -> str:
    return (TEMPLATES / "styles.css").read_text(encoding="utf-8")


def _tax_rate(data: Mapping[str, Any]) -> str:
    subtotal, tax = data.get("subtotal"), data.get("tax")
    if subtotal in (None, "") or tax in (None, "") or Decimal(str(subtotal)) == 0:
        return ""
    rate = Decimal(str(tax)) / Decimal(str(subtotal)) * 100
    return f"{rate.quantize(Decimal(1))}%"


def render_html(
    record: Mapping[str, Any], vendors: Mapping[str, Mapping[str, Any]], *, template: str
) -> str:
    """The record's document as HTML (body only; the stylesheet is applied separately)."""
    data = record["data"]
    vendor = vendors.get(str(record.get("vendor_id")))
    terms = data.get("payment_terms_days")
    if terms is None and data.get("due_date") and data.get("invoice_date"):
        terms = (
            date.fromisoformat(data["due_date"]) - date.fromisoformat(data["invoice_date"])
        ).days
    context = {
        "doc": data,
        "vendor_address": address_for(vendor, str(data.get("vendor_name", ""))),
        "customer_address": BUYER_ADDRESS,
        "tax_rate": _tax_rate(data),
        "terms_days": terms,
        "buyer": BUYER,
    }
    return environment().get_template(f"{template}.html.j2").render(**context)


# --------------------------------------------------------------------------- HTML -> PDF


@cache
def _weasyprint() -> Any | None:
    """The WeasyPrint module if it imports and its native libraries (GTK/Pango) load."""
    try:
        module = importlib.import_module("weasyprint")
        module.HTML(string="<p>probe</p>").write_pdf()
    except Exception:  # missing package, or OSError for missing GTK libraries on Windows
        return None
    return module


def available_backend() -> Backend:
    return "weasyprint" if _weasyprint() is not None else "pymupdf"


def html_to_pdf(html: str, *, title: str, backend: Backend | None = None) -> bytes:
    """``html`` (styled with ``templates/styles.css``) as deterministic A4 PDF bytes."""
    backend = backend or available_backend()
    if backend == "weasyprint":
        weasyprint = _weasyprint()
        if weasyprint is None:
            raise RuntimeError("WeasyPrint is not available")
        css = f"@page {{ size: A4; margin: {MARGIN}pt; }}\n{stylesheet()}"
        raw = weasyprint.HTML(string=html).write_pdf(stylesheets=[weasyprint.CSS(string=css)])
    else:
        raw = _story_pdf(html)
    return _normalize(bytes(raw), title)


def _story_pdf(html: str) -> bytes:
    story = pymupdf.Story(html, user_css=stylesheet())
    buffer = io.BytesIO()
    writer = pymupdf.DocumentWriter(buffer)
    area = PAGE + (MARGIN, MARGIN, -MARGIN, -MARGIN)
    more = True
    while more:
        device = writer.begin_page(PAGE)
        more, _ = story.place(area)
        story.draw(device)
        writer.end_page()
    writer.close()
    return buffer.getvalue()


def _normalize(pdf: bytes, title: str) -> bytes:
    """Fixed metadata and no random file id, so equal inputs give equal bytes."""
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        doc.set_metadata(
            {"title": title, "author": "LedgerLens synthetic data", "producer": "LedgerLens"}
        )
        return bytes(doc.tobytes(garbage=3, deflate=True, no_new_id=True))


def page_images(pdf: bytes, *, dpi: int = 150) -> list[bytes]:
    """PNG bytes of every page, rendered in grayscale at ``dpi``."""
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        return [page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY).tobytes("png") for page in doc]


def pdf_text(pdf: bytes) -> str:
    """The text layer as ingestion reads it (ligatures such as "ﬁ" expanded to "fi")."""
    flags = pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_PRESERVE_LIGATURES
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        return "\n".join(str(page.get_text("text", flags=flags)) for page in doc)


# --------------------------------------------------------------------------- records -> files


@dataclass
class Rendered:
    record_id: str
    doc_type: str
    template: str
    backend: str
    pdf: str
    pages: list[str] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)


def render_record(
    record: Mapping[str, Any],
    vendors: Mapping[str, Mapping[str, Any]],
    out: Path,
    *,
    dpi: int | None = 150,
    backend: Backend | None = None,
) -> Rendered:
    """Write one record's PDF (and page images unless ``dpi`` is ``None``) under ``out``."""
    kind, record_id = str(record["record_type"]), str(record["id"])
    template = template_for(record)
    backend = backend or available_backend()
    pdf = html_to_pdf(
        render_html(record, vendors, template=template),
        title=document_key(record),
        backend=backend,
    )
    pdf_path = out / "pdf" / kind / f"{record_id}.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(pdf)
    result = Rendered(
        record_id,
        kind,
        template,
        backend,
        pdf_path.relative_to(out).as_posix(),
        anomalies=list(record.get("anomalies") or []),
    )
    if dpi:
        for n, png in enumerate(page_images(pdf, dpi=dpi), start=1):
            image_path = out / "pages" / kind / f"{record_id}-p{n}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(png)
            result.pages.append(image_path.relative_to(out).as_posix())
    return result


def vendors_by_id(records: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(r["id"]): r["data"] for r in records if r["record_type"] == "vendor"}


def select(
    records: Sequence[Mapping[str, Any]], doc_types: Sequence[str], limit: int | None
) -> list[Mapping[str, Any]]:
    """Records of ``doc_types``, at most ``limit`` of each type.

    A limited selection is spread evenly over the file (which is in date order, with
    duplicate invoices appended last), so a sample covers the whole period and every
    kind of anomaly rather than the first weeks only.
    """
    chosen: list[Mapping[str, Any]] = []
    for kind in doc_types:
        of_kind = [r for r in records if r["record_type"] == kind]
        if limit is not None and limit < len(of_kind):
            step = len(of_kind) / limit
            of_kind = [of_kind[int(i * step)] for i in range(limit)]
        chosen += of_kind
    return chosen


def main(argv: Sequence[str] | None = None) -> int:
    from ml.synthetic.build import read_records

    parser = argparse.ArgumentParser(description="Render records.jsonl to PDFs and page images")
    parser.add_argument(
        "records", type=Path, nargs="?", default=Path("data/synthetic/records.jsonl")
    )
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--types", nargs="+", choices=DOC_TYPES, default=list(DOC_TYPES))
    parser.add_argument("--limit", type=int, help="at most this many records per type")
    parser.add_argument("--dpi", type=int, default=150, help="page image resolution")
    parser.add_argument("--no-images", action="store_true", help="PDFs only")
    parser.add_argument("--backend", choices=["weasyprint", "pymupdf"])
    args = parser.parse_args(argv)

    records = list(read_records(args.records))
    vendors = vendors_by_id(records)
    chosen = select(records, args.types, args.limit)
    backend = args.backend or available_backend()
    manifest = args.out / "pdf" / "manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with manifest.open("w", encoding="utf-8", newline="\n") as fh:
        for record in chosen:
            rendered = render_record(
                record, vendors, args.out, dpi=None if args.no_images else args.dpi, backend=backend
            )
            fh.write(json.dumps(asdict(rendered)) + "\n")
            counts[rendered.doc_type] = counts.get(rendered.doc_type, 0) + 1
    print(f"{sum(counts.values())} documents {counts} via {backend} -> {args.out / 'pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
