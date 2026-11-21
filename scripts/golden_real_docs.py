"""Real documents for the golden set: CORD v2 receipts and CUAD contracts, into the stores.

    python scripts/golden_real_docs.py prepare        # pick and write the files
    PINECONE_HOST=http://localhost:5080 python scripts/golden_real_docs.py ingest
    python scripts/golden_real_docs.py facts          # what each document says

``prepare`` writes ``data/raw/golden_real/``:

* ``cord/<split>-<n>.png`` -- receipts from CORD v2's test split, with ``cord/labels.jsonl``
  (the dataset's own parse: menu items, subtotal, tax, total). SROIE needs a registered
  manual download; CORD (CC BY 4.0) stands in for it.
* ``cuad/<slug>.pdf`` -- the shortest CUAD v1 contracts (CC BY 4.0), rendered to PDF with
  a text layer, with ``cuad/annotations.jsonl`` (the clause spans annotators marked).

``ingest`` runs the normal pipeline and stores sink (Postgres, Pinecone, Neo4j) on them as
``receipt`` / ``contract`` documents, with Tesseract run from the ``ll-tesseract`` Docker
image when it is not installed. They are stored for their **text** (chunks, vectors): no
record is extracted, so no vendor reaches the graph, where name matching could attach a
receipt to a synthetic vendor and change the synthetic golden set's answers. ``file ->
document_id`` is recorded in ``data/raw/golden_real/ingested.json``. A file already
ingested is a no-op (SHA-256 dedupe), so ``ingest`` is safe to run again.

``facts`` prints, per document, its id and the labelled facts found in its stored text, to
write golden questions from.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path("data/raw/golden_real")
CORD = Path("data/raw/cord/data")
CUAD = Path("data/raw/cuad/CUADv1.json")
CUAD_CATEGORIES = (
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Governing Law",
    "Cap On Liability",
    "Minimum Commitment",
    "Warranty Duration",
    "Insurance",
)


# --------------------------------------------------------------------------- prepare


def prepare_cord(n: int, seed: int) -> None:
    import pyarrow.parquet as pq

    out = ROOT / "cord"
    out.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(next(CORD.glob("test-*.parquet")))
    rows = list(range(table.num_rows))
    random.Random(seed).shuffle(rows)
    labels = []
    for k in rows[:n]:
        image = table.column("image")[k].as_py()
        parse = json.loads(table.column("ground_truth")[k].as_py())["gt_parse"]
        name = f"test-{k:03d}.png"
        from PIL import Image

        Image.open(io.BytesIO(image["bytes"])).convert("RGB").save(out / name)
        labels.append({"file": name, "parse": parse})
    (out / "labels.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in labels) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(labels)} receipts to {out}")


def _clean(text: str) -> str:
    """CUAD contexts pad words with runs of spaces; keep the paragraphs."""
    paragraphs = [re.sub(r"[ \t]+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    return "\n\n".join(p.replace("\n", " ") for p in paragraphs if p)


def render_pdf(text: str, path: Path, *, width: int = 100, per_page: int = 70) -> int:
    """Text to an A4 PDF with a text layer: wrapped lines, Helvetica 9 pt."""
    import textwrap

    import pymupdf

    lines: list[str] = []
    for paragraph in text.split("\n\n"):
        lines += textwrap.wrap(paragraph, width=width) or [""]
        lines.append("")
    doc = pymupdf.open()
    for start in range(0, len(lines), per_page):
        page = doc.new_page(width=595, height=842)
        for k, line in enumerate(lines[start : start + per_page]):
            if line:
                page.insert_text((50, 60 + 10.5 * k), line, fontsize=9, fontname="helv")
    pages = doc.page_count
    doc.save(path)
    return int(pages)


def prepare_cuad(n: int) -> None:
    out = ROOT / "cuad"
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads(CUAD.read_text(encoding="utf-8"))["data"]

    def annotated(contract: dict[str, Any]) -> set[str]:
        return {
            qa["id"].split("__")[-1]
            for qa in contract["paragraphs"][0]["qas"]
            if qa["answers"] and qa["id"].split("__")[-1] in CUAD_CATEGORIES
        }

    # the shortest contracts that state their governing law and at least four of the
    # clause types questions are written about
    useful = [c for c in data if "Governing Law" in annotated(c) and len(annotated(c)) >= 5]
    contracts = sorted(useful, key=lambda c: len(c["paragraphs"][0]["context"]))
    annotations: list[dict[str, Any]] = []
    slugs: set[str] = set()
    for contract in contracts:
        if len(annotations) == n:
            break
        context = contract["paragraphs"][0]["context"]
        slug = re.sub(r"[^A-Za-z0-9]+", "-", contract["title"]).strip("-")[:80]
        if slug in slugs:
            continue
        slugs.add(slug)
        pages = render_pdf(_clean(context), out / f"{slug}.pdf")
        spans: dict[str, list[str]] = {}
        for qa in contract["paragraphs"][0]["qas"]:
            category = qa["id"].split("__")[-1]
            if category in CUAD_CATEGORIES and qa["answers"]:
                spans[category] = [_clean(a["text"]) for a in qa["answers"]]
        annotations.append(
            {"file": f"{slug}.pdf", "title": contract["title"], "pages": pages, "spans": spans}
        )
    (out / "annotations.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in annotations) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(annotations)} contracts to {out}")


# --------------------------------------------------------------------------- ingest


def docker_tesseract() -> Any:
    """A :class:`TesseractEngine` that runs ``tesseract`` from the ``ll-tesseract`` image."""
    import csv

    from PIL import Image

    from ledgerlens.ingestion.ocr.preprocess import preprocess
    from ledgerlens.ingestion.ocr.tesseract import TesseractEngine, upscale_factor

    class DockerTesseract(TesseractEngine):
        def is_available(self) -> bool:
            return shutil.which("docker") is not None

        def run(self, image: Image.Image) -> list[Any]:
            factor = upscale_factor(image.size)
            cleaned = preprocess(image, self.config.preprocess, scale=factor).image
            with tempfile.TemporaryDirectory() as tmp:
                cleaned.save(Path(tmp) / "page.png")
                done = subprocess.run(
                    [
                        "docker",
                        "run",
                        "--rm",
                        "-v",
                        f"{Path(tmp).resolve()}:/work",
                        "ll-tesseract",
                        "/work/page.png",
                        "stdout",
                        "--psm",
                        str(self.config.tesseract_psm),
                        "tsv",
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    check=True,
                    timeout=120,
                )
            rows = list(
                csv.DictReader(io.StringIO(done.stdout), delimiter="\t", quoting=csv.QUOTE_NONE)
            )
            data: dict[str, list[Any]] = {key: [] for key in rows[0]} if rows else {}
            for row in rows:
                for key, value in row.items():
                    data[key].append(value)
            for key in ("left", "top", "width", "height", "conf"):
                data[key] = [float(v) for v in data.get(key, [])]
            return self.words_from_data(data, scale=float(factor))

    return DockerTesseract()


def ingest() -> None:
    from ledgerlens.core.config import get_config
    from ledgerlens.core.schemas import DocType
    from ledgerlens.extraction.base import ExtractionResult
    from ledgerlens.ingestion.loaders import FileLoader, PostgresRegistry
    from ledgerlens.ingestion.ocr.tesseract import TesseractEngine
    from ledgerlens.ingestion.pipeline import IngestionPipeline, default_embedder
    from ledgerlens.ingestion.sinks import StoresSink

    class TextOnly:
        """Extracts no record: these documents are stored for their text alone. A record
        would bring a vendor into the graph, where name matching could attach a receipt to
        a synthetic vendor and change the synthetic golden answers (counts, balances)."""

        name = "none"

        def extract(self, text: str, doc_type: DocType | str) -> ExtractionResult:
            kind = DocType(doc_type)
            reason = "golden-set document: stored for its text, no record extracted"
            return ExtractionResult(extractor=self.name, doc_type=kind, errors=[reason])

    config = get_config().ingestion
    native = TesseractEngine()
    ocr = native if native.is_available() else docker_tesseract()
    sink = StoresSink.from_settings()
    pipeline = IngestionPipeline(
        config,
        loader=FileLoader(config.loader, registry=PostgresRegistry()),
        ocr=ocr,
        extractor=TextOnly(),
        embedder=default_embedder(config),
        on_status=sink.mark,
    )
    files = sorted([*ROOT.glob("cord/*.png"), *ROOT.glob("cuad/*.pdf")])
    manifest_path = ROOT / "ingested.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for path in files:
        kind = DocType.RECEIPT if path.parent.name == "cord" else DocType.CONTRACT
        result = pipeline.process(path, doc_type=kind)
        document = sink.write(result)
        manifest[path.relative_to(ROOT).as_posix()] = {
            "document_id": document.id,
            "status": document.status.value,
            "doc_type": document.doc_type.value,
            "error": document.error,
        }
        print(f"{document.status.value:8} {document.doc_type.value:10} {path.name}")
        manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------- facts


def stored_text(document_id: str) -> str:
    from sqlalchemy import select

    from ledgerlens.storage.postgres.db import get_session_factory
    from ledgerlens.storage.postgres.models import ChunkRow

    with get_session_factory()() as session:
        rows = session.scalars(
            select(ChunkRow.text)
            .where(ChunkRow.document_id == document_id)
            .order_by(ChunkRow.chunk_index)
        )
        return "\n".join(rows)


def facts() -> None:
    manifest = json.loads((ROOT / "ingested.json").read_text())
    for line in (ROOT / "cord" / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        label = json.loads(line)
        entry = manifest[f"cord/{label['file']}"]
        flat = " ".join(stored_text(entry["document_id"]).split())
        parse = label["parse"]
        menu = parse.get("menu", [])
        menu = menu if isinstance(menu, list) else [menu]
        found = {
            key: (value, value in flat)
            for key, value in (
                ("total", (parse.get("total") or {}).get("total_price")),
                ("tax", (parse.get("sub_total") or {}).get("tax_price")),
                ("subtotal", (parse.get("sub_total") or {}).get("subtotal_price")),
            )
            if isinstance(value, str)
        }
        items = [(m.get("nm"), m.get("price"), str(m.get("nm")) in flat) for m in menu]
        print(
            json.dumps(
                {"file": label["file"], **entry, "facts": found, "items": items}, ensure_ascii=False
            )
        )
    for line in (ROOT / "cuad" / "annotations.jsonl").read_text(encoding="utf-8").splitlines():
        note = json.loads(line)
        entry = manifest[f"cuad/{note['file']}"]
        print(
            json.dumps(
                {
                    "file": note["file"],
                    **entry,
                    "title": note["title"],
                    "spans": {k: v[0][:220] for k, v in note["spans"].items()},
                },
                ensure_ascii=False,
            )
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--receipts", type=int, default=40)
    prep.add_argument("--contracts", type=int, default=24)
    prep.add_argument("--seed", type=int, default=64)
    sub.add_parser("ingest")
    sub.add_parser("facts")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare_cord(args.receipts, args.seed)
        prepare_cuad(args.contracts)
    elif args.command == "ingest":
        ingest()
    else:
        facts()
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
