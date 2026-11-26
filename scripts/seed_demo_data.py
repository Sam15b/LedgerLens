"""Seed a small, realistic demo ledger: a few vendors with a story worth clicking through.

Picks from the synthetic ledger (``data/synthetic/records.jsonl``) the vendors with the
most rendered documents and, for each, their contract and a mix of invoices -- paid,
partly paid and unpaid, plus every kind of anomaly the vendor has -- with the POs those
invoices bill against and the payments that settle them. One invoice from a vendor that
is not in the master is added too. The chosen records are loaded like
``load_synthetic.py`` loads everything: records into Postgres and Neo4j, their rendered
PDFs through the pipeline into Postgres and Pinecone.

    python scripts/seed_demo_data.py                          # 4 vendors x 8 invoices
    python scripts/seed_demo_data.py --vendors 3 --invoices 5
    python scripts/seed_demo_data.py --dry-run                # just list what would load

Meant for a fresh environment (``make up && make init`` first): it refits BM25 on the
demo documents only. Re-running is a no-op. No LLM is called: each rendered document is
stored with the ledger record it was rendered from.

The synthetic ledger is a local build (git-ignored, ~280 MB), so a cloud environment is
seeded from a **bundle**: the demo selection alone (records, manifest and PDFs, about
2 MB), made where the ledger exists and loaded where the stores are:

    python scripts/seed_demo_data.py --export demo-bundle.tar.gz
    aws s3 cp demo-bundle.tar.gz s3://<demo bucket>/demo/demo-bundle.tar.gz
    python scripts/seed_demo_data.py --bundle s3://<demo bucket>/demo/demo-bundle.tar.gz

A bundle is loaded whole (it already is the selection). In AWS the last step is the
``seed`` ECS task (infra/aws/README.md).

Exit codes: 0 seeded, 1 data missing or a store unreachable, 2 store counts disagree.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import tarfile
import tempfile
from collections import Counter, defaultdict
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from neo4j.exceptions import AuthError, ServiceUnavailable
from sqlalchemy.exc import OperationalError

from ledgerlens.core.config import repo_path
from ledgerlens.core.logging import configure_logging
from ledgerlens.storage.bulk_load import (
    LedgerLoader,
    Rendered,
    count_stores,
    load_rendered,
    parse_line,
    read_manifest,
)
from ledgerlens.storage.neo4j.driver import apply_schema, get_driver
from ledgerlens.storage.neo4j.linker import GraphLinker
from ledgerlens.storage.pinecone_store import PineconeNotConfiguredError, PineconeStore
from ledgerlens.storage.postgres.db import get_session_factory

DATA = repo_path("data/synthetic")
STATUS_ORDER = ("unpaid", "partial", "paid")  # unpaid first: the demo is about chasing them

Raw = dict[str, Any]


def select_demo(
    raw: Sequence[Raw], rendered: set[str], *, vendors: int = 4, invoices: int = 8
) -> list[Raw]:
    """The demo subset of ``raw`` ledger lines, in file order.

    Vendors: the ``vendors`` master vendors with the most rendered invoices. Per vendor:
    one invoice per anomaly kind it has, then a round robin over unpaid / partial / paid
    (rendered ones first) up to ``invoices``. Then the POs and contracts those invoices
    point to, the invoices the chosen duplicates copy, and the payments whose every
    invoice was chosen.
    """
    by_type: dict[str, list[Raw]] = defaultdict(list)
    for line in raw:
        by_type[line["record_type"]].append(line)
    master = {line["id"] for line in by_type["vendor"]}

    shown = Counter(
        line["vendor_id"]
        for line in by_type["invoice"]
        if line["id"] in rendered and line["vendor_id"] in master
    )
    chosen_vendors = sorted(shown, key=lambda v: (-shown[v], v))[:vendors]

    chosen: set[str] = set()
    for vendor_id in chosen_vendors:
        own = [line for line in by_type["invoice"] if line["vendor_id"] == vendor_id]
        own.sort(key=lambda line: (line["id"] not in rendered, line["id"]))
        picked: list[Raw] = []
        for kind in sorted({a for line in own for a in line["anomalies"]}):
            picked.append(next(line for line in own if kind in line["anomalies"]))
        queues = {s: [line for line in own if line["payment_status"] == s] for s in STATUS_ORDER}
        while len(picked) < invoices and any(queues.values()):
            for status in STATUS_ORDER:
                queue = queues[status]
                while queue and queue[0] in picked:
                    queue.pop(0)
                if queue and len(picked) < invoices:
                    picked.append(queue.pop(0))
        chosen.update(line["id"] for line in picked)
    stranger = [line for line in by_type["invoice"] if "unknown_vendor" in line["anomalies"]]
    stranger.sort(key=lambda line: (line["id"] not in rendered, line["id"]))
    chosen.update(line["id"] for line in stranger[:1])
    chosen.update(  # a duplicate is only worth showing next to the invoice it copies
        line["duplicate_of"]
        for line in by_type["invoice"]
        if line["id"] in chosen and line.get("duplicate_of")
    )

    invoice_lines = [line for line in by_type["invoice"] if line["id"] in chosen]
    po_numbers = {line["po_number"] for line in invoice_lines if line.get("po_number")}
    contracts = {line["contract_number"] for line in invoice_lines if line.get("contract_number")}
    contracts.update(
        line["contract_number"]
        for line in by_type["purchase_order"]
        if line["data"]["po_number"] in po_numbers and line.get("contract_number")
    )
    keep = set(chosen) | set(chosen_vendors)
    keep.update(
        line["id"] for line in by_type["purchase_order"] if line["data"]["po_number"] in po_numbers
    )
    keep.update(line["id"] for line in by_type["contract"] if line["id"] in contracts)
    keep.update(
        line["id"]
        for line in by_type["payment"]
        if line.get("invoice_ids") and set(line["invoice_ids"]) <= chosen
    )
    vendor_ids = {line["vendor_id"] for line in raw if line["id"] in keep and line.get("vendor_id")}
    keep.update(vendor_ids & master)  # vendors of the POs / contracts / payments chosen
    return [line for line in raw if line["id"] in keep]


BUNDLE_RECORDS = "records.jsonl"
BUNDLE_MANIFEST = "pdf/manifest.jsonl"


def export_bundle(demo: Sequence[Raw], rendered: Sequence[Rendered], root: Path, out: Path) -> Path:
    """Write ``demo`` and its ``rendered`` files to the ``.tar.gz`` ``out``.

    The layout is that of ``data/synthetic`` (``root``): ``records.jsonl``,
    ``pdf/manifest.jsonl`` and the files at their manifest paths, so a bundle unpacks
    into something :func:`read_manifest` reads as it is.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    records = "".join(json.dumps(line) + "\n" for line in demo)
    manifest = "".join(
        json.dumps(
            {
                "record_id": doc.record_id,
                "doc_type": doc.doc_type,
                "pdf": doc.path.relative_to(root).as_posix(),
            }
        )
        + "\n"
        for doc in rendered
    )
    with tarfile.open(out, "w:gz") as tar:
        for name, text in ((BUNDLE_RECORDS, records), (BUNDLE_MANIFEST, manifest)):
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for doc in rendered:
            tar.add(doc.path, arcname=doc.path.relative_to(root).as_posix())
    return out


def split_s3_uri(uri: str) -> tuple[str, str]:
    """``s3://bucket/key`` -> ``(bucket, key)``."""
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    if not uri.startswith("s3://") or not bucket or not key:
        raise ValueError(f"not an s3://bucket/key URI: {uri!r}")
    return bucket, key


def _download_s3(bucket: str, key: str, dest: Path) -> None:  # pragma: no cover - network
    import boto3  # the llm extra; credentials from the environment / task role

    boto3.client("s3").download_file(bucket, key, str(dest))


def fetch_bundle(source: str, workdir: Path) -> Path:
    """Unpack the bundle at ``source`` (a path or an ``s3://`` URI) into ``workdir``;
    returns the directory holding ``records.jsonl``."""
    workdir.mkdir(parents=True, exist_ok=True)
    archive = Path(source)
    if source.startswith("s3://"):
        archive = workdir / "bundle.tar.gz"
        _download_s3(*split_s3_uri(source), archive)
    root = workdir / "bundle"
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(root, filter="data")  # refuses absolute paths, .. and links out
    if not (root / BUNDLE_RECORDS).is_file():
        raise ValueError(f"{source} is not a demo bundle: no {BUNDLE_RECORDS} in it")
    return root


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("records", nargs="?", type=Path, default=DATA / "records.jsonl")
    parser.add_argument("--manifest", type=Path, default=DATA / "pdf" / "manifest.jsonl")
    parser.add_argument("--vendors", type=int, default=4, help="master vendors to include")
    parser.add_argument("--invoices", type=int, default=8, help="invoices per vendor")
    parser.add_argument("--namespace", help="Pinecone namespace (default: from APP_ENV)")
    parser.add_argument("--bm25-path", type=Path, help="where to save the BM25 params")
    parser.add_argument("--dry-run", action="store_true", help="list the selection and stop")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--export", type=Path, metavar="TAR_GZ", help="write the selection as a bundle and stop"
    )
    source.add_argument(
        "--bundle", metavar="PATH_OR_S3_URI", help="load this bundle instead of the ledger"
    )
    return parser.parse_args(argv)


def _read_jsonl(path: Path) -> list[Raw]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    for noisy in ("neo4j", "httpx", "pinecone"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    with ExitStack() as stack:
        if args.bundle:
            workdir = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="demo-")))
            try:
                root = fetch_bundle(args.bundle, workdir)
            except (OSError, ValueError, tarfile.TarError) as exc:
                print(f"error: cannot read the bundle {args.bundle}: {exc}")
                return 1
            demo = _read_jsonl(root / BUNDLE_RECORDS)
            manifest_path = root / BUNDLE_MANIFEST
            rendered = read_manifest(manifest_path) if manifest_path.exists() else []
        else:
            if not args.records.exists():
                print(
                    f"error: {args.records} not found -- build it with "
                    "`python -m ml.synthetic.build`"
                )
                return 1
            raw = _read_jsonl(args.records)
            manifest = read_manifest(args.manifest) if args.manifest.exists() else []
            demo = select_demo(
                raw, {m.record_id for m in manifest}, vendors=args.vendors, invoices=args.invoices
            )
            ids = {line["id"] for line in demo}
            rendered = [m for m in manifest if m.record_id in ids]
            if args.export:
                export_bundle(demo, rendered, args.manifest.parent.parent, args.export)
                size_kb = args.export.stat().st_size / 1024
                print(
                    f"wrote {args.export} ({len(demo)} records, {len(rendered)} documents, "
                    f"{size_kb:.0f} KB)"
                )
                return 0
        return _seed(args, demo, rendered)


def _seed(args: argparse.Namespace, demo: Sequence[Raw], rendered: Sequence[Rendered]) -> int:
    kinds = Counter(line["record_type"] for line in demo)
    print(f"demo: {dict(kinds)}; {len(rendered)} rendered documents")
    for line in demo:
        if line["record_type"] == "invoice":
            data = line["data"]
            flags = ", ".join(line["anomalies"]) or "-"
            print(
                f"  {line['id']:<12} {data['vendor_name'][:32]:<32} {data['invoice_number']:<12}"
                f" {data['currency']} {data['total']:>12}  {line['payment_status']:<8} {flags}"
            )
    if args.dry_run:
        return 0

    try:
        pinecone = PineconeStore.from_settings(namespace=args.namespace)
    except PineconeNotConfiguredError as exc:
        print(f"error: {exc}")
        return 1
    sessions = get_session_factory()
    driver = get_driver()
    try:
        apply_schema(driver)
        linker = GraphLinker(driver)
        loaded = LedgerLoader(sessions, linker).load(parse_line(line) for line in demo)
        docs = load_rendered(
            rendered,
            loaded,
            sessions=sessions,
            pinecone=pinecone,
            linker=linker,
            bm25_path=args.bm25_path,
        )
        counts = count_stores(sessions, pinecone, driver)
    except (OSError, OperationalError, ServiceUnavailable, AuthError) as exc:
        print(f"error: {exc}")
        return 1
    finally:
        driver.close()
    print(
        f"seeded {sum(loaded.records.values())} records, {loaded.vendors} vendors and "
        f"{docs.ready} documents ({docs.chunks} chunks; {docs.duplicates} already there)"
    )
    for name, error in sorted(docs.failed.items()):
        print(f"  failed {name}: {error}")
    problems = counts.mismatches()
    if problems:
        print("stores disagree:\n  " + "\n  ".join(problems))
        return 2
    print(f"parity: every store agrees ({counts.pinecone} vectors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
