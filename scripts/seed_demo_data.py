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
demo documents only. Re-running is a no-op.
Exit codes: 0 seeded, 1 data missing or a store unreachable, 2 store counts disagree.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from neo4j.exceptions import AuthError, ServiceUnavailable
from sqlalchemy.exc import OperationalError

from ledgerlens.core.config import repo_path
from ledgerlens.core.logging import configure_logging
from ledgerlens.storage.bulk_load import (
    LedgerLoader,
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("records", nargs="?", type=Path, default=DATA / "records.jsonl")
    parser.add_argument("--manifest", type=Path, default=DATA / "pdf" / "manifest.jsonl")
    parser.add_argument("--vendors", type=int, default=4, help="master vendors to include")
    parser.add_argument("--invoices", type=int, default=8, help="invoices per vendor")
    parser.add_argument("--namespace", help="Pinecone namespace (default: from APP_ENV)")
    parser.add_argument("--bm25-path", type=Path, help="where to save the BM25 params")
    parser.add_argument("--dry-run", action="store_true", help="list the selection and stop")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    for noisy in ("neo4j", "httpx", "pinecone"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if not args.records.exists():
        print(f"error: {args.records} not found -- build it with `python -m ml.synthetic.build`")
        return 1
    with args.records.open(encoding="utf-8") as fh:
        raw = [json.loads(line) for line in fh if line.strip()]
    manifest = read_manifest(args.manifest) if args.manifest.exists() else []
    demo = select_demo(
        raw, {m.record_id for m in manifest}, vendors=args.vendors, invoices=args.invoices
    )
    ids = {line["id"] for line in demo}
    rendered = [m for m in manifest if m.record_id in ids]
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
