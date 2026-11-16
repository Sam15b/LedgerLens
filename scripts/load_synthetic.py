"""Bulk-load the synthetic ledger into Postgres, Neo4j and Pinecone.

Writes every record of ``records.jsonl`` to Postgres and the graph (structured: no OCR,
no extraction), then runs the rendered PDFs listed in ``pdf/manifest.jsonl`` through the
ingestion pipeline so their chunks land in Postgres and Pinecone, attached to the
records. Re-running is a no-op. Ends with row / vector / node counts per store and exits
non-zero if they disagree:

    python scripts/load_synthetic.py                       # data/synthetic, all of it
    python scripts/load_synthetic.py --pdf-limit 100       # fewer PDFs (embedding is slow)
    python scripts/load_synthetic.py --no-pdfs             # records only
    python scripts/load_synthetic.py --report out.json     # also write counts and timings

Needs ``make up && make init``. Pinecone: PINECONE_HOST=http://localhost:5080 for
Pinecone Local; the namespace comes from APP_ENV unless ``--namespace`` is given.
Exit codes: 0 loaded and the stores agree, 1 a store is unreachable, 2 counts disagree.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from neo4j.exceptions import AuthError, ServiceUnavailable
from sqlalchemy.exc import OperationalError

from ledgerlens.core.config import get_config, repo_path
from ledgerlens.core.logging import configure_logging
from ledgerlens.ingestion.embedding import BM25Encoder, ChunkEmbedder, DenseEmbedder
from ledgerlens.ingestion.loaders import FileLoader, PostgresRegistry
from ledgerlens.ingestion.pipeline import IngestionPipeline
from ledgerlens.storage.bulk_load import (
    KnownRecordExtractor,
    LedgerLoader,
    LedgerSink,
    count_stores,
    ingest_rendered,
    read_ledger,
    read_manifest,
)
from ledgerlens.storage.neo4j.driver import apply_schema, get_driver
from ledgerlens.storage.neo4j.linker import GraphLinker
from ledgerlens.storage.pinecone_store import PineconeNotConfiguredError, PineconeStore
from ledgerlens.storage.postgres.db import get_session_factory

DATA = repo_path("data/synthetic")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("records", nargs="?", type=Path, default=DATA / "records.jsonl")
    parser.add_argument("--manifest", type=Path, default=DATA / "pdf" / "manifest.jsonl")
    parser.add_argument("--no-pdfs", action="store_true", help="skip the rendered documents")
    parser.add_argument("--pdf-limit", type=int, help="ingest only the first N rendered files")
    parser.add_argument("--namespace", help="Pinecone namespace (default: from APP_ENV)")
    parser.add_argument("--report", type=Path, help="write counts and timings as JSON here")
    return parser.parse_args(argv)


def progress(stage: str, done: int, total: int) -> None:
    if done == total or done % 1000 == 0 or (stage in ("process", "embed") and done % 100 == 0):
        print(f"  {stage}: {done}/{total}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    for noisy in ("neo4j", "httpx", "pinecone"):  # per-request INFO lines
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if not args.records.exists():
        print(f"error: {args.records} not found -- build it with `python -m ml.synthetic.build`")
        return 1
    try:
        pinecone = PineconeStore.from_settings(namespace=args.namespace)
    except PineconeNotConfiguredError as exc:
        print(f"error: {exc}")
        return 1
    sessions = get_session_factory()
    driver = get_driver()
    out: dict[str, Any] = {"records_file": str(args.records), "namespace": pinecone.namespace}
    try:
        apply_schema(driver)
        lines = read_ledger(args.records)
        print(f"{len(lines)} ledger lines from {args.records}")
        linker = GraphLinker(driver)
        loaded = LedgerLoader(sessions, linker).load(lines, progress=progress)
        print(
            f"structured: {sum(loaded.records.values())} records {loaded.records}, "
            f"{loaded.vendors} vendors ({loaded.new_vendors} new); "
            + ", ".join(f"{k} {v:.1f}s" for k, v in loaded.timings_s.items())
        )
        out["structured"] = {k: v for k, v in asdict(loaded).items() if k != "assignments"}

        if not args.no_pdfs and args.manifest.exists():
            rendered = read_manifest(args.manifest)[: args.pdf_limit]
            config = get_config().ingestion
            extractor = KnownRecordExtractor()
            sink = LedgerSink(
                loaded.assignments, sessions=sessions, pinecone=pinecone, linker=linker
            )
            pipeline = IngestionPipeline(
                config,
                loader=FileLoader(config.loader, registry=PostgresRegistry(sessions)),
                extractor=extractor,
                embedder=ChunkEmbedder(DenseEmbedder(config.embedding), BM25Encoder()),
                on_status=sink.mark,
            )
            started = time.perf_counter()
            docs = ingest_rendered(
                rendered, loaded.assignments, pipeline, extractor, sink, progress=progress
            )
            elapsed = time.perf_counter() - started
            medians = {
                step: round(statistics.median(ms), 1) for step, ms in docs.step_ms.items() if ms
            }
            print(
                f"rendered: {docs.documents} files -> {docs.ready} ready, "
                f"{docs.duplicates} duplicates, {len(docs.failed)} failed, {docs.chunks} chunks "
                f"in {elapsed:.1f}s; median ms per step {medians}"
            )
            for name, error in sorted(docs.failed.items()):
                print(f"  failed {name}: {error}")
            out["rendered"] = asdict(docs) | {"median_step_ms": medians, "elapsed_s": elapsed}
            out["rendered"].pop("step_ms")

        counts = count_stores(sessions, pinecone, driver)
    except (OSError, OperationalError, ServiceUnavailable, AuthError) as exc:
        print(f"error: {exc}")
        return 1
    finally:
        driver.close()

    print(f"postgres: {counts.postgres}")
    print(f"pinecone: {counts.pinecone} vectors in namespace {pinecone.namespace!r}")
    print(f"neo4j:    {counts.neo4j}")
    problems = counts.mismatches()
    out["counts"] = asdict(counts) | {"mismatches": problems}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    if problems:
        print("stores disagree:\n  " + "\n  ".join(problems))
        return 2
    print("parity: every store agrees")
    return 0


if __name__ == "__main__":
    sys.exit(main())
