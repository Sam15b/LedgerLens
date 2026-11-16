"""Rebuild the Pinecone vectors from the Postgres ``chunks`` table.

Re-embeds the chunks of every ``ready`` document (dense model from configs/ingestion.yaml,
BM25 refit on those chunks and saved for queries), replaces each document's vectors, and
prunes vectors whose document is gone. Use it after changing the embedding model, after
losing the index, or to fill a new namespace:

    python scripts/reindex_pinecone.py                 # this environment's namespace
    python scripts/reindex_pinecone.py --namespace eval
    python scripts/reindex_pinecone.py --keep-bm25     # reuse data/processed/bm25.json

Needs DATABASE_URL and PINECONE_* (the index must exist: ``make init``).
Exit codes: 0 rebuilt, 1 a store is unreachable or not configured.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence

from sqlalchemy.exc import OperationalError

from ledgerlens.core.logging import configure_logging
from ledgerlens.ingestion.embedding import BM25Encoder, DenseEmbedder
from ledgerlens.storage.pinecone_store import PineconeNotConfiguredError, PineconeStore
from ledgerlens.storage.postgres.db import get_session_factory
from ledgerlens.storage.rebuild import fit_bm25, ready_document_ids, reindex_pinecone


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--namespace", help="Pinecone namespace (default: from APP_ENV)")
    parser.add_argument(
        "--keep-bm25",
        action="store_true",
        help="load the saved BM25 params instead of refitting them on the stored chunks",
    )
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help="keep vectors whose document is not ready in Postgres",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    try:
        store = PineconeStore.from_settings(namespace=args.namespace)
    except PineconeNotConfiguredError as exc:
        print(f"error: {exc}")
        return 1
    sessions = get_session_factory()

    def progress(done: int, total: int) -> None:
        if done % 50 == 0 or done == total:
            print(f"  {done}/{total} documents")

    started = time.perf_counter()
    try:
        if args.keep_bm25:
            sparse = BM25Encoder.load()
        else:
            with sessions() as session:
                sparse = fit_bm25(session, ready_document_ids(session))
            if sparse.n_docs:  # nothing to fit on an empty database: keep the old params
                print(f"BM25 refit on {sparse.n_docs} stored chunks -> {sparse.save()}")
        result = reindex_pinecone(
            sessions,
            store,
            DenseEmbedder(),
            sparse=sparse,
            prune=not args.no_prune,
            progress=progress,
        )
    except (OSError, OperationalError) as exc:  # a store is down or unreachable
        print(f"error: {exc}")
        return 1
    elapsed = time.perf_counter() - started
    print(
        f"reindexed {result.documents} documents ({result.vectors} vectors) into "
        f"namespace {store.namespace!r} in {elapsed:.1f}s; pruned {result.pruned}; "
        f"{len(result.skipped)} ready documents had no chunks"
    )
    print(f"namespace now holds {store.count()} vectors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
