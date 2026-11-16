"""Rebuild the Neo4j knowledge graph from the Postgres records.

MERGEs every vendor (with its aliases) and every record of a ready document -- contracts,
purchase orders, invoices, receipts, payments -- through the same linker as ingestion, so
edges and unresolved links come out as they did. Safe to run on a populated graph (it is
idempotent); ``--wipe`` empties the graph first so it matches Postgres exactly:

    python scripts/rebuild_graph.py           # fill in a fresh or lost graph
    python scripts/rebuild_graph.py --wipe    # drop everything in the graph, then rebuild

Needs DATABASE_URL and NEO4J_*. Prints node and relationship counts before and after.
Exit codes: 0 rebuilt, 1 Postgres or Neo4j unreachable.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence

from neo4j.exceptions import AuthError, ServiceUnavailable
from sqlalchemy.exc import OperationalError

from ledgerlens.core.logging import configure_logging
from ledgerlens.storage.neo4j.driver import get_driver
from ledgerlens.storage.neo4j.linker import GraphLinker
from ledgerlens.storage.postgres.db import get_session_factory
from ledgerlens.storage.rebuild import GraphCounts, graph_counts, rebuild_graph


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--wipe", action="store_true", help="delete every node in the graph before rebuilding"
    )
    return parser.parse_args(argv)


def show(title: str, counts: GraphCounts) -> None:
    nodes = ", ".join(f"{k}={v}" for k, v in counts.nodes.items()) or "none"
    rels = ", ".join(f"{k}={v}" for k, v in counts.relationships.items()) or "none"
    print(f"{title}: nodes {sum(counts.nodes.values())} ({nodes})")
    print(f"{' ' * len(title)}  relationships {sum(counts.relationships.values())} ({rels})")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    for noisy in ("neo4j",):  # per-request INFO lines drown the progress output
        logging.getLogger(noisy).setLevel(logging.WARNING)
    driver = get_driver()

    def progress(done: int, total: int) -> None:
        if done % 500 == 0 or done == total:
            print(f"  {done}/{total} records")

    started = time.perf_counter()
    try:
        show("before", graph_counts(driver))
        result = rebuild_graph(
            get_session_factory(), GraphLinker(driver), wipe=args.wipe, progress=progress
        )
        after = graph_counts(driver)
    except (ServiceUnavailable, AuthError, OperationalError) as exc:
        print(f"error: {exc}")
        return 1
    finally:
        driver.close()
    elapsed = time.perf_counter() - started
    if args.wipe:
        print(f"wiped {result.wiped} nodes")
    linked = ", ".join(f"{k}={v}" for k, v in sorted(result.records.items())) or "none"
    print(
        f"merged {result.vendors} vendors and {sum(result.records.values())} records "
        f"({linked}) in {elapsed:.1f}s; {result.unresolved} unresolved links"
    )
    show("after", after)
    return 0


if __name__ == "__main__":
    sys.exit(main())
