"""Background ingestion: an arq worker that consumes ingestion jobs from Redis.

Each job is one file path (shared storage; object storage comes with the API upload
path). The worker runs the full pipeline for it and writes the result to all three stores
through :class:`~ledgerlens.ingestion.sinks.StoresSink`, so a job is safe to run twice --
arq retries a job whose worker died, and the loader's SHA-256 dedupe turns a finished file
into a no-op while an unfinished one is picked up again.

::

    python -m ledgerlens.workers.ingest_worker enqueue data/raw/sroie/*.jpg
    python -m ledgerlens.workers.ingest_worker run        # or: make worker

Embedding one document needs the BM25 parameters a batch run saved
(``embedding.sparse_params_path``); without them, storing fails at the Pinecone step and
the document is marked ``failed`` with that reason.

Jobs run one at a time per worker process (``max_jobs = 1``): the embedding model and the
OCR engine are not shared between threads. Scale out with more worker processes.

``run`` needs a POSIX system (arq's worker relies on Unix signals): use Docker or WSL on
Windows. ``enqueue`` works everywhere.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol

from arq import create_pool
from arq.connections import RedisSettings
from arq.worker import run_worker

from ledgerlens.core.config import get_config, get_settings
from ledgerlens.core.logging import configure_logging, request_context
from ledgerlens.core.schemas import Document
from ledgerlens.ingestion.loaders import LoaderError

logger = logging.getLogger(__name__)

JOB_NAME = "ingest_document"


class _Pipeline(Protocol):
    def process(self, path: Path | str, *, embed: bool = True) -> Any: ...


class _Sink(Protocol):
    def write(self, result: Any) -> Document: ...


@dataclass
class Ingestor:
    """Ingests one file: pipeline, then the sink. What a job runs."""

    pipeline: _Pipeline
    sink: _Sink

    def ingest(self, path: str) -> dict[str, Any]:
        """Returns the job result: document id, final status, error and whether the file
        was a duplicate. A file the loader rejects is reported, not raised -- retrying a
        bad file cannot help."""
        try:
            result = self.pipeline.process(path)
        except LoaderError as exc:
            logger.warning("rejected %s: %s", path, exc)
            return {"path": path, "document_id": None, "status": "rejected", "error": str(exc)}
        document = self.sink.write(result)
        return {
            "path": path,
            "document_id": document.id,
            "status": document.status.value,
            "error": document.error,
            "duplicate": bool(result.duplicate),
        }


def build_ingestor() -> Ingestor:
    """The production ingestor: Postgres dedupe, configured extractor and embedder, and
    status updates written as the document moves through the pipeline."""
    from ledgerlens.ingestion.loaders import FileLoader, PostgresRegistry
    from ledgerlens.ingestion.pipeline import IngestionPipeline, default_embedder
    from ledgerlens.ingestion.sinks import StoresSink

    config = get_config().ingestion
    sink = StoresSink.from_settings()
    pipeline = IngestionPipeline(
        config,
        loader=FileLoader(config.loader, registry=PostgresRegistry()),
        embedder=default_embedder(config),
        on_status=sink.mark,
    )
    return Ingestor(pipeline, sink)


# --------------------------------------------------------------------------- arq


async def ingest_document(ctx: dict[str, Any], path: str) -> dict[str, Any]:
    """The arq job: ingest ``path`` off the event loop (the pipeline is blocking)."""
    ingestor: Ingestor = ctx["ingestor"]
    job_id = str(ctx.get("job_id") or "")
    with request_context(job_id or "ingest"):
        return await asyncio.to_thread(ingestor.ingest, path)


async def startup(ctx: dict[str, Any]) -> None:
    configure_logging()
    if "ingestor" not in ctx:  # tests put a stub in ctx
        ctx["ingestor"] = await asyncio.to_thread(build_ingestor)


def redis_settings() -> RedisSettings:
    return RedisSettings.from_dsn(get_settings().redis_url)


class WorkerSettings:
    """``arq ledgerlens.workers.ingest_worker.WorkerSettings`` also starts the worker."""

    functions: ClassVar = [ingest_document]
    on_startup = startup
    redis_settings = redis_settings()
    max_jobs = 1
    job_timeout = 600  # seconds: OCR of a long scanned PDF on CPU
    max_tries = 3


async def enqueue(
    paths: Sequence[Path | str],
    settings: RedisSettings | None = None,
    *,
    queue_name: str | None = None,
) -> list[str]:
    """Queue one job per file (on arq's default queue unless named); returns job ids."""
    pool = await create_pool(settings or redis_settings())
    try:
        ids = []
        for path in paths:
            job = await pool.enqueue_job(
                JOB_NAME, str(Path(path).resolve()), _queue_name=queue_name
            )
            if job is not None:
                ids.append(job.job_id)
        return ids
    finally:
        await pool.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ledgerlens.workers.ingest_worker")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run", help="start a worker (Ctrl+C to stop)")
    queue = commands.add_parser("enqueue", help="queue files for ingestion")
    queue.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)

    if args.command == "run":
        run_worker(WorkerSettings)  # type: ignore[arg-type]
        return 0
    from ledgerlens.ingestion.pipeline import find_inputs

    files = [f for p in args.paths for f in find_inputs(p)]
    ids = asyncio.run(enqueue(files))
    print(f"queued {len(ids)} of {len(files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
