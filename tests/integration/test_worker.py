"""The ingestion worker against the compose Redis: jobs go through a real arq queue.

The ingestor is a stub here (the write path itself is tested in test_write_path.py), so
this checks the queue plumbing: enqueue -> worker -> job result.
"""

import asyncio
import uuid
from pathlib import Path
from typing import Any

import pytest
from arq.jobs import Job
from arq.worker import Worker

from ledgerlens.workers.ingest_worker import (
    enqueue,
    ingest_document,
    redis_settings,
    startup,
)

pytestmark = pytest.mark.integration


class RecordingIngestor:
    def __init__(self) -> None:
        self.paths: list[str] = []

    def ingest(self, path: str) -> dict[str, Any]:
        self.paths.append(path)
        return {"path": path, "document_id": f"doc-{len(self.paths)}", "status": "ready"}


async def run_queue(paths: list[Path], ingestor: RecordingIngestor) -> list[Any]:
    queue = f"it-ingest-{uuid.uuid4().hex[:8]}"
    settings = redis_settings()
    job_ids = await enqueue(paths, settings, queue_name=queue)
    worker = Worker(
        functions=[ingest_document],
        queue_name=queue,
        redis_settings=settings,
        burst=True,  # stop when the queue is empty
        handle_signals=False,
        on_startup=startup,
        ctx={"ingestor": ingestor},
        max_jobs=1,
        poll_delay=0.05,
    )
    try:
        await worker.main()
        pool = worker.pool
        return [await Job(job_id, pool, _queue_name=queue).result(timeout=5) for job_id in job_ids]
    finally:
        # Worker.close() sends SIGUSR1 to itself, which Windows does not have; the pool is
        # the only resource a burst worker holds.
        if worker.pool is not None:
            await worker.pool.aclose()


def test_queued_files_are_ingested_by_the_worker(tmp_path: Path) -> None:
    paths = [tmp_path / f"{n}.pdf" for n in range(3)]
    ingestor = RecordingIngestor()

    results = asyncio.run(run_queue(paths, ingestor))

    # arq orders its queue by enqueue time in milliseconds; jobs queued in the same
    # millisecond come out in job-id order (random), so only the set of jobs is fixed.
    assert sorted(ingestor.paths) == sorted(str(p.resolve()) for p in paths)
    assert len(ingestor.paths) == len(paths)  # each job ran exactly once
    assert [r["status"] for r in results] == ["ready"] * 3
    assert [r["path"] for r in results] == [str(p.resolve()) for p in paths]  # by job id
