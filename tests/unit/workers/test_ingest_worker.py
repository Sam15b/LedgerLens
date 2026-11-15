import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from ledgerlens.core.schemas import Document, DocumentStatus
from ledgerlens.ingestion.loaders import UnsupportedFileError
from ledgerlens.workers import ingest_worker
from ledgerlens.workers.ingest_worker import Ingestor, WorkerSettings, ingest_document, main

DOCUMENT = Document(sha256="a" * 64, filename="inv.pdf")


@dataclass
class Result:
    document: Document
    duplicate: bool = False


@dataclass
class StubPipeline:
    duplicate: bool = False
    seen: list[str] = field(default_factory=list)

    def process(self, path: Path | str, *, embed: bool = True) -> Result:
        self.seen.append(str(path))
        if str(path).endswith(".txt"):
            raise UnsupportedFileError(f"{path}: text/plain is not supported")
        return Result(DOCUMENT.with_status(DocumentStatus.PROCESSING), self.duplicate)


class StubSink:
    def __init__(self, status: DocumentStatus = DocumentStatus.READY) -> None:
        self.status = status
        self.written: list[Result] = []

    def write(self, result: Any) -> Document:
        self.written.append(result)
        error = "neo4j: down" if self.status is DocumentStatus.FAILED else None
        return result.document.with_status(self.status, error=error)  # type: ignore[no-any-return]


def test_ingest_runs_the_pipeline_then_the_sink() -> None:
    sink = StubSink()

    outcome = Ingestor(StubPipeline(), sink).ingest("inv.pdf")

    assert outcome == {
        "path": "inv.pdf",
        "document_id": DOCUMENT.id,
        "status": "ready",
        "error": None,
        "duplicate": False,
    }
    assert len(sink.written) == 1


def test_storage_failures_are_reported_in_the_result() -> None:
    outcome = Ingestor(StubPipeline(), StubSink(DocumentStatus.FAILED)).ingest("inv.pdf")

    assert (outcome["status"], outcome["error"]) == ("failed", "neo4j: down")


def test_rejected_files_are_reported_not_raised() -> None:
    sink = StubSink()

    outcome = Ingestor(StubPipeline(), sink).ingest("notes.txt")

    assert outcome["status"] == "rejected" and "not supported" in outcome["error"]
    assert sink.written == []


def test_the_job_runs_the_ingestor_from_the_worker_context() -> None:
    pipeline = StubPipeline(duplicate=True)
    ctx: dict[str, Any] = {"ingestor": Ingestor(pipeline, StubSink()), "job_id": "job-1"}

    outcome = asyncio.run(ingest_document(ctx, "inv.pdf"))

    assert outcome["duplicate"] is True
    assert pipeline.seen == ["inv.pdf"]


def test_worker_runs_one_job_at_a_time_with_retries() -> None:
    assert WorkerSettings.functions == [ingest_document]
    assert (WorkerSettings.max_jobs, WorkerSettings.max_tries) == (1, 3)


def test_enqueue_command_queues_every_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ("a.pdf", "b.png", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    queued: list[list[Path]] = []

    async def fake_enqueue(paths: list[Path], *args: Any, **kwargs: Any) -> list[str]:
        queued.append(list(paths))
        return [f"job-{n}" for n in range(len(paths))]

    monkeypatch.setattr(ingest_worker, "enqueue", fake_enqueue)

    assert main(["enqueue", str(tmp_path)]) == 0
    assert [p.name for p in queued[0]] == ["a.pdf", "b.png"]  # only formats ingestion reads
    assert "queued 2 of 2 files" in capsys.readouterr().out
