"""Upload -> queue -> worker -> ``ready``, through the API, a real arq queue and Postgres.

The fixture invoice is uploaded to ``POST /documents``; an arq worker (in-process, burst
mode, its own queue) ingests it with the rules extractor and a rules-only anomaly scorer
into a throwaway database; ``GET /documents/{id}`` then shows it ``ready`` with its
extracted fields, and ``GET /anomalies`` lists its score. Pinecone, Neo4j and the
embedding model are left out (the write path to them is test_write_path.py's job).

    docker compose up -d --wait postgres redis
    pytest -m integration tests/integration/test_api_upload_worker.py
"""

import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from arq.worker import Worker
from sqlalchemy.orm import Session, sessionmaker

from ledgerlens.anomaly.scorer import AnomalyScorer
from ledgerlens.api.main import create_app
from ledgerlens.api.routers import anomalies, documents
from ledgerlens.core.config import get_config
from ledgerlens.ingestion.loaders import FileLoader, PostgresRegistry
from ledgerlens.ingestion.pipeline import IngestionPipeline, make_extractor
from ledgerlens.ingestion.sinks import StoresSink
from ledgerlens.workers.ingest_worker import Ingestor, enqueue, ingest_document, redis_settings

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

FIXTURE = Path(__file__).parents[1] / "fixtures" / "documents" / "invoice_digital.pdf"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class Stack:
    def __init__(self, sessions: sessionmaker[Session], uploads: Path) -> None:
        self.queue = f"it-upload-{uuid.uuid4().hex[:8]}"
        config = get_config().ingestion
        sink = StoresSink(sessions=sessions, scorer=AnomalyScorer())
        self.ingestor = Ingestor(
            IngestionPipeline(
                config,
                loader=FileLoader(config.loader, registry=PostgresRegistry(sessions)),
                extractor=make_extractor("rules"),
                on_status=sink.mark,
            ),
            sink,
        )
        app = create_app()
        app.dependency_overrides[documents.get_registry] = lambda: PostgresRegistry(sessions)
        app.dependency_overrides[documents.get_raw_store] = lambda: documents.RawFileStore(uploads)
        app.dependency_overrides[documents.get_document_store] = lambda: documents.DocumentStore(
            sessions
        )
        app.dependency_overrides[documents.get_enqueue] = lambda: self.enqueue
        app.dependency_overrides[anomalies.get_anomaly_store] = lambda: anomalies.AnomalyStore(
            sessions
        )
        self.app = app

    async def enqueue(self, path: Path) -> str | None:
        (job_id,) = await enqueue([path], redis_settings(), queue_name=self.queue)
        return job_id

    async def work(self) -> None:
        worker = Worker(
            functions=[ingest_document],
            queue_name=self.queue,
            redis_settings=redis_settings(),
            burst=True,
            handle_signals=False,
            ctx={"ingestor": self.ingestor},
            max_jobs=1,
            poll_delay=0.05,
        )
        try:
            await worker.main()
        finally:  # Worker.close() needs SIGUSR1, which Windows lacks
            if worker.pool is not None:
                await worker.pool.aclose()


@pytest.fixture
async def stack(sessions: sessionmaker[Session], tmp_path: Path) -> AsyncIterator[Stack]:
    yield Stack(sessions, tmp_path)


async def test_an_uploaded_invoice_reaches_ready_through_the_worker(stack: Stack) -> None:
    transport = httpx.ASGITransport(app=stack.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api") as api:
        files = {"file": ("invoice_digital.pdf", FIXTURE.read_bytes(), "application/pdf")}
        uploaded = await api.post("/documents", files=files)
        assert uploaded.status_code == 202
        document_id = uploaded.json()["document_id"]
        queued = (await api.get(f"/documents/{document_id}")).json()
        assert queued["status"] == "queued" and queued["record"] is None

        await stack.work()

        ready: dict[str, Any] = (await api.get(f"/documents/{document_id}")).json()
        assert ready["status"] == "ready", ready["error"]
        assert ready["filename"] == "invoice_digital.pdf" and ready["doc_type"] == "invoice"
        assert ready["chunks"] >= 1 and ready["record_type"] == "invoice"
        record = ready["record"]
        assert record["invoice_number"] == "INV-2026-0417"
        assert record["total"] == "498.0000"  # rules: header fields, no line items

        scores = (await api.get("/anomalies", params={"min_score": 0})).json()
        assert [s["invoice_id"] for s in scores] == [ready["record_id"]]
        assert scores[0]["model_version"] == "rules-only"

        again = await api.post("/documents", files=files)
        assert again.status_code == 200
        assert again.json() == {
            "document_id": document_id,
            "status": "ready",
            "duplicate": True,
            "job_id": None,
        }
