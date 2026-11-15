"""SHA-256 dedupe on the ``documents`` table, across loaders, threads and restarts."""

import threading
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker

from ledgerlens.core.config import LoaderConfig
from ledgerlens.core.schemas import Document, DocumentStatus
from ledgerlens.ingestion.loaders import FileLoader, PostgresRegistry
from ledgerlens.storage.postgres.db import session_scope
from ledgerlens.storage.postgres.repositories import DocumentRepository

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).parents[2] / "fixtures" / "documents"
CONFIG = LoaderConfig(max_file_mb=5)


def test_a_second_loader_sees_the_first_loaders_upload(
    sessions: sessionmaker[Session],
) -> None:
    first = FileLoader(CONFIG, registry=PostgresRegistry(sessions)).load(FIXTURES / "contract.pdf")
    with sessions() as session:
        stored = DocumentRepository(session).get(first.document_id)
    assert stored is not None
    assert (stored.filename, stored.status, stored.pages) == (
        "contract.pdf",
        DocumentStatus.QUEUED,
        2,
    )
    with session_scope(sessions) as session:  # the first run finishes the document
        DocumentRepository(session).set_status(first.document_id, DocumentStatus.READY)

    # A new process: nothing shared with the first loader but the database.
    again = FileLoader(CONFIG, registry=PostgresRegistry(sessions)).load_bytes(
        (FIXTURES / "contract.pdf").read_bytes(), "renamed.pdf"
    )

    assert not first.duplicate
    assert again.duplicate and again.document_id == first.document_id
    assert PostgresRegistry(sessions).get(first.file.sha256) == first.document_id


@pytest.mark.parametrize("status", [DocumentStatus.QUEUED, DocumentStatus.FAILED])
def test_unfinished_documents_are_retried_not_skipped(
    sessions: sessionmaker[Session], status: DocumentStatus
) -> None:
    first = FileLoader(CONFIG, registry=PostgresRegistry(sessions)).load(FIXTURES / "contract.pdf")
    with session_scope(sessions) as session:
        DocumentRepository(session).set_status(first.document_id, status, error="ocr: boom")

    again = FileLoader(CONFIG, registry=PostgresRegistry(sessions)).load(FIXTURES / "contract.pdf")

    assert (again.duplicate, again.retry) == (False, True)
    assert again.document_id == first.document_id


def test_concurrent_claims_of_one_hash_have_a_single_winner(
    sessions: sessionmaker[Session],
) -> None:
    registry = PostgresRegistry(sessions)
    results: list[str | None] = []
    start = threading.Barrier(8)

    def claim(n: int) -> None:
        start.wait()
        results.append(registry.claim(Document(sha256="b" * 64, filename=f"copy-{n}.pdf")))

    threads = [threading.Thread(target=claim, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results.count(None) == 1
    assert {r for r in results if r is not None} == {registry.get("b" * 64)}
