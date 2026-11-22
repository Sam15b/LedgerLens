"""An evaluation run is stored in ``eval_runs`` (``pytest -m integration``, needs Postgres)."""

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker
from tests.unit.eval.test_runner import QUESTIONS, FakeRetriever

from eval.runner import config_files, config_fingerprint, git_revision, run_eval, store_run
from ledgerlens.core.config import repo_path
from ledgerlens.storage.postgres.models import EvalRunRow

pytestmark = pytest.mark.integration


def test_a_run_is_stored_with_git_sha_and_config_hash(sessions: sessionmaker[Session]) -> None:
    sha, _ = git_revision()
    golden = repo_path("data/golden/golden_ci.jsonl")
    config = config_fingerprint(config_files(golden), {"suite": "ci", "k": 5})
    run = run_eval(QUESTIONS, FakeRetriever(), suite="ci", config=config, git_sha=sha)

    row_id = store_run(run, sessions)

    with sessions() as session:
        row = session.scalars(select(EvalRunRow).where(EvalRunRow.id == row_id)).one()
    assert (row.suite, row.git_sha) == ("ci", sha)
    assert row.config["hash"] == config["hash"] and len(row.config["hash"]) == 64
    assert {"configs/eval.yaml", "data/golden/golden_ci.jsonl"} <= set(row.config["files"])
    assert row.config["run_id"] == run.run_id
    assert row.metrics["recall@5"] == 0.5
    assert row.created_at is not None
