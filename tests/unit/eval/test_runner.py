"""The evaluation runner: retrieval and answers per question, per-query cost, RAGAS when
a model answers, provenance (git sha, config hash) -- all without stores or keys."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from eval.golden_set.schema import GoldenQuestion
from eval.latency_cost import cost_report
from eval.run_ragas import Sample
from eval.runner import (
    RagasOutcome,
    config_fingerprint,
    eval_run_values,
    git_revision,
    query_request_id,
    run_eval,
    write_reports,
)
from ledgerlens.core.config import ModelsConfig
from ledgerlens.llm.router import LLMClient
from ledgerlens.llm.types import LLMProviderError, LLMRequest, LLMResponse, Usage
from ledgerlens.observability.cost import CostLedger, LLMCallRecord, document_request_id
from ledgerlens.retrieval import QueryFilters, RetrievalResult, ScoredChunk

DOC_A = "00000000-0000-4000-8000-00000000000a"
DOC_B = "00000000-0000-4000-8000-00000000000b"

QUESTIONS = [
    GoldenQuestion(
        id="q001",
        category="lookup",
        question="What is the total on invoice INV-000001?",
        ground_truth_answer="Invoice INV-000001 totals 100.00 USD.",
        relevant_document_ids=(DOC_A,),
        requires_graph=False,
        difficulty="easy",
        expected={"amount": "100.00"},
    ),
    GoldenQuestion(
        id="q002",
        category="lookup",
        question="What is the total on invoice INV-000002?",
        ground_truth_answer="Invoice INV-000002 totals 250.00 USD.",
        relevant_document_ids=(DOC_B,),
        requires_graph=False,
        difficulty="easy",
        expected={"amount": "250.00"},
    ),
    GoldenQuestion(
        id="q003",
        category="unanswerable",
        question="What is the total on invoice INV-999999?",
        ground_truth_answer="There is no invoice INV-999999 in the records.",
        requires_graph=False,
        difficulty="easy",
        expected={"answerable": False},
    ),
]

MODELS = ModelsConfig.model_validate(
    {
        "aliases": {"smart": ["openai/big"], "judge": ["openai/big"]},
        "pricing_usd_per_1m_tokens": {"openai/big": {"in": 2.0, "out": 10.0}},
    }
)


class FakeRetriever:
    """Finds document A for every question (so q002 misses)."""

    name = "r4_graphrag"

    def retrieve(
        self, query: str, *, filters: QueryFilters | None = None, k: int = 5
    ) -> RetrievalResult:
        chunk = ScoredChunk(
            id=f"{DOC_A}#c0",
            document_id=DOC_A,
            text="Invoice INV-000001 total 100.00 USD",
            score=0.9,
            doc_number="INV-000001",
        )
        return RetrievalResult.build(
            query, self.name, [chunk], timings_ms={"dense": 5.0, "total": 10.0}
        )


class AnsweringProvider:
    name = "openai"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def complete(self, req: LLMRequest) -> LLMResponse:
        if self.fail:
            raise LLMProviderError("bad key", provider="openai")
        return LLMResponse(
            text="The total is 100.00 USD [doc:INV-000001].",
            provider="openai",
            model=req.model,
            usage=Usage(tokens_in=1_000, tokens_out=100),
            latency_ms=20.0,
        )

    def stream(self, req: LLMRequest) -> Any:
        raise NotImplementedError


def client(fail: bool = False) -> LLMClient:
    return LLMClient(
        MODELS,
        providers={"openai": AnsweringProvider(fail)},
        factories={},
        ledger=CostLedger(),
        sleep=lambda _: None,
    )


CONFIG = {"hash": "abc", "files": {}, "options": {}}


def test_retrieval_only_run() -> None:
    run = run_eval(QUESTIONS, FakeRetriever(), suite="ci", config=CONFIG, run_id="r1")

    assert run.llm_status == "no LLM client"
    assert [r.question_id for r in run.results] == ["q001", "q002", "q003"]
    metrics = run.metrics
    assert (metrics["questions"], metrics["answerable"], metrics["answered"]) == (3, 2, 0)
    assert metrics["recall@5"] == 0.5 and metrics["mrr"] == 0.5
    assert metrics["recall@5.lookup"] == 0.5
    assert metrics["latency.retrieval.p50_ms"] == 10.0
    assert "latency.answer.p50_ms" not in metrics and "latency.query.p50_ms" not in metrics
    assert metrics["facts_found"] is None and metrics["cost.query.mean_usd"] == 0.0


def test_answers_cost_and_ragas_when_a_model_answers() -> None:
    seen: list[Sequence[Sample]] = []

    def fake_ragas(samples: Sequence[Sample]) -> RagasOutcome:
        seen.append(samples)
        return RagasOutcome(
            summary={"all": {"n": len(samples), "faithfulness": 0.8}},
            metrics=["faithfulness"],
            judge_calls=7,
            judge_cost_usd=0.01,
        )

    llm = client()
    run = run_eval(
        QUESTIONS,
        FakeRetriever(),
        suite="ci",
        config=CONFIG,
        llm=llm,
        ragas=fake_ragas,
        run_id="r2",
    )

    assert run.llm_status == "ok"
    (samples,) = seen
    assert [s.question_id for s in samples] == ["q001", "q002"]  # not the unanswerable one
    assert samples[0].retrieved_contexts == ["Invoice INV-000001 total 100.00 USD"]
    metrics = run.metrics
    assert metrics["answered"] == 3 and metrics["faithfulness"] == 0.8
    assert metrics["facts_found"] == pytest.approx((1 + 0 + 0) / 3)
    assert metrics["citations_ok"] == 1.0  # every citation resolves to its context
    # three answers at $0.003; the one-token check call is not a query
    assert run.cost.query.n == 3 and run.cost.query.calls == 3
    assert metrics["cost.query.mean_usd"] == pytest.approx(0.003)
    assert metrics["cost.judge_usd"] == 0.01
    ids = {r.request_id for r in llm.ledger.records}
    assert query_request_id("r2", "q001") in ids and None in ids
    assert metrics["latency.query.p50_ms"] >= metrics["latency.retrieval.p50_ms"]


def test_answers_and_ragas_are_skipped_when_no_model_answers() -> None:
    def never(samples: Sequence[Sample]) -> RagasOutcome:
        raise AssertionError("RAGAS must not run without answers")

    run = run_eval(
        QUESTIONS, FakeRetriever(), suite="ci", config=CONFIG, llm=client(fail=True), ragas=never
    )

    assert run.llm_status.startswith("skipped, AllProvidersFailedError")
    assert run.metrics["answered"] == 0 and run.metrics["recall@5"] == 0.5
    assert run.ragas is None


def test_config_fingerprint_tracks_content_and_options(tmp_path: Path) -> None:
    a, b = tmp_path / "a.yaml", tmp_path / "b.md"
    a.write_bytes(b"k: 5\n")
    b.write_bytes(b"prompt\n")

    first = config_fingerprint([a, b], {"k": 5}, root=tmp_path)
    assert config_fingerprint([b, a], {"k": 5}, root=tmp_path) == first  # order-independent
    assert sorted(first["files"]) == ["a.yaml", "b.md"]

    a.write_bytes(b"k: 5\r\n")  # a Windows checkout of the same file
    assert config_fingerprint([a, b], {"k": 5}, root=tmp_path)["hash"] == first["hash"]
    assert config_fingerprint([a, b], {"k": 10}, root=tmp_path)["hash"] != first["hash"]
    a.write_bytes(b"k: 6\n")
    assert config_fingerprint([a, b], {"k": 5}, root=tmp_path)["hash"] != first["hash"]


def test_git_revision_in_and_out_of_a_repository(tmp_path: Path) -> None:
    sha, _ = git_revision()
    assert sha is not None and len(sha) == 40
    assert git_revision(tmp_path) == (None, False)


def test_eval_run_row_values() -> None:
    run = run_eval(QUESTIONS, FakeRetriever(), suite="ci", config=CONFIG, run_id="r3")
    run.git_sha = "f" * 40

    values = eval_run_values(run)

    assert values["suite"] == "ci" and values["git_sha"] == "f" * 40
    assert values["config"]["hash"] == "abc" and values["config"]["run_id"] == "r3"
    assert values["metrics"]["recall@5"] == 0.5
    assert values["id"] == eval_run_values(run)["id"]  # stable per run id
    json.dumps(values)  # JSONB-ready


# --------------------------------------------------------------------------- reports


def answered_run() -> Any:
    def fake_ragas(samples: Sequence[Sample]) -> RagasOutcome:
        return RagasOutcome(
            summary={"all": {"n": 2, "faithfulness": 0.9}}, metrics=["faithfulness"]
        )

    config = {"hash": "c" * 64, "files": {}, "options": {"variant": "R4", "k": 5}}
    return run_eval(
        QUESTIONS,
        FakeRetriever(),
        suite="ci",
        config=config,
        llm=client(),
        ragas=fake_ragas,
        git_sha="a" * 40,
        run_id="r4",
    )


def test_reports_are_written_to_the_dated_directory(tmp_path: Path) -> None:
    run = answered_run()

    paths = write_reports(run, tmp_path / "2026-11-22", documents=None, stored="row-1")

    assert sorted(p.name for p in paths.values()) == [
        "latency_cost_ci.md",
        "results_ci.jsonl",
        "summary_ci.json",
        "summary_ci.md",
    ]
    summary = paths["summary"].read_text(encoding="utf-8")
    assert summary.startswith("# Evaluation summary (ci set, ")
    assert "| recall@5 | 0.500 | below 0.90 |" in summary
    assert "| faithfulness | 0.900 | met |" in summary  # target 0.85 in configs/eval.yaml
    assert f"| git | `{'a' * 40}` |" in summary and "eval_runs row-1" in summary
    assert "- q002 (lookup): recall 0.00" in summary
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert data["config_hash"] == "c" * 64 and data["metrics"]["faithfulness"] == 0.9
    assert data["suite"] == "ci" and data["git_sha"] == "a" * 40
    latency = paths["latency_cost"].read_text(encoding="utf-8")
    assert "| query (retrieval + answer) | 3 |" in latency
    assert "| per query | 3 | 3 |" in latency
    assert "Not read: no llm_calls source" in latency
    lines = paths["results"].read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["question_id"] for line in lines] == ["q001", "q002", "q003"]


def test_full_set_reports_have_no_suffix_and_show_document_cost(tmp_path: Path) -> None:
    run = answered_run()
    run.suite = "full"
    documents = cost_report(
        [
            LLMCallRecord(
                request_id=document_request_id("d1"),
                alias="extractor",
                provider="openai",
                model="gpt-4o-mini",
                tokens_in=900,
                tokens_out=300,
                cost_usd=0.0003,
                latency_ms=800.0,
            )
        ]
    )

    paths = write_reports(run, tmp_path, documents=documents)

    assert paths["summary"].name == "summary.md" and paths["json"].name == "summary.json"
    assert "| 1 | 1 | 0.0003 | 0.000300 |" in paths["latency_cost"].read_text(encoding="utf-8")


def test_a_skipped_model_is_explained_without_its_key(tmp_path: Path) -> None:
    class LeakyProvider(AnsweringProvider):
        def complete(self, req: LLMRequest) -> LLMResponse:
            raise LLMProviderError("HTTP 401: Incorrect API key sk-proj-abc***xyz", provider="x")

    llm = LLMClient(
        MODELS, providers={"openai": LeakyProvider()}, factories={}, ledger=CostLedger()
    )
    run = run_eval(QUESTIONS, FakeRetriever(), suite="ci", config=CONFIG | {"options": {}}, llm=llm)

    summary = write_reports(run, tmp_path)["summary"].read_text(encoding="utf-8")
    assert "skipped, AllProvidersFailedError" in summary and "sk-proj" not in summary
    assert "| faithfulness | not measured |" in summary
