"""The evaluation suite in one command: retrieval, answers, RAGAS, latency and cost.

    python -m eval.runner --set full          # make eval
    python -m eval.runner --set ci            # make eval-ci: the 40-question subset

For every golden question the default retriever (``retrieval.default``, R4) retrieves
with the query analyzer's filters; if a model is reachable, the ``smart`` alias answers
from the cited context, and RAGAS scores the answers with the ``judge`` alias
(``eval/run_ragas.py``). Each question runs under the request id
``eval:<run id>:<question id>``, so its LLM calls -- Text2Cypher, the answer -- are its
cost (``eval/latency_cost.py``); the judge's calls are counted apart.

Before any paid work a one-token call checks the answer model. If no provider answers (no
key, Ollama down), answers and RAGAS are skipped and the run says why: retrieval metrics
and latency still come out, rather than 200 failed questions.

Every run is stored in ``eval_runs`` with what makes it reproducible:

* ``git_sha`` -- the commit evaluated (``config.git_dirty`` if the tree had changes);
* ``config.hash`` -- a SHA-256 over every file that shapes the answers
  (``configs/*.yaml``, the agent prompts, the golden set) and the run options, with each
  file's own digest in ``config.files``; two runs with the same hash ran the same setup.

``--no-db`` skips storing (no Postgres); a failed insert is reported, not fatal.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import subprocess
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from eval.benchmark_retrieval import (
    Answerer,
    CostMeter,
    FilterSource,
    QuestionResult,
    llm_answerer,
    mean,
    run_benchmark,
)
from eval.golden_set.schema import GoldenQuestion, load_golden_set
from eval.latency_cost import CostReport, cost_report, latency_summary
from eval.run_ragas import SKIPPED_CATEGORIES, Sample, contexts_of
from ledgerlens.core.config import get_config, repo_path
from ledgerlens.core.logging import request_context
from ledgerlens.llm.types import Message
from ledgerlens.observability.cost import LLMCallRecord

if TYPE_CHECKING:
    from sqlalchemy.orm import Session, sessionmaker

    from ledgerlens.llm.router import LLMClient
    from ledgerlens.retrieval import RetrievalResult, Retriever

logger = logging.getLogger(__name__)

REPO = repo_path(".")
SUITES = ("full", "ci")
CONFIG_GLOBS = ("configs/*.yaml", "src/ledgerlens/agents/prompts/*.md")
"""Files whose content changes what the system answers: part of the config hash."""

RagasScorer = Callable[[Sequence[Sample]], "RagasOutcome"]


@dataclass
class RagasOutcome:
    """RAGAS means per metric (``eval.run_ragas.summarize``: ``all`` and per category)."""

    summary: dict[str, dict[str, Any]]
    metrics: list[str]
    judge_calls: int = 0
    judge_cost_usd: float = 0.0
    not_scored: int = 0


@dataclass
class EvalRun:
    run_id: str
    suite: str
    started_at: datetime
    git_sha: str | None
    config: dict[str, Any]
    results: list[QuestionResult]
    cost: CostReport
    llm_status: str
    """``ok``, or why answers and RAGAS were skipped."""
    ragas: RagasOutcome | None = None
    finished_at: datetime | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def config_hash(self) -> str:
        return str(self.config["hash"])


# --------------------------------------------------------------------------- provenance


def git_revision(repo: Path = REPO) -> tuple[str | None, bool]:
    """The checked-out commit and whether tracked files have changes (``None`` outside git)."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, False
    return sha or None, bool(status.strip())


def file_digest(path: Path) -> str:
    """SHA-256 of a file with line endings normalised (a Windows checkout hashes the same)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def config_fingerprint(
    files: Iterable[Path], options: Mapping[str, Any], *, root: Path = REPO
) -> dict[str, Any]:
    """``{"hash", "files", "options"}``: the hash covers every file's digest and the options."""
    digests = {
        path.resolve().relative_to(root.resolve()).as_posix(): file_digest(path)
        for path in sorted(files)
    }
    payload = json.dumps({"files": digests, "options": options}, sort_keys=True, default=str)
    return {
        "hash": hashlib.sha256(payload.encode()).hexdigest(),
        "files": {name: digest[:16] for name, digest in sorted(digests.items())},
        "options": dict(options),
    }


def config_files(golden: Path, root: Path = REPO) -> list[Path]:
    files = [path for pattern in CONFIG_GLOBS for path in root.glob(pattern)]
    return [*files, golden]


def golden_path(suite: str) -> Path:
    config = get_config().eval
    return repo_path(config.ci_subset if suite == "ci" else config.golden_set)


# --------------------------------------------------------------------------- the run


def query_request_id(run_id: str, question_id: str) -> str:
    return f"eval:{run_id}:{question_id}"


def check_llm(llm: LLMClient, model: str) -> str:
    """``ok`` if ``model`` answers a one-token request, else why not. The call has no
    request id, so it is not counted as a query."""
    try:
        llm.complete([Message.user("Reply with OK.")], model=model, max_tokens=5)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return "ok"


def run_eval(
    questions: Sequence[GoldenQuestion],
    retriever: Retriever,
    *,
    suite: str,
    config: dict[str, Any],
    variant: str = "R4",
    k: int = 5,
    filters: FilterSource | None = None,
    llm: LLMClient | None = None,
    answer_model: str = "smart",
    ragas: RagasScorer | None = None,
    git_sha: str | None = None,
    run_id: str | None = None,
    on_result: Callable[[QuestionResult], None] | None = None,
) -> EvalRun:
    """Retrieve (and, with a working ``llm``, answer) every question, then score the
    answers with ``ragas``. Nothing here needs a store: the caller brings the retriever."""
    run_id = run_id or uuid.uuid4().hex[:12]
    started = datetime.now(UTC)
    llm_status = "no LLM client" if llm is None else check_llm(llm, answer_model)
    answerer: Answerer | None = None
    meter: CostMeter | None = None
    if llm is not None and llm_status == "ok":
        answerer = llm_answerer(llm, model=answer_model)
        meter = CostMeter()
        _meter_calls(llm, meter)
    first_record = len(llm.ledger.records) if llm is not None else 0
    samples: list[Sample] = []

    def keep(scored: QuestionResult, result: RetrievalResult | None) -> None:
        question = by_id[scored.question_id]
        scorable = question.category not in SKIPPED_CATEGORIES  # no reference to judge by
        if result is not None and scored.answer is not None and scorable:
            samples.append(
                Sample(
                    question_id=question.id,
                    category=question.category,
                    user_input=question.question,
                    response=scored.answer,
                    retrieved_contexts=contexts_of(result),
                    reference=question.ground_truth_answer,
                )
            )
        if on_result is not None:
            on_result(scored)

    by_id = {q.id: q for q in questions}
    results: list[QuestionResult] = []
    for question in questions:
        with request_context(query_request_id(run_id, question.id)):
            results += run_benchmark(
                [question],
                {variant: retriever},
                k=k,
                filters=filters,
                answerer=answerer,
                meter=meter,
                on_result=keep,
            )
    records: list[LLMCallRecord] = llm.ledger.records[first_record:] if llm is not None else []
    prefix = query_request_id(run_id, "")
    cost = cost_report(r for r in records if (r.request_id or "").startswith(prefix))
    outcome = None
    if ragas is not None and samples:
        outcome = ragas(samples)
    elif ragas is not None:
        llm_status = llm_status if llm_status != "ok" else "ok (no answers to score)"
    run = EvalRun(
        run_id=run_id,
        suite=suite,
        started_at=started,
        git_sha=git_sha,
        config=config,
        results=results,
        cost=cost,
        llm_status=llm_status,
        ragas=outcome,
        finished_at=datetime.now(UTC),
    )
    run.metrics = run_metrics(run, k=k)
    return run


def _meter_calls(llm: LLMClient, meter: CostMeter) -> None:
    from eval.benchmark_retrieval import metered

    metered(llm, meter)


def _rate(values: Sequence[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def run_metrics(run: EvalRun, *, k: int) -> dict[str, Any]:
    """The run's headline numbers, flat (``recall@5``, ``faithfulness``,
    ``latency.query.p95_ms``, ``cost.query.mean_usd``, ...); scores are 0-1, ``None`` when
    not measured."""
    results = run.results
    answerable = [r for r in results if r.recall is not None]
    answered = [r for r in results if r.answer is not None]
    metrics: dict[str, Any] = {
        "questions": len(results),
        "answerable": len(answerable),
        "errors": sum(r.error is not None for r in results),
        f"recall@{k}": mean(r.recall or 0.0 for r in answerable) if answerable else None,
        "mrr": mean(r.reciprocal_rank or 0.0 for r in answerable) if answerable else None,
        "evidence_recall": mean(r.evidence_recall or 0.0 for r in answerable)
        if answerable
        else None,
        "answered": len(answered),
        "facts_found": _mean_of(r.facts_found for r in answered),
        "citations_ok": _rate([bool(r.citations_ok) for r in answered]),
    }
    for category in sorted({r.category for r in answerable}):
        group = [r for r in answerable if r.category == category]
        metrics[f"recall@{k}.{category}"] = mean(r.recall or 0.0 for r in group)
    ragas_all = run.ragas.summary.get("all", {}) if run.ragas else {}
    for name in run.ragas.metrics if run.ragas else ():
        metrics[name] = ragas_all.get(name)
    latencies = {
        "retrieval": latency_summary(r.latency_ms for r in results),
        "answer": latency_summary(r.answer_latency_ms for r in answered),
        "query": latency_summary(
            r.latency_ms + (r.answer_latency_ms or 0.0) for r in (answered or results)
        ),
    }
    for name, summary in latencies.items():
        if summary.n:
            metrics[f"latency.{name}.p50_ms"] = round(summary.p50_ms, 1)
            metrics[f"latency.{name}.p95_ms"] = round(summary.p95_ms, 1)
    metrics["cost.query.mean_usd"] = run.cost.query.mean_usd
    metrics["cost.query.p95_usd"] = run.cost.query.p95_usd
    metrics["cost.total_usd"] = run.cost.query.total_usd
    if run.ragas:
        metrics["cost.judge_usd"] = round(run.ragas.judge_cost_usd, 6)
    return metrics


def _mean_of(values: Iterable[float | None]) -> float | None:
    items = [v for v in values if v is not None]
    return mean(items) if items else None


# --------------------------------------------------------------------------- RAGAS


def ragas_scorer(
    llm: LLMClient, *, judge: str = "judge", cache: Path | None = None, concurrency: int = 4
) -> RagasScorer:
    """RAGAS through the router's ``judge`` alias, with the judge reply cache."""

    def score(samples: Sequence[Sample]) -> RagasOutcome:
        from eval.benchmark_retrieval import metered
        from eval.run_ragas import (
            JudgeCache,
            build_metrics,
            cached_judge,
            dense_embeddings,
            router_judge,
            score_samples,
            summarize,
        )

        meter = CostMeter()
        judged = router_judge(metered(llm, meter), model=judge)
        if cache is not None:
            resolved = ",".join(llm.config.resolve(judge))
            judged = cached_judge(judged, JudgeCache(cache), model=resolved)
        metrics = build_metrics(judged, dense_embeddings())
        scored = asyncio.run(score_samples(samples, metrics, concurrency=concurrency))
        calls, cost = meter.take()
        return RagasOutcome(
            summary=summarize(scored, list(metrics)),
            metrics=list(metrics),
            judge_calls=calls,
            judge_cost_usd=cost,
            not_scored=sum(len(s.errors) for s in scored),
        )

    return score


# --------------------------------------------------------------------------- eval_runs


def eval_run_values(run: EvalRun) -> dict[str, Any]:
    """Column values for the run's ``eval_runs`` row."""
    return {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"ledgerlens-eval:{run.run_id}")),
        "suite": run.suite,
        "git_sha": run.git_sha,
        "config": run.config
        | {
            "run_id": run.run_id,
            "llm_status": run.llm_status,
            "started_at": run.started_at.isoformat(),
            "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        },
        "metrics": run.metrics,
    }


def store_run(run: EvalRun, sessions: sessionmaker[Session] | None = None) -> str:
    """Insert the run into ``eval_runs``; returns the row id."""
    from ledgerlens.storage.postgres.db import get_session_factory, session_scope
    from ledgerlens.storage.postgres.models import EvalRunRow

    values = eval_run_values(run)
    with session_scope(sessions or get_session_factory()) as session:
        session.add(EvalRunRow(**values))
    return str(values["id"])


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.runner", description="Run the evaluation suite."
    )
    parser.add_argument("--set", dest="suite", choices=SUITES, default="ci")
    parser.add_argument("--variant", default=None, help="retriever (default: retrieval.default)")
    parser.add_argument("--k", type=int, default=None, help="default: eval.k")
    parser.add_argument("--limit", type=int, default=None, help="first N questions only")
    parser.add_argument("--filters", choices=("analyzer", "none"), default="analyzer")
    parser.add_argument(
        "--hybrid",
        choices=("auto", "exact", "index"),
        default="auto",
        help="exact: score hybrid search client-side (Pinecone Local ignores sparse "
        "vectors); auto: exact when PINECONE_HOST is local",
    )
    parser.add_argument("--no-answers", action="store_true", help="retrieval only (no LLM)")
    parser.add_argument("--no-ragas", action="store_true", help="answers but no RAGAS judge")
    parser.add_argument("--answer-model", default="smart")
    parser.add_argument("--judge", default="judge")
    parser.add_argument("--no-db", action="store_true", help="do not store the run")
    return parser


def exact_hybrid(choice: str) -> bool:
    from ledgerlens.core.config import get_settings

    if choice != "auto":
        return choice == "exact"
    host = get_settings().pinecone_host
    return "localhost" in host or "127.0.0.1" in host


def main(argv: Sequence[str] | None = None) -> int:
    from eval.benchmark_retrieval import analyzer_filters, build_retrievers, warm_up
    from eval.run_ragas import DEFAULT_CACHE
    from ledgerlens.llm.router import LLMClient

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    app = get_config()
    variant = (args.variant or app.retrieval.default).upper()
    k = args.k or app.eval.k
    golden = golden_path(args.suite)
    questions = load_golden_set(golden)[: args.limit]
    hybrid = exact_hybrid(args.hybrid)
    options = {
        "suite": args.suite,
        "questions": len(questions),
        "variant": variant,
        "k": k,
        "filters": args.filters,
        "exact_hybrid": hybrid,
        "answers": not args.no_answers,
        "answer_model": args.answer_model,
        "ragas": not (args.no_answers or args.no_ragas),
        "judge": args.judge,
    }
    sha, dirty = git_revision()
    config = config_fingerprint(config_files(golden), options) | {"git_dirty": dirty}
    llm = LLMClient()
    retriever = build_retrievers([variant], llm=llm, exact_hybrid=hybrid)[variant]
    warm_up({variant: retriever})
    filters = analyzer_filters() if args.filters == "analyzer" else None
    ragas = (
        None if not options["ragas"] else ragas_scorer(llm, judge=args.judge, cache=DEFAULT_CACHE)
    )

    def progress(result: QuestionResult) -> None:
        recall = "  -  " if result.recall is None else f"{result.recall:.2f}"
        print(f"{result.question_id} recall@{k}={recall} {result.latency_ms:7.0f} ms", flush=True)

    run = run_eval(
        questions,
        retriever,
        suite=args.suite,
        config=config,
        variant=variant,
        k=k,
        filters=filters,
        llm=None if args.no_answers else llm,
        answer_model=args.answer_model,
        ragas=ragas,
        git_sha=sha,
        on_result=progress,
    )
    if not args.no_db:
        try:
            print(f"stored eval_runs {store_run(run)}")
        except Exception as exc:
            print(f"not stored in eval_runs: {type(exc).__name__}: {exc}")
    print(json.dumps(run.metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
