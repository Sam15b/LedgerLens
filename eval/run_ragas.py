"""RAGAS scores for LedgerLens answers on the golden set.

    python -m eval.run_ragas --set ci --report eval/reports/<date>/ragas.md

For each golden question the runner builds a **sample** -- the question, the retrieved
contexts (chunk texts and GraphRAG facts from the default retriever), the answer the
``smart`` model gives from them (the ``answer_grounded`` prompt, as in
``benchmark_retrieval.py``) and the ground-truth answer -- and scores it with RAGAS:

* **faithfulness** -- the share of the answer's claims the contexts support;
* **answer relevancy** -- how well questions regenerated from the answer match the asked
  question (cosine similarity of their embeddings; the local dense model);
* **context precision** -- whether the contexts that hold the reference answer are ranked
  above those that do not;
* **context recall** -- the share of the reference answer's claims the contexts support;
* **answer correctness** -- the answer against the reference: claim-level F1 (0.75)
  blended with their embedding similarity (0.25).

The last three compare with ``ground_truth_answer``, which is why unanswerable questions
(no reference) are left out.

The judge is the model router's ``judge`` alias (``--judge`` to override), through
:func:`router_judge`; its calls and cost are in the report. Samples can be saved
(``--save-samples``) and scored again later (``--samples``), so the answers are paid for
once. Judge replies are cached too (:class:`JudgeCache`, ``data/cache/ragas_judge.sqlite``;
``--no-cache`` to bypass), keyed by the judge's resolved model, the prompt and the
response schema: re-scoring the same samples costs nothing, and changing the judge model
or a RAGAS prompt misses the cache instead of reusing a stale verdict.

Unanswerable questions are not scored: they have no reference and the right answer is a
refusal, which ``benchmark_retrieval.py`` checks.

Built on RAGAS 0.4's ``ragas.metrics.collections`` API: metrics take an
``InstructorBaseRagasLLM`` judge (``agenerate(prompt, response_model)``) and a
``BaseRagasEmbedding``. Earlier releases used ``evaluate()`` with LangChain wrappers; that
path is deprecated, and the collections metrics refuse those wrappers.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import sqlite3
import threading
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

from pydantic import BaseModel

from eval.golden_set.schema import GoldenQuestion, load_golden_set
from ledgerlens.llm.types import Message

if TYPE_CHECKING:
    from ragas.embeddings.base import BaseRagasEmbedding
    from ragas.llms.base import InstructorBaseRagasLLM
    from ragas.metrics.collections.base import BaseMetric

    from ledgerlens.retrieval import RetrievalResult, Retriever

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R", bound=BaseModel)

CATEGORY_ORDER = ("lookup", "clause", "multi_hop", "aggregation", "reconciliation")
SKIPPED_CATEGORIES = ("unanswerable",)


# --------------------------------------------------------------------------- samples


@dataclass(frozen=True)
class Sample:
    """One question as RAGAS sees it."""

    question_id: str
    category: str
    user_input: str
    response: str
    retrieved_contexts: list[str]
    reference: str


def contexts_of(result: RetrievalResult) -> list[str]:
    """The texts the answer was given: retrieved chunks, then graph facts."""
    return [c.text for c in result.chunks] + [f.text for f in result.graph_facts]


Answer = Callable[[GoldenQuestion, "RetrievalResult"], str]


def collect_samples(
    questions: Sequence[GoldenQuestion],
    retriever: Retriever,
    answer: Answer,
    *,
    k: int = 5,
    on_sample: Callable[[Sample], None] | None = None,
) -> list[Sample]:
    samples = []
    for q in questions:
        if q.category in SKIPPED_CATEGORIES:
            continue
        result = retriever.retrieve(q.question, k=k)
        sample = Sample(
            question_id=q.id,
            category=q.category,
            user_input=q.question,
            response=answer(q, result),
            retrieved_contexts=contexts_of(result),
            reference=q.ground_truth_answer,
        )
        samples.append(sample)
        if on_sample:
            on_sample(sample)
    return samples


def save_samples(samples: Sequence[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(asdict(s), ensure_ascii=False) for s in samples]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_samples(path: Path) -> list[Sample]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Sample(**json.loads(line)) for line in lines if line.strip()]


# --------------------------------------------------------------------------- judge

JUDGE_ALIAS = "judge"
"""``configs/models.yaml``: one fixed model for evaluation, so scores stay comparable."""


class StructuredLLM(Protocol):
    """What the judge needs from :class:`~ledgerlens.llm.router.LLMClient`."""

    def structured(
        self,
        messages: list[Message],
        schema: type[T],
        *,
        model: str = ...,
        temperature: float = ...,
        max_tokens: int | None = ...,
    ) -> T: ...


def router_judge(llm: StructuredLLM, *, model: str = JUDGE_ALIAS) -> InstructorBaseRagasLLM:
    """The RAGAS judge on the model router: each metric prompt goes to ``model`` (the
    ``judge`` alias by default) as one user message, and the reply comes back as the
    metric's response model through :meth:`LLMClient.structured` -- native structured
    output, with one repair round if it does not validate. Fallbacks, retries, cost
    logging and the ``llm_calls`` record all come from the router, like every other call."""
    from ragas.llms.base import InstructorBaseRagasLLM

    class RouterJudge(InstructorBaseRagasLLM):
        def __init__(self) -> None:
            self.model = model

        def generate(self, prompt: str, response_model: type[R]) -> R:
            return llm.structured([Message.user(prompt)], response_model, model=self.model)

        async def agenerate(self, prompt: str, response_model: type[R]) -> R:
            return await asyncio.to_thread(self.generate, prompt, response_model)

    return RouterJudge()


# --------------------------------------------------------------------------- cache

DEFAULT_CACHE = Path("data/cache/ragas_judge.sqlite")
CACHE_VERSION = 1


def judge_key(model: str, prompt: str, response_model: type[BaseModel]) -> str:
    """SHA-256 of everything that decides a judge reply: the model(s) behind the judge, the
    prompt and the schema asked for. Another judge model, a changed RAGAS prompt or a new
    response model each miss the cache instead of reusing a stale verdict."""
    payload = json.dumps(
        {
            "v": CACHE_VERSION,
            "model": model,
            "prompt": prompt,
            "schema": response_model.model_json_schema(),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JudgeCache:
    """Judge replies by :func:`judge_key`, in SQLite (safe across threads and runs)."""

    def __init__(self, path: Path | str = ":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS judge (key TEXT PRIMARY KEY, reply TEXT)")
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT reply FROM judge WHERE key = ?", (key,)).fetchone()
            if row is None:
                self.misses += 1
                return None
            self.hits += 1
            return str(row[0])

    def put(self, key: str, reply: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO judge VALUES (?, ?)", (key, reply))
            self._db.commit()

    def __len__(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT count(*) FROM judge").fetchone()[0])

    def close(self) -> None:
        self._db.close()


def cached_judge(
    judge: InstructorBaseRagasLLM, cache: JudgeCache, *, model: str
) -> InstructorBaseRagasLLM:
    """``judge`` with its replies cached under ``model`` (pass the resolved model refs, not
    just the alias, so re-pointing the alias invalidates them). Only replies that validate
    are stored; failures are retried on the next run."""
    from ragas.llms.base import InstructorBaseRagasLLM

    class CachedJudge(InstructorBaseRagasLLM):
        def generate(self, prompt: str, response_model: type[R]) -> R:
            key = judge_key(model, prompt, response_model)
            if (hit := cache.get(key)) is not None:
                return response_model.model_validate_json(hit)
            reply: R = cast("R", judge.generate(prompt, response_model))
            cache.put(key, reply.model_dump_json())
            return reply

        async def agenerate(self, prompt: str, response_model: type[R]) -> R:
            key = judge_key(model, prompt, response_model)
            if (hit := cache.get(key)) is not None:
                return response_model.model_validate_json(hit)
            reply: R = cast("R", await judge.agenerate(prompt, response_model))
            cache.put(key, reply.model_dump_json())
            return reply

    return CachedJudge()


# --------------------------------------------------------------------------- embeddings


class TextEncoder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


def dense_embeddings(encoder: TextEncoder | None = None) -> BaseRagasEmbedding:
    """RAGAS embeddings on the local dense model (``embedding.dense_model``), so answer
    relevancy needs no embedding API. Texts are embedded as documents (no query
    instruction): both sides of the comparison are questions."""
    from ragas.embeddings.base import BaseRagasEmbedding

    from ledgerlens.ingestion.embedding import DenseEmbedder

    model: TextEncoder = encoder or DenseEmbedder()

    class DenseRagasEmbedding(BaseRagasEmbedding):
        def embed_text(self, text: str, **kwargs: Any) -> list[float]:
            return [float(x) for x in model.embed_documents([text])[0]]

        async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
            return await asyncio.to_thread(self.embed_text, text)

        def embed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            return [[float(x) for x in v] for v in model.embed_documents(texts)]

    return DenseRagasEmbedding()


# --------------------------------------------------------------------------- metrics


@dataclass(frozen=True)
class MetricSpec:
    """A metric and the sample fields it reads."""

    metric: BaseMetric
    """A ``ragas.metrics.collections`` metric; ``ascore`` takes the named ``fields``."""
    fields: tuple[str, ...]


def build_metrics(
    llm: InstructorBaseRagasLLM, embeddings: BaseRagasEmbedding
) -> dict[str, MetricSpec]:
    from ragas.metrics.collections import (
        AnswerCorrectness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    return {
        "faithfulness": MetricSpec(
            Faithfulness(llm=llm), ("user_input", "response", "retrieved_contexts")
        ),
        "answer_relevancy": MetricSpec(
            AnswerRelevancy(llm=llm, embeddings=embeddings), ("user_input", "response")
        ),
        "context_precision": MetricSpec(
            ContextPrecision(llm=llm), ("user_input", "reference", "retrieved_contexts")
        ),
        "context_recall": MetricSpec(
            ContextRecall(llm=llm), ("user_input", "retrieved_contexts", "reference")
        ),
        "answer_correctness": MetricSpec(
            AnswerCorrectness(llm=llm, embeddings=embeddings),
            ("user_input", "response", "reference"),
        ),
    }


@dataclass
class Scored:
    sample: Sample
    scores: dict[str, float] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


async def score_sample(sample: Sample, metrics: Mapping[str, MetricSpec]) -> Scored:
    scored = Scored(sample)
    for name, spec in metrics.items():
        kwargs = {f: getattr(sample, f) for f in spec.fields}
        if "retrieved_contexts" in kwargs and not kwargs["retrieved_contexts"]:
            scored.errors[name] = "no contexts retrieved"
            continue
        try:
            result = await spec.metric.ascore(**kwargs)
        except Exception as exc:  # one failed judge call must not sink the run
            scored.errors[name] = f"{type(exc).__name__}: {exc}"
            continue
        value = float(getattr(result, "value", result))
        if math.isnan(value):
            scored.errors[name] = "no score (nothing to judge)"
        else:
            scored.scores[name] = value
    return scored


async def score_samples(
    samples: Sequence[Sample],
    metrics: Mapping[str, MetricSpec],
    *,
    concurrency: int = 4,
    on_scored: Callable[[Scored], None] | None = None,
) -> list[Scored]:
    gate = asyncio.Semaphore(concurrency)

    async def one(sample: Sample) -> Scored:
        async with gate:
            scored = await score_sample(sample, metrics)
        if on_scored:
            on_scored(scored)
        return scored

    return list(await asyncio.gather(*(one(s) for s in samples)))


# --------------------------------------------------------------------------- report

_KEYLIKE = re.compile(r"\b(sk|pk|AIza|AKIA)[-_A-Za-z0-9*]{6,}")


def clean_error(text: str, limit: int = 200) -> str:
    """A judge error fit for a committed report: key-like strings redacted (providers echo
    a masked key), one line, shortened."""
    flat = " ".join(_KEYLIKE.sub("<redacted>", text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def summarize(results: Sequence[Scored], metrics: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Mean (and how many samples scored) per metric, overall and per category."""
    groups: dict[str, list[Scored]] = defaultdict(list)
    for r in results:
        groups["all"].append(r)
        groups[r.sample.category].append(r)
    out: dict[str, dict[str, Any]] = {}
    for group, items in groups.items():
        row: dict[str, Any] = {"n": len(items)}
        for metric in metrics:
            values = [r.scores[metric] for r in items if metric in r.scores]
            row[metric] = fmean(values) if values else None
            row[f"{metric}_n"] = len(values)
        out[group] = row
    return out


def render_report(
    results: Sequence[Scored],
    metrics: Sequence[str],
    *,
    title: str = "RAGAS",
    notes: Sequence[str] = (),
) -> str:
    summary = summarize(results, metrics)
    lines = [f"# {title}", ""]
    lines += [f"- {note}" for note in notes]
    if notes:
        lines.append("")
    header = ["Group", "n", *metrics]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|")
    for group in ("all", *CATEGORY_ORDER):
        if group not in summary:
            continue
        row = summary[group]
        cells = ["—" if row[m] is None else f"{row[m]:.3f} ({row[f'{m}_n']})" for m in metrics]
        lines.append(f"| {group} | {row['n']} | " + " | ".join(cells) + " |")
    failures = [(r.sample.question_id, m, e) for r in results for m, e in r.errors.items()]
    if failures:
        lines += ["", f"## Not scored ({len(failures)})", ""]
        lines += [f"- {qid} {metric}: {clean_error(error)}" for qid, metric, error in failures[:50]]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- CLI


def select(questions: Sequence[GoldenQuestion], ids: str | None) -> list[GoldenQuestion]:
    if not ids:
        return list(questions)
    wanted = {i.strip() for i in ids.split(",")}
    return [q for q in questions if q.id in wanted]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAGAS scores on the golden set.")
    parser.add_argument("--set", choices=("ci", "full"), default="ci")
    parser.add_argument("--ids", help="only these question ids (comma-separated)")
    parser.add_argument("--variant", default="R4", help="retriever variant (R1..R5)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--samples", type=Path, help="score saved samples instead")
    parser.add_argument("--save-samples", type=Path, help="write the samples built here")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--judge", default=JUDGE_ALIAS, help="model alias or provider/model (default: judge)"
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="judge reply cache")
    parser.add_argument("--no-cache", action="store_true", help="always call the judge")
    return parser


def golden_path(which: str) -> Path:
    from ledgerlens.core.config import get_config, repo_path

    config = get_config().eval
    return repo_path(config.ci_subset if which == "ci" else config.golden_set)


def live_samples(args: argparse.Namespace, questions: Sequence[GoldenQuestion]) -> list[Sample]:
    from eval.benchmark_retrieval import build_retrievers, llm_answerer
    from ledgerlens.llm.router import LLMClient

    llm = LLMClient()
    retriever = build_retrievers([args.variant], llm=llm)[args.variant]
    answerer = llm_answerer(llm)
    return collect_samples(
        questions,
        retriever,
        lambda q, result: answerer(q, result).text,
        k=args.k,
        on_sample=lambda s: print(f"answered {s.question_id}", flush=True),
    )


def main(argv: Sequence[str] | None = None) -> int:
    from eval.benchmark_retrieval import CostMeter, metered
    from ledgerlens.llm.router import LLMClient

    args = build_parser().parse_args(argv)
    if args.samples:
        samples = load_samples(args.samples)
    else:
        questions = select(load_golden_set(golden_path(args.set)), args.ids)
        samples = live_samples(args, questions)
    if args.save_samples:
        save_samples(samples, args.save_samples)
    meter = CostMeter()
    llm = metered(LLMClient(), meter)
    judge = router_judge(llm, model=args.judge)
    cache = None if args.no_cache else JudgeCache(args.cache)
    if cache is not None:
        resolved = ",".join(llm.config.resolve(args.judge))
        judge = cached_judge(judge, cache, model=resolved)
    metrics = build_metrics(judge, dense_embeddings())
    results = asyncio.run(score_samples(samples, metrics, concurrency=args.concurrency))
    calls, cost = meter.take()
    cached = f"; cache {cache.hits} hits, {cache.misses} misses" if cache else "; no cache"
    notes = [
        f"judge: `{args.judge}` ({calls} calls, ${cost:.4f}{cached})",
        f"samples: {len(samples)} ({args.samples or args.set + ' set, ' + args.variant})",
    ]
    report = render_report(results, list(metrics), title=f"RAGAS ({args.set} set)", notes=notes)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
