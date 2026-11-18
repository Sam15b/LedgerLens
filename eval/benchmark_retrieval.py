"""Benchmark retriever variants on the golden set: recall@k, MRR, latency, cost.

Every retriever returns chunks (best first) and, for R4/R5, graph facts. The documents
it *retrieved* are read off that result in rank order -- the chunks' documents, then the
documents the facts cite -- each once (:func:`retrieved_documents`). They are scored
against the question's ``relevant_document_ids``:

* **recall@k** -- the share of relevant documents among the first ``k`` retrieved;
* **MRR** -- 1 / the rank of the first relevant document (0 if none is retrieved);
* **evidence recall** -- the share of relevant documents anywhere in the result, facts
  included: what the answer step gets to see.

Unanswerable questions (no relevant documents) have no recall; they are counted, not
averaged.

    python -m eval.benchmark_retrieval --variants R1,R2,R3,R4,R5 --k 5 \
        --report eval/reports/<date>/retrieval.md

All variants see each question with the same ``k`` and, by default, the same metadata
filters from the query analyzer (``--filters none`` for raw vector search), so the
comparison isolates what each stage adds.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from eval.golden_set.schema import DEFAULT_PATH as DEFAULT_GOLDEN
from eval.golden_set.schema import GoldenQuestion, load_golden_set
from ledgerlens.extraction.normalize import find_dates
from ledgerlens.llm.router import LLMClient
from ledgerlens.llm.types import LLMResponse
from ledgerlens.retrieval import QueryFilters, RetrievalResult, Retriever
from ledgerlens.retrieval.context import answer_messages, assemble_context

logger = logging.getLogger(__name__)


def unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def retrieved_documents(result: RetrievalResult) -> list[str]:
    """Documents in rank order: chunk documents, then the documents graph facts cite."""
    chunk_documents = (chunk.document_id for chunk in result.chunks)
    fact_documents = (doc for fact in result.graph_facts for doc in fact.document_ids)
    return unique([*chunk_documents, *fact_documents])


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Share of ``relevant`` among the first ``k`` of ``retrieved`` (1.0 when nothing is
    relevant: there was nothing to miss)."""
    if k < 1:
        raise ValueError("k must be at least 1")
    wanted = set(relevant)
    if not wanted:
        return 1.0
    return len(wanted & set(unique(retrieved)[:k])) / len(wanted)


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """1 / rank of the first relevant document in ``retrieved`` (1-based), else 0."""
    wanted = set(relevant)
    for rank, document in enumerate(unique(retrieved), start=1):
        if document in wanted:
            return 1.0 / rank
    return 0.0


def evidence_recall(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Share of ``relevant`` anywhere in ``retrieved``."""
    wanted = list(relevant)
    return recall_at_k(retrieved, wanted, max(len(retrieved), 1))


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def percentile(values: Iterable[float], pct: float) -> float:
    """The ``pct`` percentile (0-100) by linear interpolation; 0.0 for no values."""
    items = sorted(values)
    if not items:
        return 0.0
    position = (len(items) - 1) * pct / 100
    low = int(position)
    high = min(low + 1, len(items) - 1)
    return items[low] + (items[high] - items[low]) * (position - low)


# --------------------------------------------------------------------------- runner

VARIANTS: dict[str, str] = {
    "R1": "dense",
    "R2": "hybrid (dense + BM25)",
    "R3": "hybrid + cross-encoder rerank",
    "R4": "GraphRAG (R3 seeds + graph templates)",
    "R5": "GraphRAG + Text2Cypher",
}


@dataclass
class QuestionResult:
    """One variant on one golden question."""

    variant: str
    question_id: str
    category: str
    requires_graph: bool
    retrieved: list[str]
    relevant: list[str]
    recall: float | None
    """recall@k; ``None`` for a question with nothing relevant (unanswerable)."""
    reciprocal_rank: float | None
    evidence_recall: float | None
    latency_ms: float
    timings_ms: dict[str, float] = field(default_factory=dict)
    chunks: int = 0
    facts: int = 0
    error: str | None = None
    """The retriever raised: the question scores 0 and the error is kept."""
    llm_calls: int = 0
    """LLM calls made for this question (Text2Cypher in R5, plus the answer)."""
    cost_usd: float = 0.0
    """What those calls cost (0 for calls the price table does not know)."""
    answer: str | None = None
    answer_latency_ms: float | None = None
    citations_ok: bool | None = None
    """Every citation in the answer resolves to a document in its context (and it has one)."""
    unknown_citations: list[str] = field(default_factory=list)
    facts_found: float | None = None
    """Share of the golden ``expected`` facts the answer states."""

    @property
    def degraded(self) -> list[str]:
        """Stages that failed and fell back (``graph``, ``text2cypher``)."""
        return [key.removesuffix("_error") for key in self.timings_ms if key.endswith("_error")]


def score(
    variant: str,
    question: GoldenQuestion,
    result: RetrievalResult | None,
    *,
    k: int,
    latency_ms: float,
    error: str | None = None,
) -> QuestionResult:
    retrieved = retrieved_documents(result) if result is not None else []
    relevant = list(question.relevant_document_ids)
    answerable = bool(relevant)
    return QuestionResult(
        variant=variant,
        question_id=question.id,
        category=question.category,
        requires_graph=question.requires_graph,
        retrieved=retrieved,
        relevant=relevant,
        recall=recall_at_k(retrieved, relevant, k) if answerable else None,
        reciprocal_rank=reciprocal_rank(retrieved, relevant) if answerable else None,
        evidence_recall=evidence_recall(retrieved, relevant) if answerable else None,
        latency_ms=latency_ms,
        timings_ms=dict(result.timings_ms) if result is not None else {},
        chunks=len(result.chunks) if result is not None else 0,
        facts=len(result.graph_facts) if result is not None else 0,
        error=error,
    )


FilterSource = Callable[[str], QueryFilters | None]


class CostMeter:
    """Counts LLM calls and their cost through an :class:`LLMClient` (wrap the one the
    retrievers and the answer step share): :meth:`take` returns and resets the tally."""

    def __init__(self) -> None:
        self.calls = 0
        self.cost_usd = 0.0

    def record(self, response: LLMResponse) -> None:
        self.calls += 1
        self.cost_usd += response.cost_usd or 0.0

    def take(self) -> tuple[int, float]:
        tally = (self.calls, self.cost_usd)
        self.calls, self.cost_usd = 0, 0.0
        return tally


def metered(llm: LLMClient, meter: CostMeter) -> LLMClient:
    """``llm`` with every completed call recorded in ``meter`` (structured calls too: they
    go through ``complete``)."""
    complete = llm.complete

    def recording(*args: Any, **kwargs: Any) -> LLMResponse:
        response = complete(*args, **kwargs)
        meter.record(response)
        return response

    llm.complete = recording  # type: ignore[method-assign]
    return llm


_NEGATIVE = re.compile(
    r"\b(?:no|not|none|cannot|can't|doesn't|does not|isn't|is not|missing|unknown|"
    r"no record|not found|no evidence)\b",
    re.IGNORECASE,
)


def _mentions_amount(answer: str, amount: str) -> bool:
    plain = amount.replace(",", "")
    return amount in answer or plain in answer.replace(",", "")


def expected_facts_found(answer: str, expected: Mapping[str, Any]) -> float | None:
    """Share of the golden ``expected`` facts that ``answer`` states: amounts (with or
    without separators), document numbers, dates (any printed format), counts, day counts,
    vendor names, a yes/no verdict, and "not in the records" for unanswerable questions.
    ``None`` when there is nothing checkable."""
    checks: list[bool] = []
    answer_dates = set(find_dates(answer))
    for key, value in expected.items():
        if key == "amount":
            checks.append(_mentions_amount(answer, str(value)))
        elif key in ("number", "po_number", "vendor", "currency"):
            checks.append(str(value).lower() in answer.lower())
        elif key == "numbers" or key == "items":
            checks += [str(item).lower() in answer.lower() for item in value]
        elif key == "date":
            checks.append(date.fromisoformat(str(value)) in answer_dates or value in answer)
        elif key in ("count", "days"):
            checks.append(re.search(rf"\b{int(value)}\b", answer) is not None)
        elif key in ("match", "duplicate"):
            verdict = answer.strip().lower()
            checks.append(verdict.startswith("yes") if value else verdict.startswith("no"))
        elif key == "answerable" and value is False:
            checks.append(_NEGATIVE.search(answer) is not None)
    return mean(1.0 if ok else 0.0 for ok in checks) if checks else None


Answerer = Callable[[GoldenQuestion, RetrievalResult], "AnswerOutcome"]


@dataclass(frozen=True)
class AnswerOutcome:
    text: str
    latency_ms: float
    citations_ok: bool
    unknown_citations: list[str]


def llm_answerer(llm: LLMClient, *, model: str = "smart") -> Answerer:
    """Answers from the assembled, cited context with the ``answer_grounded`` prompt."""

    def answer(question: GoldenQuestion, result: RetrievalResult) -> AnswerOutcome:
        context = assemble_context(result)
        start = time.perf_counter()
        response = llm.complete(answer_messages(question.question, context), model=model)
        latency = (time.perf_counter() - start) * 1000
        check = context.resolve(response.text)
        unanswerable = question.category == "unanswerable"
        return AnswerOutcome(
            text=response.text,
            latency_ms=latency,
            citations_ok=(not check.unknown) if unanswerable else check.ok,
            unknown_citations=check.unknown,
        )

    return answer


def attach_answer(scored: QuestionResult, question: GoldenQuestion, outcome: AnswerOutcome) -> None:
    scored.answer = outcome.text
    scored.answer_latency_ms = outcome.latency_ms
    scored.citations_ok = outcome.citations_ok
    scored.unknown_citations = outcome.unknown_citations
    scored.facts_found = expected_facts_found(outcome.text, question.expected)


def run_benchmark(
    questions: Sequence[GoldenQuestion],
    retrievers: Mapping[str, Retriever],
    *,
    k: int = 5,
    filters: FilterSource | None = None,
    on_result: Callable[[QuestionResult, RetrievalResult | None], None] | None = None,
    answerer: Answerer | None = None,
    meter: CostMeter | None = None,
) -> list[QuestionResult]:
    """Every variant on every question, on the same footing: the same ``k`` and the same
    filters (``filters(question)``, e.g. the query analyzer's; ``None``: no filters, and
    R4/R5 analyze the question themselves). A retriever that raises scores 0 on that
    question instead of stopping the run. With an ``answerer``, every retrieval is also
    answered; ``meter`` attributes LLM calls and cost to each question and variant."""
    results: list[QuestionResult] = []
    for question in questions:
        question_filters = filters(question.question) if filters is not None else None
        for variant, retriever in retrievers.items():
            if meter is not None:
                meter.take()
            start = time.perf_counter()
            try:
                result: RetrievalResult | None = retriever.retrieve(
                    question.question, filters=question_filters, k=k
                )
                error = None
            except Exception as exc:  # one broken question must not end the benchmark
                logger.warning("%s failed on %s", variant, question.id, exc_info=True)
                result, error = None, f"{type(exc).__name__}: {exc}"
            latency = (time.perf_counter() - start) * 1000
            if result is not None:
                latency = result.timings_ms.get("total", latency)
            scored = score(variant, question, result, k=k, latency_ms=latency, error=error)
            if answerer is not None and result is not None:
                try:
                    attach_answer(scored, question, answerer(question, result))
                except Exception as exc:
                    logger.warning("answering %s failed", question.id, exc_info=True)
                    scored.error = scored.error or f"answer: {type(exc).__name__}: {exc}"
            if meter is not None:
                scored.llm_calls, scored.cost_usd = meter.take()
            results.append(scored)
            if on_result is not None:
                on_result(scored, result)
    return results


def build_retrievers(
    variants: Sequence[str],
    *,
    namespace: str | None = None,
    llm: LLMClient | None = None,
    exact_hybrid: bool = False,
) -> dict[str, Retriever]:
    """The live retrievers (Pinecone, Postgres, Neo4j, models) for ``R1``..``R5``, sharing
    one embedder, reranker, analyzer and driver, sized by ``configs/`` (rerank candidates:
    ``pinecone.query_top_k``; graph: ``retrieval.graph``). R5 needs an LLM
    (``models.yaml``). ``exact_hybrid`` scores the hybrid query client-side
    (:class:`eval.tune_retrieval.ExactHybridStore`), for Pinecone Local, which ignores
    sparse vectors."""
    from ledgerlens.core.config import get_config
    from ledgerlens.ingestion.embedding import BM25Encoder, DenseEmbedder
    from ledgerlens.retrieval.analyze import QueryAnalyzer
    from ledgerlens.retrieval.graph import GraphExpander
    from ledgerlens.retrieval.graphrag import GraphRAGRetriever, Text2CypherRetriever
    from ledgerlens.retrieval.rerank import CrossEncoderReranker
    from ledgerlens.retrieval.text2cypher import CypherGenerator, CypherGuard
    from ledgerlens.retrieval.vector import (
        DenseRetriever,
        HybridRetriever,
        PostgresChunkTexts,
        RerankRetriever,
        VectorIndex,
    )
    from ledgerlens.storage.neo4j.driver import get_driver
    from ledgerlens.storage.pinecone_store import PineconeStore

    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants {unknown}; choose from {', '.join(VARIANTS)}")
    store: VectorIndex = PineconeStore.from_settings(namespace=namespace)
    embedder = DenseEmbedder()
    texts = PostgresChunkTexts()
    if exact_hybrid:
        from eval.tune_retrieval import BM25Documents, ExactHybridStore

        pinecone = get_config().pinecone
        store = ExactHybridStore(
            store,
            BM25Documents(BM25Encoder.load(), texts),
            alpha=pinecone.hybrid_alpha,
            top_k=pinecone.query_top_k,
        )
    r1 = DenseRetriever(store, embedder, texts=texts)
    r2 = HybridRetriever(store, embedder, BM25Encoder.load(), texts=texts)
    config = get_config()
    reranker = CrossEncoderReranker(config.retrieval.rerank_model)
    r3 = RerankRetriever(r2, reranker, candidates=config.pinecone.query_top_k)
    built: dict[str, Retriever] = {"R1": r1, "R2": r2, "R3": r3}
    if {"R4", "R5"} & set(variants):
        analyzer = QueryAnalyzer.from_postgres()
        graph = config.retrieval.graph
        expander = GraphExpander(
            get_driver(), max_hops=graph.max_hops, per_seed=graph.per_seed, limit=graph.limit
        )
        r4 = GraphRAGRetriever(r3, expander, analyzer=analyzer)
        built["R4"] = r4
        if "R5" in variants:
            generator = CypherGenerator(llm or LLMClient())
            built["R5"] = Text2CypherRetriever(r4, generator, CypherGuard(get_driver()))
    return {variant: built[variant] for variant in variants}


# --------------------------------------------------------------------------- report

CATEGORY_ORDER = ("lookup", "clause", "multi_hop", "aggregation", "reconciliation", "unanswerable")


@dataclass(frozen=True)
class Summary:
    """One variant over a set of questions."""

    questions: int
    answerable: int
    recall: float
    mrr: float
    evidence_recall: float
    p50_ms: float
    p95_ms: float
    degraded: int
    errors: int


def summarize(results: Iterable[QuestionResult]) -> Summary:
    items = list(results)
    scored = [r for r in items if r.recall is not None]
    latencies = [r.latency_ms for r in items]
    return Summary(
        questions=len(items),
        answerable=len(scored),
        recall=mean(r.recall or 0.0 for r in scored),
        mrr=mean(r.reciprocal_rank or 0.0 for r in scored),
        evidence_recall=mean(r.evidence_recall or 0.0 for r in scored),
        p50_ms=percentile(latencies, 50),
        p95_ms=percentile(latencies, 95),
        degraded=sum(bool(r.degraded) for r in items),
        errors=sum(r.error is not None for r in items),
    )


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def _matrix(
    results: Sequence[QuestionResult],
    variants: Sequence[str],
    groups: Sequence[tuple[str, Callable[[QuestionResult], bool]]],
    metric: Callable[[Summary], float],
) -> list[str]:
    """Rows: groups of questions; columns: variants; cells: ``metric``, best in bold."""
    rows = []
    for label, member in groups:
        summaries = {
            v: summarize(r for r in results if r.variant == v and member(r)) for v in variants
        }
        n = next(iter(summaries.values())).answerable if summaries else 0
        if n == 0:
            continue
        values = {v: metric(s) for v, s in summaries.items()}
        best = max(values.values())
        cells = [
            f"**{value:.3f}**" if value == best else f"{value:.3f}" for value in values.values()
        ]
        rows.append([label, str(n), *cells])
    return _table(["questions", "n", *variants], rows)


def render_report(
    results: Sequence[QuestionResult],
    variants: Sequence[str],
    *,
    k: int,
    setup: Mapping[str, str],
) -> str:
    """The comparison as markdown: overall, then recall@k, MRR and evidence recall per
    category and for questions that do / do not need the graph."""
    lines = [f"# Retrieval benchmark ({len({r.question_id for r in results})} questions)", ""]
    lines += [f"- **{key}:** {value}" for key, value in setup.items()]
    lines += ["", "## Overall", ""]
    rows = []
    for variant in variants:
        s = summarize(r for r in results if r.variant == variant)
        rows.append(
            [
                variant,
                VARIANTS[variant],
                f"{s.recall:.3f}",
                f"{s.mrr:.3f}",
                f"{s.evidence_recall:.3f}",
                f"{s.p50_ms:,.0f}",
                f"{s.p95_ms:,.0f}",
                str(s.degraded),
                str(s.errors),
            ]
        )
    lines += _table(
        [
            "variant",
            "retriever",
            f"recall@{k}",
            "MRR",
            "evidence recall",
            "p50 ms",
            "p95 ms",
            "degraded",
            "errors",
        ],
        rows,
    )
    present = [c for c in CATEGORY_ORDER if any(r.category == c for r in results)]

    def in_category(category: str) -> Callable[[QuestionResult], bool]:
        return lambda r: r.category == category

    groups: list[tuple[str, Callable[[QuestionResult], bool]]] = [
        (category, in_category(category)) for category in present
    ]
    groups += [
        ("needs graph", lambda r: r.requires_graph),
        ("no graph", lambda r: not r.requires_graph),
        ("all", lambda r: True),
    ]
    for title, metric in (
        (f"recall@{k} by category", lambda s: s.recall),
        ("MRR by category", lambda s: s.mrr),
        ("Evidence recall by category (chunks + graph facts)", lambda s: s.evidence_recall),
    ):
        lines += ["", f"## {title}", ""]
        lines += _matrix(results, variants, groups, metric)
    if any(r.answer is not None or r.llm_calls for r in results):
        lines += ["", "## Answers, latency and cost", ""]
        rows = []
        for variant in variants:
            mine = [r for r in results if r.variant == variant]
            answered = [r for r in mine if r.answer is not None]
            facts = [r.facts_found for r in answered if r.facts_found is not None]
            answer_ms = [r.answer_latency_ms or 0.0 for r in answered]
            end_to_end = [r.latency_ms + (r.answer_latency_ms or 0.0) for r in mine]
            cost = sum(r.cost_usd for r in mine)
            rows.append(
                [
                    variant,
                    f"{len(answered)}/{len(mine)}",
                    f"{mean(1.0 if r.citations_ok else 0.0 for r in answered):.3f}",
                    f"{mean(facts):.3f}",
                    f"{percentile(answer_ms, 50):,.0f}",
                    f"{percentile(end_to_end, 95):,.0f}",
                    f"{mean(r.llm_calls for r in mine):.2f}",
                    f"{cost / max(len(mine), 1):.5f}",
                    f"{cost:.4f}",
                ]
            )
        lines += _table(
            [
                "variant",
                "answered",
                "citations ok",
                "expected facts",
                "answer p50 ms",
                "end-to-end p95 ms",
                "LLM calls / q",
                "USD / q",
                "USD total",
            ],
            rows,
        )
        lines += [
            "",
            "*citations ok*: the answer cites at least one tag and every tag resolves to a "
            "document in its context (unanswerable: no invented tags). *expected facts*: share "
            "of the golden `expected` facts the answer states.",
        ]
    unanswerable = sum(1 for r in results if r.recall is None and r.variant == variants[0])
    lines += [
        "",
        "## Notes",
        "",
        f"- recall@{k} and MRR rank documents as the retriever returns them: chunk documents "
        "first, then the documents graph facts cite. Evidence recall counts every document "
        "in the result.",
        f"- {unanswerable} unanswerable question(s) have no relevant documents: they count "
        "towards latency but not towards recall or MRR.",
        "- *degraded*: questions where a stage failed and the retriever fell back "
        "(Neo4j for R4, Text2Cypher for R5). *errors*: the retriever raised.",
        "",
    ]
    return "\n".join(lines)


def warm_up(retrievers: Mapping[str, Retriever]) -> None:
    """One untimed query per variant, so model loading (embedder, cross-encoder) does not
    land in the first question's latency."""
    for retriever in retrievers.values():
        try:
            retriever.retrieve("total on invoice INV-000001", k=1)
        except Exception:  # the benchmark run will report it per question
            logger.warning("warm-up query failed", exc_info=True)


def analyzer_filters() -> FilterSource:
    """Filters from the query analyzer (vendors from Postgres), for every variant."""
    from ledgerlens.retrieval.analyze import QueryAnalyzer

    analyzer = QueryAnalyzer.from_postgres()
    return lambda question: analyzer.analyze(question).filters()


def write_results(results: Sequence[QuestionResult], path: Path) -> None:
    """One JSON object per variant and question, for later analysis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(asdict(result) | {"degraded": result.degraded}) for result in results]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_variants(text: str) -> list[str]:
    variants = [v.strip().upper() for v in text.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown or not variants:
        raise argparse.ArgumentTypeError(f"choose variants from {', '.join(VARIANTS)}")
    return variants


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.benchmark_retrieval",
        description="Benchmark retriever variants on the golden set.",
    )
    parser.add_argument("--variants", type=parse_variants, default=list(VARIANTS))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--limit", type=int, default=None, help="first N questions only")
    parser.add_argument(
        "--filters",
        choices=["analyzer", "none"],
        default="analyzer",
        help="metadata filters for every variant: the query analyzer's (default) or none",
    )
    parser.add_argument("--namespace", default=None, help="Pinecone namespace (default: env)")
    parser.add_argument("--out", type=Path, default=None, help="per-question results (JSONL)")
    parser.add_argument("--report", type=Path, default=None, help="markdown comparison report")
    parser.add_argument(
        "--answers",
        action="store_true",
        help="also answer every question with the LLM (paid calls) and score the answers",
    )
    parser.add_argument("--answer-model", default="smart", help="model alias for answers")
    parser.add_argument(
        "--exact-hybrid",
        action="store_true",
        help="score R2's hybrid query client-side (Pinecone Local ignores sparse vectors)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    questions = load_golden_set(args.golden)[: args.limit]
    meter = CostMeter()
    llm = metered(LLMClient(), meter) if args.answers or "R5" in args.variants else None
    retrievers = build_retrievers(
        args.variants, namespace=args.namespace, llm=llm, exact_hybrid=args.exact_hybrid
    )
    answerer = llm_answerer(llm, model=args.answer_model) if args.answers and llm else None
    filters = analyzer_filters() if args.filters == "analyzer" else None

    def progress(result: QuestionResult, _: RetrievalResult | None) -> None:
        recall = "  -  " if result.recall is None else f"{result.recall:.2f}"
        print(
            f"{result.question_id} {result.variant} recall@{args.k}={recall} "
            f"{result.latency_ms:7.0f} ms {' '.join(result.degraded)}",
            flush=True,
        )

    warm_up(retrievers)
    results = run_benchmark(
        questions,
        retrievers,
        k=args.k,
        filters=filters,
        on_result=progress,
        answerer=answerer,
        meter=meter if llm is not None else None,
    )
    if args.out:
        write_results(results, args.out)
    setup = {
        "date": date.today().isoformat(),
        "golden set": f"`{args.golden}` ({len(questions)} questions)",
        "k": str(args.k),
        "answers": f"`{args.answer_model}` alias" if args.answers else "not generated",
        "hybrid scoring": "exact, client-side (Pinecone Local ignores sparse vectors)"
        if args.exact_hybrid
        else "by the index",
        "filters": "query analyzer (same for every variant)"
        if args.filters == "analyzer"
        else "none",
        "command": "`python -m eval.benchmark_retrieval " + " ".join(argv or sys.argv[1:]) + "`",
    }
    report = render_report(results, args.variants, k=args.k, setup=setup)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
        print(f"wrote {args.report}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
