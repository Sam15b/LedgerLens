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
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from eval.golden_set.schema import DEFAULT_PATH as DEFAULT_GOLDEN
from eval.golden_set.schema import GoldenQuestion, load_golden_set
from ledgerlens.retrieval import QueryFilters, RetrievalResult, Retriever

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


def run_benchmark(
    questions: Sequence[GoldenQuestion],
    retrievers: Mapping[str, Retriever],
    *,
    k: int = 5,
    filters: FilterSource | None = None,
    on_result: Callable[[QuestionResult, RetrievalResult | None], None] | None = None,
) -> list[QuestionResult]:
    """Every variant on every question, on the same footing: the same ``k`` and the same
    filters (``filters(question)``, e.g. the query analyzer's; ``None``: no filters, and
    R4/R5 analyze the question themselves). A retriever that raises scores 0 on that
    question instead of stopping the run."""
    results: list[QuestionResult] = []
    for question in questions:
        question_filters = filters(question.question) if filters is not None else None
        for variant, retriever in retrievers.items():
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
            results.append(scored)
            if on_result is not None:
                on_result(scored, result)
    return results


def build_retrievers(
    variants: Sequence[str], *, namespace: str | None = None
) -> dict[str, Retriever]:
    """The live retrievers (Pinecone, Postgres, Neo4j, models) for ``R1``..``R5``, sharing
    one embedder, reranker, analyzer and driver. R5 needs an LLM (``models.yaml``)."""
    from ledgerlens.ingestion.embedding import BM25Encoder, DenseEmbedder
    from ledgerlens.llm.router import LLMClient
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
    )
    from ledgerlens.storage.neo4j.driver import get_driver
    from ledgerlens.storage.pinecone_store import PineconeStore

    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants {unknown}; choose from {', '.join(VARIANTS)}")
    store = PineconeStore.from_settings(namespace=namespace)
    embedder = DenseEmbedder()
    texts = PostgresChunkTexts()
    r1 = DenseRetriever(store, embedder, texts=texts)
    r2 = HybridRetriever(store, embedder, BM25Encoder.load(), texts=texts)
    r3 = RerankRetriever(r2, CrossEncoderReranker())
    built: dict[str, Retriever] = {"R1": r1, "R2": r2, "R3": r3}
    if {"R4", "R5"} & set(variants):
        analyzer = QueryAnalyzer.from_postgres()
        r4 = GraphRAGRetriever(r3, GraphExpander(get_driver()), analyzer=analyzer)
        built["R4"] = r4
        if "R5" in variants:
            generator = CypherGenerator(LLMClient())
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    questions = load_golden_set(args.golden)[: args.limit]
    retrievers = build_retrievers(args.variants, namespace=args.namespace)
    filters = analyzer_filters() if args.filters == "analyzer" else None

    def progress(result: QuestionResult, _: RetrievalResult | None) -> None:
        recall = "  -  " if result.recall is None else f"{result.recall:.2f}"
        print(
            f"{result.question_id} {result.variant} recall@{args.k}={recall} "
            f"{result.latency_ms:7.0f} ms {' '.join(result.degraded)}",
            flush=True,
        )

    warm_up(retrievers)
    results = run_benchmark(questions, retrievers, k=args.k, filters=filters, on_result=progress)
    if args.out:
        write_results(results, args.out)
    setup = {
        "date": date.today().isoformat(),
        "golden set": f"`{args.golden}` ({len(questions)} questions)",
        "k": str(args.k),
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
