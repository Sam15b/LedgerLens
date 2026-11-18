"""Tune retrieval settings on the golden set, one knob at a time.

Each sweep runs :func:`eval.benchmark_retrieval.run_benchmark` with one retriever per value
of the knob, so every value sees the same questions, ``k`` and analyzer filters. The pick is
the best value on the knob's metric, then MRR, then the cheaper value::

    PINECONE_HOST=http://localhost:5080 python -m eval.tune_retrieval alpha \
        --report eval/reports/<date>/tuning.md

* ``alpha`` -- the dense weight of R2's hybrid query (``hybrid_alpha`` in
  ``configs/pinecone.yaml``), on R2 and on R3 (whose rerank pool R2 supplies);
* ``candidates`` -- how many of R2's chunks R3 reranks (``query_top_k`` in
  ``configs/pinecone.yaml``), picked by recall@k, ties to fewer (the cross-encoder's cost
  grows with each one);
* ``hops`` -- how far R4 walks the graph from its seeds (``graph.max_hops`` in
  ``configs/retrieval.yaml``), picked by evidence recall, ties to fewer facts per question
  (a smaller context to answer from).

**Exact hybrid scoring.** Pinecone Local stores no sparse values and ignores the sparse half
of a query, so on it every ``alpha`` returns the dense ranking. :class:`ExactHybridStore`
does the hybrid scoring itself: it fetches the dense ranking of *every* chunk the filter
allows (the dev namespace is ~1.7k chunks), encodes those chunks with the same fitted BM25
encoder ingestion used, and ranks by ``alpha * dense + (1 - alpha) * sparse`` -- the score a
dotproduct index gives an ``alpha``-scaled hybrid query, computed exactly instead of
approximately.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from eval.benchmark_retrieval import (
    QuestionResult,
    Summary,
    analyzer_filters,
    run_benchmark,
    summarize,
    warm_up,
)
from eval.golden_set.schema import DEFAULT_PATH as DEFAULT_GOLDEN
from eval.golden_set.schema import GoldenQuestion, load_golden_set
from ledgerlens.ingestion.embedding import BM25Encoder, SparseVector
from ledgerlens.retrieval import Retriever
from ledgerlens.retrieval.vector import ChunkTexts, VectorIndex
from ledgerlens.storage.pinecone_store import VectorHit

logger = logging.getLogger(__name__)

DEFAULT_POOL = 5000
"""Dense candidates fetched before hybrid rescoring: more than the dev namespace holds."""


def sparse_dot(a: SparseVector, b: SparseVector) -> float:
    """Dot product of two sparse vectors."""
    weights = dict(zip(a.indices, a.values, strict=True))
    return sum(weights.get(i, 0.0) * v for i, v in zip(b.indices, b.values, strict=True))


class BM25Documents:
    """Sparse document vectors by vector id, encoded from the chunks' full text (cached)."""

    def __init__(self, encoder: BM25Encoder, texts: ChunkTexts) -> None:
        self.encoder = encoder
        self.texts = texts
        self._cache: dict[str, SparseVector] = {}

    def vectors(self, ids: Sequence[str]) -> dict[str, SparseVector]:
        missing = [i for i in dict.fromkeys(ids) if i not in self._cache]
        if missing:
            found = self.texts.texts(missing)
            known = [i for i in missing if i in found]
            encoded = self.encoder.encode_documents([found[i] for i in known])
            self._cache.update(zip(known, encoded, strict=True))
            # a chunk with no stored text has no sparse terms
            self._cache.update({i: SparseVector() for i in missing if i not in found})
        return {i: self._cache[i] for i in ids}


class ExactHybridStore:
    """A store whose hybrid query is scored here, exactly: see the module docstring.

    Dense rankings are cached per (query vector, filter), so sweeping ``alpha`` costs one
    vector query per question, not one per value.
    """

    def __init__(
        self,
        store: VectorIndex,
        documents: BM25Documents,
        *,
        alpha: float = 0.7,
        top_k: int = 20,
        pool: int = DEFAULT_POOL,
    ) -> None:
        self.store = store
        self.documents = documents
        self.alpha = alpha
        self.top_k = top_k
        self.pool = pool
        self._dense: dict[str, list[VectorHit]] = {}

    def dense_ranking(
        self,
        dense: Sequence[float],
        filter: Mapping[str, Any] | None,
        namespace: str | None,
    ) -> list[VectorHit]:
        key = json.dumps([list(dense), filter, namespace], sort_keys=True, default=str)
        if key not in self._dense:
            self._dense[key] = self.store.query(
                dense, None, top_k=self.pool, filter=filter, namespace=namespace
            )
        return self._dense[key]

    def query(
        self,
        dense: Sequence[float],
        sparse: SparseVector | None = None,
        *,
        top_k: int | None = None,
        filter: Mapping[str, Any] | None = None,
        namespace: str | None = None,
        alpha: float | None = None,
    ) -> list[VectorHit]:
        """Best ``top_k`` by ``alpha * dense + (1 - alpha) * sparse`` (ties: dense score).
        Without sparse terms the dense score is used unscaled, as the real index does."""
        weight = self.alpha if alpha is None else alpha
        if not 0 <= weight <= 1:
            raise ValueError(f"alpha must be in [0, 1], got {weight}")
        hits = self.dense_ranking(dense, filter, namespace)
        if sparse is None or not sparse.indices or weight == 1:
            scored = [(hit.score, hit.score, hit) for hit in hits]
        else:
            docs = self.documents.vectors([hit.id for hit in hits])
            scored = [
                (
                    weight * hit.score + (1 - weight) * sparse_dot(sparse, docs[hit.id]),
                    hit.score,
                    hit,
                )
                for hit in hits
            ]
        scored.sort(key=lambda item: (-item[0], -item[1]))
        return [
            VectorHit(id=hit.id, score=score, metadata=hit.metadata)
            for score, _, hit in scored[: top_k or self.top_k]
        ]


# --------------------------------------------------------------------------- sweeps


@dataclass(frozen=True)
class SweepRow:
    """One value of a knob on one retriever variant."""

    value: float
    summary: Summary
    mean_facts: float


@dataclass(frozen=True)
class Sweep:
    """A knob swept on one variant: the rows, the metric that picks, and the pick."""

    knob: str
    variant: str
    metric: str
    rows: list[SweepRow]
    best: float
    current: float


METRICS: dict[str, Callable[[Summary], float]] = {
    "recall": lambda s: s.recall,
    "mrr": lambda s: s.mrr,
    "evidence_recall": lambda s: s.evidence_recall,
}


def choose(
    rows: Sequence[SweepRow],
    metric: str,
    *,
    cost: Callable[[SweepRow], float] | None = None,
    tolerance: float = 1e-9,
) -> float:
    """The best value on ``metric``, then on MRR; a tie goes to the cheaper value
    (``cost``, default: the summary's p50 latency), then the smaller value."""
    if not rows:
        raise ValueError("nothing to choose from")
    primary = METRICS[metric]
    cost = cost or (lambda row: row.summary.p50_ms)
    top = max(primary(r.summary) for r in rows)
    tied = [r for r in rows if primary(r.summary) >= top - tolerance]
    best_mrr = max(r.summary.mrr for r in tied)
    tied = [r for r in tied if r.summary.mrr >= best_mrr - tolerance]
    return min(tied, key=lambda r: (cost(r), r.value)).value


def label(knob: str, value: float) -> str:
    return f"{knob}={value:g}"


def sweep(
    questions: Sequence[GoldenQuestion],
    knob: str,
    variant: str,
    build: Callable[[float], Retriever],
    values: Sequence[float],
    *,
    metric: str,
    current: float,
    k: int = 5,
    filters: Callable[[str], Any] | None = None,
    cost: Callable[[SweepRow], float] | None = None,
    on_result: Callable[[QuestionResult], None] | None = None,
) -> Sweep:
    """Run ``build(value)`` for every value over ``questions`` and pick the best."""
    retrievers = {label(knob, v): build(v) for v in values}
    results = run_benchmark(
        questions,
        retrievers,
        k=k,
        filters=filters,
        on_result=(lambda r, _: on_result(r)) if on_result else None,
    )
    rows = []
    for value in values:
        mine = [r for r in results if r.variant == label(knob, value)]
        facts = sum(r.facts for r in mine) / len(mine) if mine else 0.0
        rows.append(SweepRow(value, summarize(mine), facts))
    best = choose(rows, metric, cost=cost)
    return Sweep(knob, variant, metric, rows, best, current)


def render_sweep(result: Sweep, *, k: int) -> list[str]:
    """One sweep as a markdown table, the pick in bold."""
    names = {"recall": f"recall@{k}", "mrr": "MRR", "evidence_recall": "evidence recall"}
    lines = [
        f"### {result.knob} on {result.variant} (picked by {names[result.metric]})",
        "",
        f"| {result.knob} | recall@{k} | MRR | evidence recall | facts / question "
        "| p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in result.rows:
        s = row.summary
        value = f"{row.value:g}"
        if row.value == result.best:
            value = f"**{value}** (picked)"
        if row.value == result.current:
            value += " (was)"
        lines.append(
            f"| {value} | {s.recall:.3f} | {s.mrr:.3f} | {s.evidence_recall:.3f} "
            f"| {row.mean_facts:.1f} | {s.p50_ms:,.0f} | {s.p95_ms:,.0f} |"
        )
    return [*lines, ""]


def render_report(sweeps: Sequence[Sweep], *, k: int, setup: Mapping[str, str]) -> str:
    lines = ["# Retrieval tuning", ""]
    lines += [f"- **{key}:** {value}" for key, value in setup.items()]
    lines += ["", "## Picks", "", "| knob | variant | was | picked |", "|---|---|---|---|"]
    lines += [f"| {s.knob} | {s.variant} | {s.current:g} | {s.best:g} |" for s in sweeps]
    lines += ["", "## Sweeps", ""]
    for result in sweeps:
        lines += render_sweep(result, k=k)
    return "\n".join(lines)


# --------------------------------------------------------------------------- live run


@dataclass
class Live:
    """The live components every sweep shares (models load once)."""

    store: ExactHybridStore
    embedder: Any
    sparse: BM25Encoder
    texts: ChunkTexts
    reranker: Any


def live_components(namespace: str | None = None) -> Live:
    from ledgerlens.core.config import get_config
    from ledgerlens.ingestion.embedding import DenseEmbedder
    from ledgerlens.retrieval.rerank import CrossEncoderReranker
    from ledgerlens.retrieval.vector import PostgresChunkTexts
    from ledgerlens.storage.pinecone_store import PineconeStore

    config = get_config().pinecone
    texts = PostgresChunkTexts()
    encoder = BM25Encoder.load()
    store = ExactHybridStore(
        PineconeStore.from_settings(namespace=namespace),
        BM25Documents(encoder, texts),
        alpha=config.hybrid_alpha,
        top_k=config.query_top_k,
    )
    return Live(store, DenseEmbedder(), encoder, texts, CrossEncoderReranker())


def alpha_sweeps(
    live: Live,
    questions: Sequence[GoldenQuestion],
    values: Sequence[float],
    *,
    current: float,
    k: int,
    filters: Callable[[str], Any] | None,
    on_result: Callable[[QuestionResult], None] | None = None,
) -> list[Sweep]:
    """``alpha`` on R2 (recall@k), then on R3 (recall@k after reranking R2's pool). Ties
    go to the value nearest ``current``: every alpha costs the same."""
    from ledgerlens.retrieval.vector import HybridRetriever, RerankRetriever

    def r2(alpha: float) -> Retriever:
        return HybridRetriever(
            live.store,
            live.embedder,
            live.sparse,
            alpha=alpha,
            texts=live.texts,
        )

    def r3(alpha: float) -> Retriever:
        return RerankRetriever(r2(alpha), live.reranker)

    common: dict[str, Any] = {
        "metric": "recall",
        "current": current,
        "k": k,
        "filters": filters,
        "on_result": on_result,
        "cost": lambda row: abs(row.value - current),
    }
    return [
        sweep(questions, "alpha", "R2", r2, values, **common),
        sweep(questions, "alpha", "R3", r3, values, **common),
    ]


def candidates_sweep(
    live: Live,
    questions: Sequence[GoldenQuestion],
    values: Sequence[float],
    *,
    current: float,
    k: int,
    filters: Callable[[str], Any] | None,
    on_result: Callable[[QuestionResult], None] | None = None,
) -> Sweep:
    """How many hybrid candidates R3 reranks: recall@k, ties to fewer candidates."""
    from ledgerlens.retrieval.vector import RerankRetriever

    def r3(candidates: float) -> Retriever:
        return RerankRetriever(live_retriever(live), live.reranker, candidates=int(candidates))

    return sweep(
        questions,
        "candidates",
        "R3",
        r3,
        values,
        metric="recall",
        current=current,
        k=k,
        filters=filters,
        cost=lambda row: row.value,
        on_result=on_result,
    )


def hops_sweep(
    live: Live,
    questions: Sequence[GoldenQuestion],
    values: Sequence[float],
    *,
    current: float,
    candidates: int,
    k: int,
    filters: Callable[[str], Any] | None,
    on_result: Callable[[QuestionResult], None] | None = None,
) -> Sweep:
    """R4's graph hops: evidence recall, ties to fewer facts per question."""
    from ledgerlens.core.config import get_config
    from ledgerlens.retrieval.analyze import QueryAnalyzer
    from ledgerlens.retrieval.graph import GraphExpander
    from ledgerlens.retrieval.graphrag import GraphRAGRetriever
    from ledgerlens.retrieval.vector import RerankRetriever
    from ledgerlens.storage.neo4j.driver import get_driver

    graph = get_config().retrieval.graph
    analyzer = QueryAnalyzer.from_postgres()
    driver = get_driver()

    def r4(hops: float) -> Retriever:
        expander = GraphExpander(
            driver, max_hops=int(hops), per_seed=graph.per_seed, limit=graph.limit
        )
        seeds = RerankRetriever(live_retriever(live), live.reranker, candidates=candidates)
        return GraphRAGRetriever(seeds, expander, analyzer=analyzer)

    return sweep(
        questions,
        "hops",
        "R4",
        r4,
        values,
        metric="evidence_recall",
        current=current,
        k=k,
        filters=filters,
        cost=lambda row: row.mean_facts,
        on_result=on_result,
    )


def parse_floats(text: str) -> list[float]:
    try:
        return [float(v) for v in text.split(",") if v.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.tune_retrieval",
        description="Sweep retrieval settings on the golden set.",
    )
    parser.add_argument("knobs", nargs="+", choices=["alpha", "candidates", "hops"])
    parser.add_argument(
        "--alpha", type=parse_floats, default=[i / 10 for i in range(11)], help="alpha values"
    )
    parser.add_argument(
        "--candidates",
        type=parse_floats,
        default=[5, 10, 20, 30, 50],
        help="rerank candidate counts",
    )
    parser.add_argument("--hops", type=parse_floats, default=[0, 1, 2], help="graph hop limits")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--limit", type=int, default=None, help="first N questions only")
    parser.add_argument("--namespace", default=None, help="Pinecone namespace (default: env)")
    parser.add_argument("--report", type=Path, default=None, help="markdown report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from ledgerlens.core.config import get_config

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    questions = load_golden_set(args.golden)[: args.limit]
    live = live_components(args.namespace)
    filters = analyzer_filters()
    config = get_config()

    def progress(result: QuestionResult) -> None:
        print(f"{result.variant:<14} {result.question_id}", end="\r", flush=True)

    from ledgerlens.retrieval.vector import RerankRetriever

    warm_up({"warm": RerankRetriever(live_retriever(live), live.reranker)})
    sweeps: list[Sweep] = []
    if "alpha" in args.knobs:
        sweeps += alpha_sweeps(
            live,
            questions,
            args.alpha,
            current=config.pinecone.hybrid_alpha,
            k=args.k,
            filters=filters,
            on_result=progress,
        )
    candidates = config.pinecone.query_top_k
    if "candidates" in args.knobs:
        picked = candidates_sweep(
            live,
            questions,
            args.candidates,
            current=candidates,
            k=args.k,
            filters=filters,
            on_result=progress,
        )
        sweeps.append(picked)
        candidates = int(picked.best)
    if "hops" in args.knobs:
        sweeps.append(
            hops_sweep(
                live,
                questions,
                args.hops,
                current=config.retrieval.graph.max_hops,
                candidates=candidates,
                k=args.k,
                filters=filters,
                on_result=progress,
            )
        )
    print()
    setup = {
        "date": date.today().isoformat(),
        "golden set": f"`{args.golden}` ({len(questions)} questions)",
        "k": str(args.k),
        "hybrid scoring": "exact, client-side (`ExactHybridStore`): Pinecone Local ignores "
        "sparse vectors",
        "filters": "query analyzer (same for every value)",
        "command": "`python -m eval.tune_retrieval " + " ".join(argv or sys.argv[1:]) + "`",
    }
    report = render_report(sweeps, k=args.k, setup=setup)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
        print(f"wrote {args.report}")
    else:
        print(report)
    return 0


def live_retriever(live: Live) -> Retriever:
    from ledgerlens.retrieval.vector import HybridRetriever

    return HybridRetriever(live.store, live.embedder, live.sparse, texts=live.texts)


if __name__ == "__main__":
    raise SystemExit(main())
