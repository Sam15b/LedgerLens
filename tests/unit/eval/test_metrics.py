"""Retrieval benchmark metrics, the runner and the report, with stub retrievers."""

from typing import Any

import pytest

from eval.benchmark_retrieval import (
    AnswerOutcome,
    CostMeter,
    expected_facts_found,
    metered,
    percentile,
    recall_at_k,
    reciprocal_rank,
    render_report,
    retrieved_documents,
    run_benchmark,
)
from eval.golden_set.schema import GoldenQuestion
from ledgerlens.llm.types import LLMResponse
from ledgerlens.retrieval import GraphFact, QueryFilters, RetrievalResult, ScoredChunk

A, B, C, D = (f"{n}0000000-0000-0000-0000-000000000000" for n in "abcd")


# --------------------------------------------------------------------------- metrics


@pytest.mark.parametrize(
    ("retrieved", "relevant", "k", "expected"),
    [
        ([A, B, C], [A], 1, 1.0),
        ([B, A, C], [A], 1, 0.0),
        ([B, A, C], [A], 2, 1.0),
        ([A, B, C], [A, C], 2, 0.5),
        ([A, B, C], [A, C], 3, 1.0),
        ([A, A, A, B], [B], 2, 1.0),  # repeats do not use up the k slots
        ([], [A], 5, 0.0),
        ([A], [], 5, 1.0),  # nothing to miss
    ],
)
def test_recall_at_k(retrieved: list[str], relevant: list[str], k: int, expected: float) -> None:
    assert recall_at_k(retrieved, relevant, k) == expected


def test_recall_needs_a_positive_k() -> None:
    with pytest.raises(ValueError):
        recall_at_k([A], [A], 0)


@pytest.mark.parametrize(
    ("retrieved", "relevant", "expected"),
    [
        ([A, B], [A], 1.0),
        ([B, A], [A], 0.5),
        ([B, C, D, A], [A, D], 1 / 3),  # the first relevant one counts
        ([B, B, A], [A], 0.5),  # rank among unique documents
        ([B, C], [A], 0.0),
    ],
)
def test_reciprocal_rank(retrieved: list[str], relevant: list[str], expected: float) -> None:
    assert reciprocal_rank(retrieved, relevant) == pytest.approx(expected)


def test_percentile() -> None:
    assert percentile([], 95) == 0.0
    assert percentile([5.0], 95) == 5.0
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([0, 100], 95) == pytest.approx(95)
    assert percentile(range(1, 101), 95) == pytest.approx(95.05)


def chunk(document_id: str, number: str | None = None) -> ScoredChunk:
    return ScoredChunk(
        id=f"{document_id}#c0", document_id=document_id, text="t", score=1.0, doc_number=number
    )


def test_retrieved_documents_are_chunks_then_fact_documents_once_each() -> None:
    result = RetrievalResult.build(
        "q",
        "r4",
        [chunk(A), chunk(B), chunk(A)],
        graph_facts=[
            GraphFact(text="f", document_ids=(B, C)),
            GraphFact(text="g", document_ids=(D,)),
        ],
    )

    assert retrieved_documents(result) == [A, B, C, D]


# --------------------------------------------------------------------------- expected facts


@pytest.mark.parametrize(
    ("answer", "expected", "share"),
    [
        (
            "Invoice INV-1 totals 12,656.66 GBP [doc:INV-1].",
            {"amount": "12,656.66", "currency": "GBP"},
            1.0,
        ),
        ("It totals 12656.66 [doc:INV-1].", {"amount": "12,656.66", "currency": "GBP"}, 0.5),
        ("Settled by PAY-7 on 26 May 2025.", {"number": "PAY-7", "date": "2025-05-26"}, 1.0),
        ("Settled by PAY-7 on 2025-05-27.", {"number": "PAY-7", "date": "2025-05-26"}, 0.5),
        ("INV-1 and INV-3 have no payment.", {"numbers": ["INV-1", "INV-2", "INV-3"]}, 2 / 3),
        ("Acme issued 22 invoices in 2026.", {"count": 22}, 1.0),
        ("Acme issued 122 invoices.", {"count": 22}, 0.0),
        ("Payment is due within 30 days.", {"days": 30}, 1.0),
        ("No. Coffee is billed 25% above the rate.", {"match": False, "items": ["Coffee"]}, 1.0),
        ("Yes, they match.", {"match": False}, 0.0),
        ("Yes. It appears twice.", {"duplicate": True}, 1.0),
        ("There is no invoice INV-999999 in the records.", {"answerable": False}, 1.0),
        ("INV-999999 totals 10.00 GBP.", {"answerable": False}, 0.0),
        ("Anything.", {}, None),
    ],
)
def test_expected_facts_found(answer: str, expected: dict[str, Any], share: float | None) -> None:
    assert expected_facts_found(answer, expected) == (
        None if share is None else pytest.approx(share)
    )


# --------------------------------------------------------------------------- runner and report


def question(id: str, category: str, relevant: list[str], **expected: Any) -> GoldenQuestion:
    return GoldenQuestion(
        id=id,
        category=category,  # type: ignore[arg-type]
        question=f"Question {id}?",
        ground_truth_answer="a",
        relevant_document_ids=tuple(relevant),
        requires_graph=category == "multi_hop",
        difficulty="easy",
        expected=expected,
    )


QUESTIONS = [
    question("q001", "lookup", [A], amount="1.00"),
    question("q002", "multi_hop", [B, C]),
    question("q003", "unanswerable", [], answerable=False),
]


class Stub:
    """Returns fixed documents per question; ``facts`` adds graph documents."""

    def __init__(self, name: str, ranking: dict[str, list[str]], facts: bool = False) -> None:
        self.name = name
        self.ranking = ranking
        self.facts = facts
        self.calls: list[QueryFilters | None] = []

    def retrieve(
        self, query: str, *, filters: QueryFilters | None = None, k: int = 5
    ) -> RetrievalResult:
        self.calls.append(filters)
        qid = query.removeprefix("Question ").rstrip("?")
        if qid == "q999":
            raise RuntimeError("boom")
        documents = self.ranking.get(qid, [])
        facts = [GraphFact(text="fact", document_ids=(C,))] if self.facts else []
        timings = {"total": 10.0 if self.facts else 5.0}
        if self.facts and qid == "q002":
            timings["graph_error"] = 1.0
        return RetrievalResult.build(
            query, self.name, [chunk(d) for d in documents], facts, timings
        )


def test_runner_scores_every_variant_on_every_question() -> None:
    r1 = Stub("r1", {"q001": [A], "q002": [D, B]})
    r4 = Stub("r4", {"q001": [A], "q002": [B]}, facts=True)
    filters = QueryFilters(vendor_ids=("V001",))

    results = run_benchmark(QUESTIONS, {"R1": r1, "R4": r4}, k=1, filters=lambda q: filters)

    by = {(r.variant, r.question_id): r for r in results}
    assert len(results) == 6
    assert by[("R1", "q002")].recall == 0.0 and by[("R1", "q002")].reciprocal_rank == 0.5
    assert by[("R1", "q002")].evidence_recall == 0.5
    assert by[("R4", "q002")].recall == 0.5  # B in the top 1; C only via the facts
    assert by[("R4", "q002")].evidence_recall == 1.0
    assert by[("R4", "q002")].degraded == ["graph"]
    assert by[("R1", "q003")].recall is None  # unanswerable: nothing to recall
    assert by[("R4", "q001")].latency_ms == 10.0
    assert r1.calls == [filters] * 3 and r4.calls == [filters] * 3


def test_a_failing_retriever_scores_zero_and_the_run_goes_on() -> None:
    broken = [question("q999", "lookup", [A]), QUESTIONS[0]]

    results = run_benchmark(broken, {"R1": Stub("r1", {"q001": [A]})}, k=5)

    assert results[0].error == "RuntimeError: boom" and results[0].recall == 0.0
    assert results[1].recall == 1.0


def test_answers_are_scored_and_costs_attributed() -> None:
    meter = CostMeter()

    def answerer(q: GoldenQuestion, result: RetrievalResult) -> AnswerOutcome:
        meter.record(LLMResponse(text="x", provider="p", model="m", cost_usd=0.002))
        text = "It is 1.00 [doc:x]." if q.id == "q001" else "There is no such record."
        return AnswerOutcome(text, 50.0, citations_ok=q.id != "q002", unknown_citations=[])

    results = run_benchmark(
        QUESTIONS, {"R1": Stub("r1", {"q001": [A]})}, k=5, answerer=answerer, meter=meter
    )

    assert [r.facts_found for r in results] == [1.0, None, 1.0]
    assert [r.citations_ok for r in results] == [True, False, True]
    assert all(r.llm_calls == 1 and r.cost_usd == pytest.approx(0.002) for r in results)


def test_metered_client_counts_every_completion() -> None:
    class FakeLLM:
        def complete(self, messages: Any, **_: Any) -> LLMResponse:
            return LLMResponse(text="ok", provider="p", model="m", cost_usd=0.5)

    meter = CostMeter()
    llm = metered(FakeLLM(), meter)  # type: ignore[arg-type]
    llm.complete([])
    llm.complete([])

    assert meter.take() == (2, 1.0)
    assert meter.take() == (0, 0.0)


def test_report_has_overall_and_per_category_tables() -> None:
    r1 = Stub("r1", {"q001": [A], "q002": [D, B]})
    r4 = Stub("r4", {"q001": [A], "q002": [B]}, facts=True)
    results = run_benchmark(QUESTIONS, {"R1": r1, "R4": r4}, k=1)

    report = render_report(results, ["R1", "R4"], k=1, setup={"k": "1"})

    assert report.startswith("# Retrieval benchmark (3 questions)")
    assert "| R4 | GraphRAG (R3 seeds + graph templates) | 0.750 | 1.000 | 1.000 |" in report
    assert "| multi_hop | 1 | 0.000 | **0.500** |" in report
    assert "| needs graph | 1 |" in report and "| unanswerable |" not in report
    assert "Answers, latency and cost" not in report
    assert "1 unanswerable question(s)" in report
