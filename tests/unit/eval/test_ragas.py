"""The RAGAS runner with a scripted judge: each metric's score follows from the verdicts
the judge gives; the router judge, the cache and the CLI do what they say. No model is
called."""

import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("ragas.metrics.collections")

from pydantic import BaseModel  # noqa: E402
from ragas.metrics.collections.faithfulness.util import StatementGeneratorOutput  # noqa: E402

from eval import run_ragas  # noqa: E402
from eval.golden_set.schema import GoldenQuestion  # noqa: E402
from eval.run_ragas import (  # noqa: E402
    JudgeCache,
    Sample,
    build_metrics,
    cached_judge,
    clean_error,
    collect_samples,
    dense_embeddings,
    judge_key,
    load_samples,
    render_report,
    router_judge,
    save_samples,
    score_samples,
    summarize,
)
from ledgerlens.llm.types import Message  # noqa: E402
from ledgerlens.retrieval import GraphFact, RetrievalResult, ScoredChunk  # noqa: E402

Reply = Callable[[str, type[BaseModel]], dict[str, Any]]

QUESTION = "What is the total on invoice INV-1?"
SAMPLE = Sample(
    question_id="q001",
    category="lookup",
    user_input=QUESTION,
    response="INV-1 totals 90.00 GBP. It is due on 2025-03-12.",
    retrieved_contexts=["INVOICE INV-1 Total 90.00 GBP", "PO-7 for pallet wrap"],
    reference="Invoice INV-1 totals 90.00 GBP.",
)


def verdicts(
    *,
    statements: Sequence[str] = ("INV-1 totals 90.00 GBP.", "It is due on 2025-03-12."),
    supported: Sequence[int] = (1, 0),
    question: str = QUESTION,
    noncommittal: int = 0,
    relevant: Sequence[int] = (1, 0),
    attributed: Sequence[int] = (1, 1, 0),
    tp: int = 1,
    fp: int = 1,
    fn: int = 0,
) -> Reply:
    """A judge reply for each RAGAS response model, by its name."""
    contexts = iter(list(relevant) * 10)

    def reply(prompt: str, model: type[BaseModel]) -> dict[str, Any]:
        def items(n: int) -> list[dict[str, str]]:
            return [{"statement": f"s{k}", "reason": "r"} for k in range(n)]

        table: dict[str, Callable[[], dict[str, Any]]] = {
            "StatementGeneratorOutput": lambda: {"statements": list(statements)},
            "NLIStatementOutput": lambda: {
                "statements": [
                    {"statement": s, "reason": "r", "verdict": v}
                    for s, v in zip(statements, supported, strict=True)
                ]
            },
            "AnswerRelevanceOutput": lambda: {"question": question, "noncommittal": noncommittal},
            "ContextPrecisionOutput": lambda: {"reason": "r", "verdict": next(contexts)},
            "ContextRecallOutput": lambda: {
                "classifications": [
                    {"statement": f"s{k}", "reason": "r", "attributed": a}
                    for k, a in enumerate(attributed)
                ]
            },
            "ClassificationWithReason": lambda: {"TP": items(tp), "FP": items(fp), "FN": items(fn)},
        }
        return table[model.__name__]()

    return reply


class Scripted:
    """An LLMClient stand-in: ``structured`` answers from a :data:`Reply`."""

    def __init__(self, reply: Reply, *, fail: bool = False) -> None:
        self.reply = reply
        self.fail = fail
        self.calls: list[tuple[list[Message], str, str]] = []

    def structured(self, messages: list[Message], schema: type[Any], **kwargs: Any) -> Any:
        self.calls.append((messages, schema.__name__, kwargs.get("model", "")))
        if self.fail:
            raise RuntimeError("judge unavailable: sk-proj-abcdef123456 rejected")
        return schema.model_validate(self.reply(messages[0].content, schema))


class SameVector:
    """Every text gets the same embedding: cosine similarity 1."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.6, 0.8] for _ in texts]


class ByText:
    """The original question and anything else point in orthogonal directions."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] if t == QUESTION else [0.0, 1.0] for t in texts]


def scores(llm: Scripted, encoder: Any = None, sample: Sample = SAMPLE) -> dict[str, float]:
    metrics = build_metrics(router_judge(llm), dense_embeddings(encoder or SameVector()))
    (scored,) = asyncio.run(score_samples([sample], metrics))
    assert scored.errors == {}
    return scored.scores


# --------------------------------------------------------------------------- metrics


def test_scores_follow_the_judges_verdicts() -> None:
    got = scores(Scripted(verdicts()))
    assert got["faithfulness"] == pytest.approx(0.5)  # 1 of 2 claims supported
    assert got["answer_relevancy"] == pytest.approx(1.0)  # same question, same vector
    assert got["context_precision"] == pytest.approx(1.0)  # the relevant context is first
    assert got["context_recall"] == pytest.approx(2 / 3)  # 2 of 3 reference claims found
    f1 = 1 / (1 + 0.5 * (1 + 0))  # TP 1, FP 1, FN 0
    assert got["answer_correctness"] == pytest.approx(0.75 * f1 + 0.25 * 1.0)


def test_scores_move_the_right_way() -> None:
    worse = verdicts(
        supported=(0, 0), noncommittal=1, relevant=(0, 1), attributed=(0, 0, 1), tp=0, fp=1, fn=1
    )
    got = scores(Scripted(worse))
    assert got["faithfulness"] == 0.0
    assert got["answer_relevancy"] == 0.0  # a noncommittal answer is not relevant
    assert got["context_precision"] == pytest.approx(0.5)  # relevant context ranked second
    assert got["context_recall"] == pytest.approx(1 / 3)
    assert got["answer_correctness"] == pytest.approx(0.25)  # F1 0, similarity 1


def test_answer_relevancy_uses_the_embeddings() -> None:
    got = scores(Scripted(verdicts(question="Something else entirely?")), ByText())
    assert got["answer_relevancy"] == pytest.approx(0.0)


def test_failures_are_recorded_per_metric_and_the_run_goes_on() -> None:
    no_context = Sample(**{**SAMPLE.__dict__, "question_id": "q002", "retrieved_contexts": []})
    metrics = build_metrics(
        router_judge(Scripted(verdicts(), fail=True)), dense_embeddings(SameVector())
    )
    results = asyncio.run(score_samples([SAMPLE, no_context], metrics, concurrency=2))
    assert [r.scores for r in results] == [{}, {}]
    assert set(results[0].errors) == set(metrics)
    assert "RuntimeError" in results[0].errors["faithfulness"]
    assert results[1].errors["context_recall"] == "no contexts retrieved"


# --------------------------------------------------------------------------- judge + cache


def test_router_judge_sends_the_prompt_to_the_judge_alias() -> None:
    llm = Scripted(verdicts())
    scores(llm)
    messages, schema, model = llm.calls[0]
    assert model == "judge" and len(messages) == 1 and messages[0].role == "user"
    assert {name for _, name, _ in llm.calls} >= {"StatementGeneratorOutput", "ContextRecallOutput"}
    other = Scripted(verdicts())
    asyncio.run(router_judge(other, model="openai/gpt-4o").agenerate("p", _Small))
    assert other.calls[0][2] == "openai/gpt-4o"


_Small = StatementGeneratorOutput


def test_cache_spares_repeat_calls_and_keys_on_the_model() -> None:
    cache = JudgeCache()
    llm = Scripted(verdicts())
    judge = cached_judge(router_judge(llm), cache, model="openai/gpt-4o")
    metrics = build_metrics(judge, dense_embeddings(SameVector()))
    first = asyncio.run(score_samples([SAMPLE], metrics))[0].scores
    calls = len(llm.calls)
    second = asyncio.run(score_samples([SAMPLE], metrics))[0].scores
    assert len(llm.calls) == calls and first == second and cache.hits >= calls
    other = cached_judge(router_judge(llm), cache, model="openai/gpt-4.1")
    asyncio.run(score_samples([SAMPLE], build_metrics(other, dense_embeddings(SameVector()))))
    assert len(llm.calls) == 2 * calls


def test_failed_calls_are_not_cached(tmp_path: Path) -> None:
    path = tmp_path / "judge.sqlite"
    failing = cached_judge(
        router_judge(Scripted(verdicts(), fail=True)), JudgeCache(path), model="m"
    )
    with pytest.raises(RuntimeError):
        failing.generate("p", _Small)
    reopened = JudgeCache(path)
    assert len(reopened) == 0
    working = cached_judge(router_judge(Scripted(verdicts())), reopened, model="m")
    reply: object = working.generate("p", _Small)
    assert isinstance(reply, StatementGeneratorOutput) and reply.statements
    assert len(JudgeCache(path)) == 1  # persisted across connections


def test_judge_key_changes_with_model_prompt_and_schema() -> None:
    class Other(BaseModel):
        statements: list[int]

    base = judge_key("m", "p", _Small)
    assert base == judge_key("m", "p", _Small)
    assert (
        len(
            {
                base,
                judge_key("n", "p", _Small),
                judge_key("m", "q", _Small),
                judge_key("m", "p", Other),
            }
        )
        == 4
    )


# --------------------------------------------------------------------------- samples + report


class OneHit:
    def retrieve(self, query: str, *, k: int = 5, **kwargs: Any) -> RetrievalResult:
        chunk = ScoredChunk(id="d1#c0", document_id="d1", text=f"chunk for {query}", score=0.9)
        fact = GraphFact(text="INV-1 -> PO-7 -> no payment")
        return RetrievalResult.build(query, "fake", [chunk], [fact])


def golden(qid: str, category: str) -> GoldenQuestion:
    return GoldenQuestion(
        id=qid,
        category=category,  # type: ignore[arg-type]
        question=f"Question {qid}?",
        ground_truth_answer="the answer",
        relevant_document_ids=()
        if category == "unanswerable"
        else ("0" * 8 + "-0000-4000-8000-" + "0" * 12,),
        requires_graph=False,
        difficulty="easy",
    )


def test_samples_skip_unanswerables_and_keep_graph_facts(tmp_path: Path) -> None:
    questions = [golden("q001", "lookup"), golden("q002", "unanswerable")]
    samples = collect_samples(questions, OneHit(), lambda q, r: f"answer to {q.id}")  # type: ignore[arg-type]
    (sample,) = samples
    assert sample.retrieved_contexts == ["chunk for Question q001?", "INV-1 -> PO-7 -> no payment"]
    assert sample.response == "answer to q001" and sample.reference == "the answer"
    path = tmp_path / "samples.jsonl"
    save_samples(samples, path)
    assert load_samples(path) == samples


def test_summary_and_report() -> None:
    a = run_ragas.Scored(SAMPLE, {"faithfulness": 0.5})
    b = run_ragas.Scored(
        Sample(**{**SAMPLE.__dict__, "question_id": "q2", "category": "clause"}),
        {"faithfulness": 1.0},
        {"context_recall": "Error: key sk-abc12345678 bad"},
    )
    summary = summarize([a, b], ["faithfulness", "context_recall"])
    assert summary["all"]["faithfulness"] == pytest.approx(0.75)
    assert summary["clause"]["context_recall"] is None
    report = render_report([a, b], ["faithfulness", "context_recall"], notes=["judge: `judge`"])
    assert "| all | 2 | 0.750 (2) | — |" in report
    assert "| lookup | 1 | 0.500 (1) | — |" in report
    assert "q2 context_recall" in report and "sk-abc" not in report and "<redacted>" in report


def test_clean_error() -> None:
    assert clean_error("key sk-proj-****NZUA rejected") == "key <redacted> rejected"
    assert clean_error("ask-me-later\nplease") == "ask-me-later please"
    assert len(clean_error("x" * 500)) == 200


# --------------------------------------------------------------------------- CLI


def test_cli_scores_saved_samples_with_the_cached_router_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeClient(Scripted):
        def __init__(self) -> None:
            super().__init__(verdicts())
            self.config = type(
                "C", (), {"resolve": staticmethod(lambda alias: ["openai/gpt-4o"])}
            )()

        def complete(self, *args: Any, **kwargs: Any) -> Any:  # wrapped by the cost meter
            raise AssertionError("structured() is stubbed")

    import ledgerlens.llm.router as router

    clients: list[FakeClient] = []

    def new_client() -> FakeClient:
        clients.append(FakeClient())
        return clients[-1]

    monkeypatch.setattr(router, "LLMClient", new_client)
    monkeypatch.setattr(run_ragas, "dense_embeddings", lambda: dense_embeddings(SameVector()))
    samples, report, cache = tmp_path / "s.jsonl", tmp_path / "ragas.md", tmp_path / "c.sqlite"
    save_samples([SAMPLE], samples)
    argv = ["--samples", str(samples), "--report", str(report), "--cache", str(cache)]
    assert run_ragas.main(argv) == 0
    text = report.read_text(encoding="utf-8")
    assert "judge: `judge`" in text and "| all | 1 | 0.500 (1)" in text
    assert len(clients[0].calls) > 0
    assert run_ragas.main(argv) == 0  # again: everything from the cache
    assert len(clients[1].calls) == 0
    assert ", 0 misses)" in report.read_text(encoding="utf-8")
