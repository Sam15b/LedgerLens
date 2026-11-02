# Layer 9 · Evaluation

**Code:** `eval/`, `src/ledgerlens/observability/` · **Phase:** 4 (v0) → 7 (full)

## Purpose
Turn every claim about LedgerLens into a **reproducible number**: retrieval quality,
answer faithfulness, extraction accuracy, latency and cost. These numbers become the
resume bullets.

## Golden set (`data/golden/golden_set.jsonl`, ~200 Qs)

```json
{"id": "q042",
 "category": "multi_hop",
 "question": "Which Acme invoices reference a PO that has no payment?",
 "ground_truth_answer": "INV-889 and INV-902 …",
 "relevant_document_ids": ["…", "…"],
 "requires_graph": true,
 "difficulty": "medium"}
```

| Category | Count | Example |
|----------|-------|---------|
| Single-doc lookup | 50 | "Total on INV-889?" |
| Contract clause | 35 | "Late-payment penalty in the Acme MSA?" |
| Multi-hop / relationship | 45 | "Invoices whose PO has no payment" |
| Aggregation | 30 | "Unpaid total by vendor in Q1" |
| Reconciliation | 20 | "Does INV-889 match its PO?" |
| Unanswerable | 20 | Questions with no evidence → must say so |

Built from synthetic data (ground truth is known), plus hand-written questions on real
SROIE/CUAD docs. LLM-drafted questions are **human-reviewed** before entering the set.
Keep a 40-question **CI subset** for fast regression checks.

## Metrics

| Metric | Tool | Target |
|--------|------|--------|
| Faithfulness | RAGAS | ≥ 0.85 |
| Answer relevancy | RAGAS | ≥ 0.85 |
| Context precision / recall | RAGAS | ≥ 0.80 / ≥ 0.90 |
| recall@k, MRR | `benchmark_retrieval.py` (vs `relevant_document_ids`) | recall@5 ≥ 0.90 |
| Answer correctness | RAGAS / LLM judge vs ground truth | report per category |
| Extraction F1 | `extraction_f1.py` (field-level) | ≥ 0.90 |
| OCR CER | clean vs degraded | report |
| Anomaly PR-AUC | `ml/anomaly/evaluate.py` | ≥ 0.85 |
| p50 / p95 latency | Traces | p95 < 4 s for QA |
| Cost per document / per query | `llm_calls` table | < $0.01 / doc |
| Router intent accuracy | labeled utterances | ≥ 0.95 |

## Tracing & observability

Use **LangSmith** *or* **Arize Phoenix** (OpenTelemetry-based, self-hostable):
- Every `/chat` request = one trace: router → agent → tool calls → LLM calls.
- Span attributes: provider, model, tokens, cost, retriever variant, top-k ids.
- Eval runs are tagged with `git_sha` and config hash so results are comparable.

## Running

```bash
make eval                       # full suite → eval/reports/2026-11-02/
make eval-ci                    # 40-question subset, used in GitHub Actions
python eval/benchmark_retrieval.py --variants R1,R3,R4,R5 --k 5
```

## Report format (`eval/reports/<date>/summary.md`)

```markdown
| Retriever | recall@5 | faithfulness | answer correctness | p95 ms | $/query |
|-----------|---------:|-------------:|-------------------:|-------:|--------:|
| R1 vector |          |              |                    |        |         |
| R4 GraphRAG |        |              |                    |        |         |
```

Also stored as JSON in `eval_runs` (Postgres) for the dashboard trend chart.

## CI regression gate (`.github/workflows/eval.yml`)
- Triggers on PRs touching `retrieval/`, `agents/`, `prompts/`, `configs/models.yaml`.
- Runs the CI subset against the `eval` Pinecone namespace.
- Fails if faithfulness or recall@5 drops by > 3 points vs `main`'s last report.
- Posts the results table as a PR comment.

## Turning results into resume bullets
Each bullet should cite a measured number from a report, e.g.:
- "GraphRAG improved multi-hop answer correctness from X% to Y% over vector RAG on a 200-question golden set."
- "LoRA-fine-tuned Llama 3 8B reached F1 X on SROIE vs Y for GPT-4o at $0 marginal cost."
- "p95 latency X s, $Y per document across N documents."
