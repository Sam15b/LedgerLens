# Retrieval benchmark (51 questions)

- **date:** 2026-11-18
- **golden set:** `data/golden/golden_set.jsonl` v0 (51 questions)
- **k:** 5
- **answers:** not generated
- **hybrid scoring:** exact, client-side (Pinecone Local ignores sparse vectors)
- **filters:** query analyzer (same for every variant)
- **settings:** `hybrid_alpha` 0.7, rerank candidates (`query_top_k`) 50, graph `max_hops` 1 ([tuning_alpha.md](tuning_alpha.md), [tuning_rerank_graph.md](tuning_rerank_graph.md))
- **stores:** docker compose dev stores with the Day 32 synthetic load (Pinecone Local, namespace `dev`)
- **command:** `PINECONE_HOST=http://localhost:5080 python -m eval.benchmark_retrieval --variants R1,R2,R3,R4 --k 5 --exact-hybrid --report eval/reports/2026-11-18/retrieval.md`

## Overall

| variant | retriever | recall@5 | MRR | evidence recall | p50 ms | p95 ms | degraded | errors |
|---|---|---|---|---|---|---|---|---|
| R1 | dense | 0.649 | 0.633 | 0.649 | 120 | 184 | 0 | 0 |
| R2 | hybrid (dense + BM25) | 0.777 | 0.798 | 0.777 | 73 | 133 | 0 | 0 |
| R3 | hybrid + cross-encoder rerank | 0.868 | 0.901 | 0.868 | 611 | 7,623 | 0 | 0 |
| R4 | GraphRAG (R3 seeds + graph templates) | 0.889 | 0.904 | 0.953 | 697 | 7,428 | 0 | 0 |

## recall@5 by category

| questions | n | R1 | R2 | R3 | R4 |
|---|---|---|---|---|---|
| lookup | 12 | 0.833 | **1.000** | **1.000** | **1.000** |
| clause | 8 | **1.000** | **1.000** | **1.000** | **1.000** |
| multi_hop | 12 | 0.375 | 0.708 | 0.875 | **0.958** |
| aggregation | 9 | 0.667 | 0.667 | **0.756** | **0.756** |
| reconciliation | 6 | 0.333 | 0.333 | **0.583** | **0.583** |
| needs graph | 27 | 0.463 | 0.611 | 0.770 | **0.807** |
| no graph | 20 | 0.900 | **1.000** | **1.000** | **1.000** |
| all | 47 | 0.649 | 0.777 | 0.868 | **0.889** |

## MRR by category

| questions | n | R1 | R2 | R3 | R4 |
|---|---|---|---|---|---|
| lookup | 12 | 0.833 | **1.000** | 0.958 | 0.958 |
| clause | 8 | **1.000** | **1.000** | **1.000** | **1.000** |
| multi_hop | 12 | 0.312 | 0.792 | **0.944** | **0.944** |
| aggregation | 9 | 0.667 | 0.667 | **0.722** | **0.722** |
| reconciliation | 6 | 0.333 | 0.333 | 0.833 | **0.861** |
| needs graph | 27 | 0.435 | 0.648 | 0.846 | **0.852** |
| no graph | 20 | 0.900 | **1.000** | 0.975 | 0.975 |
| all | 47 | 0.633 | 0.798 | 0.901 | **0.904** |

## Evidence recall by category (chunks + graph facts)

| questions | n | R1 | R2 | R3 | R4 |
|---|---|---|---|---|---|
| lookup | 12 | 0.833 | **1.000** | **1.000** | **1.000** |
| clause | 8 | **1.000** | **1.000** | **1.000** | **1.000** |
| multi_hop | 12 | 0.375 | 0.708 | 0.875 | **1.000** |
| aggregation | 9 | 0.667 | 0.667 | **0.756** | **0.756** |
| reconciliation | 6 | 0.333 | 0.333 | 0.583 | **1.000** |
| needs graph | 27 | 0.463 | 0.611 | 0.770 | **0.919** |
| no graph | 20 | 0.900 | **1.000** | **1.000** | **1.000** |
| all | 47 | 0.649 | 0.777 | 0.868 | **0.953** |

## Notes

- recall@5 and MRR rank documents as the retriever returns them: chunk documents first, then the documents graph facts cite. Evidence recall counts every document in the result.
- 4 unanswerable question(s) have no relevant documents: they count towards latency but not towards recall or MRR.
- *degraded*: questions where a stage failed and the retriever fell back (Neo4j for R4, Text2Cypher for R5). *errors*: the retriever raised.

## Vector vs hybrid vs GraphRAG

Phase 4 exit: what each stage adds on golden set v0, with the tuned settings.

| | recall@5 | MRR | evidence recall | needs-graph recall@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| vector (R1) | 0.649 | 0.633 | 0.649 | 0.463 | 120 | 184 |
| hybrid (R2) | 0.777 | 0.798 | 0.777 | 0.611 | 73 | 133 |
| hybrid + rerank (R3) | 0.868 | 0.901 | 0.868 | 0.770 | 611 | 7,623 |
| GraphRAG (R4) | **0.889** | **0.904** | **0.953** | **0.807** | 697 | 7,428 |

- **Hybrid over vector, +0.128 recall@5.** BM25 matches the exact document numbers that
  dense embeddings blur: lookup goes 0.833 -> 1.000 and multi-hop 0.375 -> 0.708. The
  2026-11-17 report had R2 = R1 because Pinecone Local drops sparse values; the hybrid
  scores here are computed exactly (`--exact-hybrid`). Expect the same on the serverless
  index, which scores sparse values itself.
- **Reranking, +0.091.** With 50 candidates instead of 20, the cross-encoder lifts the
  right document into the top 5 for multi-hop (0.708 -> 0.875) and reconciliation
  (0.333 -> 0.583) questions. It costs ~0.5 s at p50 and up to ~7.5 s at p95 on CPU.
- **GraphRAG, +0.085 evidence recall.** The graph facts bring multi-hop and reconciliation
  evidence to 1.000 and needs-graph evidence to 0.919, at ~0.1 s per question for the
  graph. Its recall@5 (0.889) is above R3's because fact-cited documents fill top-5
  slots that chunks leave free.
- **Against the 2026-11-17 baseline** (alpha 0.7 unscored, 20 candidates, 2 hops): R3
  recall@5 0.698 -> 0.868, R4 MRR 0.738 -> 0.904; R4 evidence recall stays 0.953 with
  fewer facts per question (1 hop).
- **Still open:** aggregation stays at 0.756 for every variant (the totals come from the
  graph, but the golden evidence lists every invoice in the total), and R5 plus the
  answer columns need a working LLM key.

**Default:** R4 (GraphRAG): best on every column, at about the same latency as R3.
