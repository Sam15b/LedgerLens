# Retrieval benchmark (51 questions)

- **date:** 2026-11-17
- **golden set:** `data/golden/golden_set.jsonl` v0 (51 questions)
- **stores:** docker compose dev stores with the Day 32 synthetic load (Pinecone Local, namespace `dev`)
- **k:** 5
- **answers:** not generated (no working LLM key in this environment)
- **filters:** query analyzer (same for every variant)
- **command:** `PINECONE_HOST=http://localhost:5080 python -m eval.benchmark_retrieval --variants R1,R2,R3,R4 --k 5 --report eval/reports/2026-11-17/retrieval.md`

## Overall

| variant | retriever | recall@5 | MRR | evidence recall | p50 ms | p95 ms | degraded | errors |
|---|---|---|---|---|---|---|---|---|
| R1 | dense | 0.649 | 0.633 | 0.649 | 115 | 143 | 0 | 0 |
| R2 | hybrid (dense + BM25) | 0.649 | 0.633 | 0.649 | 110 | 165 | 0 | 0 |
| R3 | hybrid + cross-encoder rerank | 0.698 | 0.699 | 0.698 | 649 | 3,530 | 0 | 0 |
| R4 | GraphRAG (R3 seeds + graph templates) | 0.698 | 0.738 | 0.953 | 678 | 3,507 | 0 | 0 |

## recall@5 by category

| questions | n | R1 | R2 | R3 | R4 |
|---|---|---|---|---|---|
| lookup | 12 | **0.833** | **0.833** | **0.833** | **0.833** |
| clause | 8 | **1.000** | **1.000** | **1.000** | **1.000** |
| multi_hop | 12 | 0.375 | 0.375 | **0.500** | **0.500** |
| aggregation | 9 | 0.667 | 0.667 | **0.756** | **0.756** |
| reconciliation | 6 | **0.333** | **0.333** | **0.333** | **0.333** |
| needs graph | 27 | 0.463 | 0.463 | **0.548** | **0.548** |
| no graph | 20 | **0.900** | **0.900** | **0.900** | **0.900** |
| all | 47 | 0.649 | 0.649 | **0.698** | **0.698** |

## MRR by category

| questions | n | R1 | R2 | R3 | R4 |
|---|---|---|---|---|---|
| lookup | 12 | 0.833 | 0.833 | 0.833 | **0.861** |
| clause | 8 | **1.000** | **1.000** | **1.000** | **1.000** |
| multi_hop | 12 | 0.312 | 0.312 | 0.528 | **0.597** |
| aggregation | 9 | 0.667 | 0.667 | **0.722** | **0.722** |
| reconciliation | 6 | 0.333 | 0.333 | 0.333 | **0.444** |
| needs graph | 27 | 0.435 | 0.435 | 0.549 | **0.605** |
| no graph | 20 | 0.900 | 0.900 | 0.900 | **0.917** |
| all | 47 | 0.633 | 0.633 | 0.699 | **0.738** |

## Evidence recall by category (chunks + graph facts)

| questions | n | R1 | R2 | R3 | R4 |
|---|---|---|---|---|---|
| lookup | 12 | 0.833 | 0.833 | 0.833 | **1.000** |
| clause | 8 | **1.000** | **1.000** | **1.000** | **1.000** |
| multi_hop | 12 | 0.375 | 0.375 | 0.500 | **1.000** |
| aggregation | 9 | 0.667 | 0.667 | **0.756** | **0.756** |
| reconciliation | 6 | 0.333 | 0.333 | 0.333 | **1.000** |
| needs graph | 27 | 0.463 | 0.463 | 0.548 | **0.919** |
| no graph | 20 | 0.900 | 0.900 | 0.900 | **1.000** |
| all | 47 | 0.649 | 0.649 | 0.698 | **0.953** |

## Notes

- recall@5 and MRR rank documents as the retriever returns them: chunk documents first, then the documents graph facts cite. Evidence recall counts every document in the result.
- 4 unanswerable question(s) have no relevant documents: they count towards latency but not towards recall or MRR.
- *degraded*: questions where a stage failed and the retriever fell back (Neo4j for R4, Text2Cypher for R5). *errors*: the retriever raised.

## Reading the numbers

- **R2 = R1** here because Pinecone Local ignores sparse values: on the serverless index
  R2 adds BM25 matching. Re-run against the cloud index before choosing between them.
- **R3 > R1** on the questions that need the graph (recall@5 0.463 -> 0.548, MRR 0.435 ->
  0.549): the cross-encoder lifts the right invoice into the top 5 more often. It costs
  ~0.5 s per question at p50 on CPU.
- **R4 = R3 on recall@5** by construction: R4 returns R3's chunks unchanged, and the
  documents its graph facts cite rank after them, so they fall outside the top 5. What
  the graph adds shows in **evidence recall** (all documents in the result, which is
  what the answer step sees): 0.698 -> **0.953**, with multi-hop 0.500 -> **1.000** and
  reconciliation 0.333 -> **1.000**. It also lifts **MRR** (0.699 -> 0.738), because a fact
  can cite a relevant document that no chunk found.
- **Aggregation stays at 0.756** for R4: the template answers totals from the graph, but
  the golden evidence lists every rendered invoice in the total, and the facts cite only
  the vendor's contract. Whether R5's Text2Cypher rows close this gap needs an LLM key.
- **Not measured yet:** R5 and the answer columns (answer correctness, citations, latency,
  cost) need a working LLM key: OpenAI rejected the configured key (HTTP 401), Bedrock's
  token is invalid and no Gemini key is set. Re-run with `--variants R1,R2,R3,R4,R5
  --answers` once one works.
