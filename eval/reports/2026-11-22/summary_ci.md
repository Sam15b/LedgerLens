# Evaluation summary (ci set, 2026-11-22)

| | |
|---|---|
| questions | 40 (36 answerable) |
| retriever | `R4`, k=5, filters: analyzer, exact hybrid scoring |
| answers | `smart`: skipped, AllProvidersFailedError: all models for 'smart' failed (openai/gpt-4o: openai returned HTTP 401: Incorrect API key provided: <redacted>. You can find your API key at https://platform.openai.com/accou… |
| git | `b8bbc3a90222691cf7c3961aba0d5b9b8a4b970a` |
| config hash | `ddb8c4b7de5f1c0c2a437565c3a10173261517574ae7dcf69167c0ee0ff11c53` |
| run | `21b011ca3904`, eval_runs 60e67623-7713-5291-8ead-5b2a9558be71 |

## Headline metrics

| metric | value | target |
|---|---:|---|
| recall@5 | 0.869 | below 0.90 |
| mrr | 0.894 |  |
| evidence_recall | 0.967 |  |
| facts_found | not measured |  |
| citations_ok | not measured |  |
| faithfulness | not measured |  |

## recall@5 by category

| category | recall |
|---|---:|
| aggregation | 0.800 |
| clause | 1.000 |
| lookup | 1.000 |
| multi_hop | 0.778 |
| reconciliation | 0.625 |

## Latency and cost

| | p50 | p95 |
|---|---:|---:|
| retrieval latency (ms) | 4620 | 11322 |
| cost per query ($) | 0.000000 | 0.000000 |

Total query cost $0.0000; details in `latency_cost_ci.md`.

## Misses (7) and errors (0)

- q037 (aggregation): recall 0.80
- q042 (reconciliation): recall 0.00
- q078 (multi_hop): recall 0.50
- q087 (multi_hop): recall 0.00
- q096 (multi_hop): recall 0.50
- q115 (aggregation): recall 0.00
- q129 (reconciliation): recall 0.50
