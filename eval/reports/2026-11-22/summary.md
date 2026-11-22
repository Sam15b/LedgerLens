# Evaluation summary (full set, 2026-11-22)

| | |
|---|---|
| questions | 200 (180 answerable) |
| retriever | `R4`, k=5, filters: analyzer, exact hybrid scoring |
| answers | `smart`: skipped, AllProvidersFailedError: all models for 'smart' failed (openai/gpt-4o: openai returned HTTP 401: Incorrect API key provided: <redacted>. You can find your API key at https://platform.openai.com/accou… |
| git | `d33fce718cd39e845b954d01ff90037c4e6eeb6b` |
| config hash | `b82859432edb89b7c212a91741b63a7a72feace921c7992f4b0ff5ac445b8af9` |
| run | `37a65def7869`, eval_runs f8b7184f-a46f-59f4-bf51-adec8f853c3c |

## Headline metrics

| metric | value | target |
|---|---:|---|
| recall@5 | 0.954 | met |
| mrr | 0.929 |  |
| evidence_recall | 0.954 |  |
| facts_found | not measured |  |
| citations_ok | not measured |  |
| faithfulness | not measured |  |

## recall@5 by category

| category | recall |
|---|---:|
| aggregation | 0.793 |
| clause | 0.971 |
| lookup | 0.980 |
| multi_hop | 1.000 |
| reconciliation | 1.000 |

## Latency and cost

| | p50 | p95 |
|---|---:|---:|
| retrieval latency (ms) | 6296 | 13525 |
| cost per query ($) | 0.000000 | 0.000000 |

Total query cost $0.0000; details in `latency_cost.md`.

## Misses (9) and errors (0)

- q037 (aggregation): recall 0.80
- q040 (aggregation): recall 0.00
- q041 (aggregation): recall 0.00
- q107 (aggregation): recall 0.00
- q115 (aggregation): recall 0.00
- q122 (aggregation): recall 0.00
- q123 (aggregation): recall 0.00
- q171 (lookup): recall 0.00
- q198 (clause): recall 0.00
