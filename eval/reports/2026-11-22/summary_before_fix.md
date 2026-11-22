# Evaluation summary (full set, 2026-11-22)

| | |
|---|---|
| questions | 200 (180 answerable) |
| retriever | `R4`, k=5, filters: analyzer, exact hybrid scoring |
| answers | `smart`: skipped, AllProvidersFailedError: all models for 'smart' failed (openai/gpt-4o: openai returned HTTP 401: Incorrect API key provided: <redacted>. You can find your API key at https://platform.openai.com/accou… |
| git | `6c527c96f188cfe4c274b9175eb01377ccc5fc3b` |
| config hash | `b82859432edb89b7c212a91741b63a7a72feace921c7992f4b0ff5ac445b8af9` |
| run | `8e5960a963ef`, eval_runs 50097f99-a654-5ee9-a7c1-f465cb6fa4f3 |

## Headline metrics

| metric | value | target |
|---|---:|---|
| recall@5 | 0.882 | below 0.90 |
| mrr | 0.899 |  |
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
| multi_hop | 0.833 |
| reconciliation | 0.725 |

## Latency and cost

| | p50 | p95 |
|---|---:|---:|
| retrieval latency (ms) | 5673 | 13355 |
| cost per query ($) | 0.000000 | 0.000000 |

Total query cost $0.0000; details in `latency_cost.md`.

## Misses (29) and errors (0)

- q026 (multi_hop): recall 0.50
- q037 (aggregation): recall 0.80
- q040 (aggregation): recall 0.00
- q041 (aggregation): recall 0.00
- q042 (reconciliation): recall 0.00
- q043 (reconciliation): recall 0.50
- q044 (reconciliation): recall 0.50
- q045 (reconciliation): recall 0.50
- q072 (multi_hop): recall 0.50
- q075 (multi_hop): recall 0.00
- q078 (multi_hop): recall 0.50
- q081 (multi_hop): recall 0.00
- q087 (multi_hop): recall 0.00
- q090 (multi_hop): recall 0.50
- q093 (multi_hop): recall 0.00
- q096 (multi_hop): recall 0.50
- q099 (multi_hop): recall 0.00
- q107 (aggregation): recall 0.00
- q115 (aggregation): recall 0.00
- q122 (aggregation): recall 0.00
- q123 (aggregation): recall 0.00
- q126 (reconciliation): recall 0.50
- q129 (reconciliation): recall 0.50
- q130 (reconciliation): recall 0.50
- q133 (reconciliation): recall 0.50
- q134 (reconciliation): recall 0.50
- q139 (reconciliation): recall 0.50
- q171 (lookup): recall 0.00
- q198 (clause): recall 0.00
