# Evaluation summary (ci set, 2026-11-22)

| | |
|---|---|
| questions | 40 (36 answerable) |
| retriever | `R4`, k=5, filters: analyzer, exact hybrid scoring |
| answers | `smart`: skipped, AllProvidersFailedError: all models for 'smart' failed (openai/gpt-4o: openai returned HTTP 401: Incorrect API key provided: <redacted>. You can find your API key at https://platform.openai.com/accou… |
| git | `d33fce718cd39e845b954d01ff90037c4e6eeb6b` |
| config hash | `7e0023bbba5c237def65ef9daecfdb0ad6397e19bfa987320e9bb9609d4f7212` |
| run | `e598cc433245`, eval_runs de2b3823-20ec-5e02-8dbe-fca632152300 |

## Headline metrics

| metric | value | target |
|---|---:|---|
| recall@5 | 0.967 | met |
| mrr | 0.936 |  |
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
| multi_hop | 1.000 |
| reconciliation | 1.000 |

## Latency and cost

| | p50 | p95 |
|---|---:|---:|
| retrieval latency (ms) | 5538 | 11193 |
| cost per query ($) | 0.000000 | 0.000000 |

Total query cost $0.0000; details in `latency_cost_ci.md`.

## Misses (2) and errors (0)

- q037 (aggregation): recall 0.80
- q115 (aggregation): recall 0.00
