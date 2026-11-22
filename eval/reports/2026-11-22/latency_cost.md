# Latency and cost (full set, 2026-11-22)

200 questions, retriever `R4`; git `d33fce718cd3`, config `b82859432edb`.

## Latency

| | n | mean ms | p50 ms | p95 ms | max ms |
|---|---:|---:|---:|---:|---:|
| retrieval | 200 | 5488 | 6296 | 13525 | 17903 |
| stage: analyze | 200 | 1 | 1 | 2 | 2 |
| stage: anchor | 59 | 1029 | 1092 | 1402 | 1673 |
| stage: embed | 200 | 63 | 61 | 77 | 111 |
| stage: fetch_text | 200 | 4 | 5 | 6 | 7 |
| stage: graph | 200 | 11 | 6 | 45 | 68 |
| stage: rerank | 200 | 5037 | 5462 | 13368 | 17730 |
| stage: search | 200 | 67 | 58 | 118 | 397 |
| stage: total | 200 | 5488 | 6296 | 13525 | 17903 |

## Cost per query

No LLM calls were made for the questions (answers: skipped, AllProvidersFailedError: all models for 'smart' failed (openai/gpt-4o: openai returned HTTP 401: Incorrect API key provided: <redacted>. You can find your API key at https://platform.openai.com/accou…).

## Cost per document (ingestion, from llm_calls)

No recorded ingestion LLM calls (documents ingested with the rules extractor or before calls were recorded per document).
