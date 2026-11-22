# Latency and cost (ci set, 2026-11-22)

40 questions, retriever `R4`; git `b8bbc3a90222`, config `ddb8c4b7de5f`.

## Latency

| | n | mean ms | p50 ms | p95 ms | max ms |
|---|---:|---:|---:|---:|---:|
| retrieval | 40 | 5077 | 4620 | 11322 | 16776 |
| stage: analyze | 40 | 1 | 1 | 2 | 2 |
| stage: embed | 40 | 66 | 64 | 79 | 108 |
| stage: fetch_text | 40 | 5 | 5 | 7 | 10 |
| stage: graph | 40 | 19 | 9 | 74 | 147 |
| stage: rerank | 40 | 4905 | 4473 | 11101 | 16622 |
| stage: search | 40 | 82 | 57 | 147 | 689 |
| stage: total | 40 | 5077 | 4620 | 11322 | 16776 |

## Cost per query

No LLM calls were made for the questions (answers: skipped, AllProvidersFailedError: all models for 'smart' failed (openai/gpt-4o: openai returned HTTP 401: Incorrect API key provided: <redacted>. You can find your API key at https://platform.openai.com/accou…).

## Cost per document (ingestion, from llm_calls)

No recorded ingestion LLM calls (documents ingested with the rules extractor or before calls were recorded per document).
