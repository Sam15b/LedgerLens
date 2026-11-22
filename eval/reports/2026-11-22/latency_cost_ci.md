# Latency and cost (ci set, 2026-11-22)

40 questions, retriever `R4`; git `d33fce718cd3`, config `7e0023bbba5c`.

## Latency

| | n | mean ms | p50 ms | p95 ms | max ms |
|---|---:|---:|---:|---:|---:|
| retrieval | 40 | 5378 | 5538 | 11193 | 16359 |
| stage: analyze | 40 | 1 | 1 | 2 | 2 |
| stage: anchor | 11 | 1125 | 1174 | 1381 | 1420 |
| stage: embed | 40 | 65 | 63 | 76 | 127 |
| stage: fetch_text | 40 | 4 | 4 | 7 | 7 |
| stage: graph | 40 | 11 | 6 | 39 | 61 |
| stage: rerank | 40 | 4904 | 4594 | 10979 | 16203 |
| stage: search | 40 | 82 | 59 | 159 | 370 |
| stage: total | 40 | 5378 | 5538 | 11193 | 16359 |

## Cost per query

No LLM calls were made for the questions (answers: skipped, AllProvidersFailedError: all models for 'smart' failed (openai/gpt-4o: openai returned HTTP 401: Incorrect API key provided: <redacted>. You can find your API key at https://platform.openai.com/accou…).

## Cost per document (ingestion, from llm_calls)

No recorded ingestion LLM calls (documents ingested with the rules extractor or before calls were recorded per document).
