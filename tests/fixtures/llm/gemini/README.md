Gemini API responses used by `tests/unit/llm/providers/test_gemini_provider.py` and
`tests/unit/llm/test_structured_contract.py`.

They follow the response format in the Gemini API reference
(`POST /v1beta/models/{model}:generateContent`, `:streamGenerateContent?alt=sse`, and a 429
`RESOURCE_EXHAUSTED` error carrying `RetryInfo`). Ids are illustrative. Re-record with a
real key, e.g.:

    curl -s "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent" \
      -H "x-goog-api-key: $GOOGLE_API_KEY" -H "Content-Type: application/json" \
      -d '{"contents":[{"role":"user","parts":[{"text":"What is a PO? One sentence."}]}]}'
