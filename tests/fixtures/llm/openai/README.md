OpenAI Chat Completions responses used by `tests/unit/llm/test_openai_provider.py` and
`tests/unit/llm/test_router.py`.

They follow the response format in OpenAI's API reference (`POST /v1/chat/completions`,
plain, structured-output and `stream: true`). Ids and fingerprints are illustrative.
Re-record with a real key, e.g.:

    curl -s https://api.openai.com/v1/chat/completions \
      -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" \
      -d '{"model":"gpt-4o","messages":[{"role":"user","content":"What is a PO? One sentence."}]}'
