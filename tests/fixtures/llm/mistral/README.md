Mistral Chat Completions responses used by `tests/unit/llm/providers/test_mistral_provider.py`
and `tests/unit/llm/test_structured_contract.py`.

They follow the response format in Mistral's API reference (`POST /v1/chat/completions`:
plain, `response_format: json_schema`, and `stream: true`). Ids are illustrative. Re-record
with a real key, e.g.:

    curl -s https://api.mistral.ai/v1/chat/completions \
      -H "Authorization: Bearer $MISTRAL_API_KEY" -H "Content-Type: application/json" \
      -d '{"model":"mistral-small-2603","messages":[{"role":"user","content":"What is a PO? One sentence."}]}'
