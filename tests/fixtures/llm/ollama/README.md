Ollama `/api/chat` responses used by `tests/unit/llm/test_ollama_provider.py`.

They follow the response format in Ollama's API docs (`docs/api.md`, "Generate a chat
completion"). Re-record from a live server with e.g.:

    curl -s localhost:11434/api/chat -d '{"model":"llama3:8b","stream":false,
      "messages":[{"role":"user","content":"What is an invoice? One sentence."}]}'

`chat_json.json` has no `prompt_eval_count`: Ollama omits it when the prompt is served
from its cache, and the adapter must treat that as 0 input tokens.

`chat_finetuned_receipt.json` is the reply of the fine-tuned extractor
(`ledgerlens-llama3-lora`, `tests/unit/extraction/test_finetuned.py`). It uses the same
response envelope, and its content is the training target for the receipt in
`tests/fixtures/documents/receipt_scanned.json`: the training keys, in the training
order, in the trained value formats. The model has not been trained yet, so this is not
a recording of it. Once `ollama create ledgerlens-llama3-lora` has run, re-record the
fixture with the request the test checks for (`FineTunedExtractor.messages`, `"format": "json"`,
`"options": {"temperature": 0, "num_predict": 768}`).
