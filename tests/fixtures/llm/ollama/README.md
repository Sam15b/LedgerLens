Ollama `/api/chat` responses used by `tests/unit/llm/test_ollama_provider.py`.

They follow the response format in Ollama's API docs (`docs/api.md`, "Generate a chat
completion"). Re-record from a live server with e.g.:

    curl -s localhost:11434/api/chat -d '{"model":"llama3:8b","stream":false,
      "messages":[{"role":"user","content":"What is an invoice? One sentence."}]}'

`chat_json.json` has no `prompt_eval_count`: Ollama omits it when the prompt is served
from its cache, and the adapter must treat that as 0 input tokens.
