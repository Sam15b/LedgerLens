# Changelog

All notable changes to LedgerLens are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions
follow the roadmap: `v0.<phase>.0` marks the end of each phase in
[docs/04-implementation-roadmap.md](docs/04-implementation-roadmap.md).

## [Unreleased]

## [0.5.0] - 2026-11-21

Phase 5 — Agents + MCP: a LangGraph agent graph serves all five user stories of
[docs/01-project-overview.md](docs/01-project-overview.md) through the API. Fourteen MCP
tools are the graph's only way to reach the stores, and the same tools serve any external
MCP client. `tests/e2e/test_stories.py` runs every story through the API against
Postgres, Neo4j, Redis and the arq worker:

- unpaid invoices over $5k for a vendor;
- upload a document and extract it;
- reconcile an invoice;
- review an anomaly flag;
- approve a reminder.

The release also carries the anomaly detection (6c) and fine-tuning (6a) milestones.

### Agents + MCP (Phase 5)

#### Added

- **MCP server** (`ledgerlens.mcp_server`, FastMCP over stdio or streamable HTTP,
  `make mcp`, `make mcp-inspect`). Its 14 tools:
  - search and query: `search_documents` (default retriever), `graph_query` (templates
    or Text2Cypher), `run_analytics`;
  - records: `get_invoice` / `get_po` / `get_contract`, `get_payments`, `list_overdue`,
    `get_vendor_contact`, `get_document_text`, `save_extraction`;
  - checks and drafts: `reconcile`, `score_anomaly`, `draft_reminder`.

  The search models warm up when the server starts.
- **Reconciler** (`ledgerlens.reconcile`): a deterministic 3-way match of invoice, PO,
  contract and payments with tolerances, which reports each check as passed, a mismatch
  or not checked.
- **Agents** (`ledgerlens.agents`). `LedgerState` and a graph that routes on intent:
  - the router uses rules, then a model (`fast`);
  - QA plans its search, graph and analytics calls and answers with enforced citations;
  - the extractor saves its record and reconciles invoices on its own;
  - the reconciler explains the match report;
  - the reminder writer picks the tone from days overdue and pauses with `interrupt()`
    until a person decides.

  Step limit and tool-error guardrails. Postgres checkpointer.
- **API**:
  - `POST /chat` streams SSE events: `token`, `tool_call`, `tool_result`, `citation`,
    `interrupt`, `error`, `done`;
  - `POST /reminders/{id}/approve|reject` resumes the paused run;
  - `POST /documents` stores the upload and queues an ingestion job. A duplicate returns
    200, and the size and type limits answer 413 / 415 / 422;
  - `GET /documents[/{id}]` returns status and extracted fields;
  - `GET /anomalies` and `POST /anomalies/{id}/review` (confirmed / false positive);
  - request ids on every log line and in one error shape.
- Migrations `0002` (`reminders.thread_id`) and `0003` (`anomaly_scores.review`,
  `review_note`, `reviewed_at`).
- Tests:
  - router intent accuracy on a labelled utterance set;
  - one reconciler test per mismatch type;
  - agent trajectory tests;
  - the SSE event sequence;
  - the API endpoints;
  - upload → worker → `ready`;
  - the five end-to-end stories.
- [docs/layers/04-agents-mcp.md](docs/layers/04-agents-mcp.md) explains how to connect
  Claude Desktop, Claude Code, MCP Inspector or the Python SDK. Both transports were
  checked with an external client.

#### Changed

- The reminder writer keeps the overdue threshold a request sets ("30+ days overdue").
  Before, it drafted every overdue invoice.
- Run uvicorn on a selector event loop (`make api`), because psycopg's async mode does
  not run on Windows' Proactor loop.
- Package version 0.5.0.

#### Known issues

- No working LLM key here, so every model path has only run with scripted models or
  fallen back to rules or keywords:
  - router model accuracy;
  - QA planning and answers;
  - reconciler narration;
  - LLM extraction;
  - Text2Cypher.

  On fresh utterances the router's rules alone score 0.50.
- `token` events are pieces of finished answers: model output is not streamed into the
  graph yet.
- The end-to-end stories use a stand-in with no hits for vector search: they are answered
  from the graph and Postgres.
- The arq worker needs POSIX (Docker/WSL). The API and the worker must share
  `loader.upload_dir`.
- The rules extractor reads the fixture invoice's header fields but not its line items.

### Anomaly detection milestone (Phase 6c: rules + Keras autoencoder)

Every invoice the stores sink writes is scored for anomalies, with reasons a reviewer can
read. On the held-out test split (the latest 750 synthetic invoices, 66 injected anomalies)
the combination of rules and autoencoder reaches **PR-AUC 0.830**. Rules alone reach
0.661, the autoencoder 0.394 and Isolation Forest 0.200
(`eval/reports/2026-11-20/anomaly.md`). The roadmap's other 6c exit item, the
`score_anomaly` MCP tool, lands with the Phase 5 tools.

#### Added

- `ledgerlens.anomaly.features` and `ml/anomaly/build_features.py`: 19 features per
  invoice (amount, timing, structure, vendor history before it), with optional PCA text
  features. Train / val / test are split by time, and the scaler is fit on training
  normals only.
- `ledgerlens.anomaly.rules`: exact and near duplicates (with a `POSSIBLE_DUPLICATE`
  edge in Neo4j), over-PO, contract-rate breach and unknown vendor, each with a one-line
  reason.
- `ml/anomaly/train_autoencoder.py`: a 64-32-8-32-64 Keras 3 autoencoder trained on
  normal invoices with early stopping. The threshold is chosen on the labelled val split
  (precision ≥ 0.8). Model, scaler, threshold and metadata are versioned in
  `models/anomaly/<version>/`, and the reloaded model reproduces the scores.
- `ml/anomaly/evaluate.py`: PR-AUC, ROC-AUC, precision@k and recall per anomaly type for
  the rules, Isolation Forest, the autoencoder and rules + AE. The rules run over the
  ledger in arrival order. The rules + AE threshold is chosen on val (`--save-combined`).
- `ledgerlens.anomaly.scorer`: rule hits are hard hits that rank first, and the
  autoencoder orders invoices within each band. The model is loaded once per worker from
  `configs/anomaly.yaml` (newest version by default, rules only without a model or
  Keras). Reasons: rule sentences, the autoencoder's top-3 features in plain language
  ("total higher than this vendor's usual (z = +3.7)"), or "nothing unusual".
- Ingestion: `StoresSink` scores every invoice it writes into `anomaly_scores` (score and
  reasons per model version). The context is read from Postgres: the vendor's invoices
  stored before it, the PO balance, the contract and the vendor master. Scoring failures
  are logged, not fatal. `python -m ledgerlens.anomaly.scorer --missing` scores whatever
  is still unscored.

#### Changed

- `scikit-learn` joins the `anomaly` extra (Isolation Forest baseline).
- Rule reasons print amounts at two decimals (Postgres returns four).

#### Known issues

- No PCA text features yet: they need the header-chunk embeddings exported from
  Pinecone.
- Labels are synthetic. Most rule misses and false alarms on them are near-duplicate
  copies dated before the invoice they copy, so the rules flag the pair on the original.
  The pair is still caught.

### Fine-tuning milestone (Phase 6a: Llama 3 8B + LoRA), tooling only

Everything needed to train, evaluate, export and serve a fine-tuned extraction model, run
end to end on CPU with a tiny random Llama. **The model itself has not been trained**:
the dev machine has no GPU and no access to the gated Llama 3 weights. So there is no
fine-tuned F1 yet. The Phase 6a gate (F1 above the rules baseline, 0.601) is still open,
and this milestone gets no version.

#### Added

- `ml/finetune/prepare_dataset.py`: instruction data from CORD, SROIE (when downloaded)
  and the synthetic invoices. Splits are leak-free by source document, the encoding uses
  the Llama 3 chat template, and the loss covers response tokens only. 1,016 / 130 / 130
  examples without SROIE.
- `ml/finetune/train_lora.py`: QLoRA on a 16 GB GPU (4-bit NF4, paged 8-bit AdamW,
  gradient checkpointing). It reads a YAML config with `--set` overrides, resumes from
  checkpoints on Drive, keeps the best checkpoint by validation loss with early stopping,
  and logs each run to `runs/<id>/`. Comes with a Colab notebook.
- Runs 1 and 2, pinned but untrained: `runs/run-1` (r 16, lr 2e-4, 3 epochs) and
  `runs/run-2` (r 32, lr 1e-4, 2 epochs). Each has its config, dataset hashes and notes.
- `ml/finetune/infer.py` and `eval_extraction.py`: adapter inference, then field F1,
  JSON validity, latency and tokens for rules, API, base and LoRA models with one
  harness (`eval/reports/2026-11-18/finetune.md`).
- `ml/finetune/merge_and_export.py`: merges the adapter into the float16 base, then
  produces a GGUF through llama.cpp, quantized to Q4_K_M, with an `export.json`
  manifest.
- `ml/finetune/Modelfile` for `ollama create ledgerlens-llama3-lora`: Llama 3 template,
  the training system prompt, greedy decoding.
- `ledgerlens.extraction.finetuned.FineTunedExtractor` (`extraction.default: finetuned`,
  `pipeline --extractor finetuned`). It sends the training prompt through the `extractor`
  alias in JSON mode, maps the reply onto the record schema, and passes purchase orders,
  payments and contracts to the LLM extractor.

#### Changed

- `configs/models.yaml`: `extractor` is now `[ollama/ledgerlens-llama3-lora,
  openai/gpt-4o-mini, gemini/gemini-3.8-flash, mistral/mistral-small-2603]`. The new
  `extractor_api` alias (the old list) is what `LLMExtractor` uses
  (`extraction.llm_alias`), so its schema prompt never reaches the fine-tuned model.
- The fine-tuned prompt contract (system prompt, key lists, instruction) lives in
  `ledgerlens.extraction.finetuned`, and the training data builder imports it from
  there. The rebuilt dataset is byte-identical.

#### Known issues

- No training run, so no adapter, no GGUF and no fine-tuned or base-model F1. The steps
  are in `ml/finetune/runs/run-1/NOTES.md` and `run-2/NOTES.md`.
- GPT-4o row of the comparison not measured: it needs a working OpenAI key and about
  130 paid calls.
- llama.cpp's converter and `llama-quantize` have only been exercised through unit tests
  with a fake runner.
- `extraction.default` stays `llm` until the fine-tuned model passes the gate.

## [0.4.0] - 2026-11-18

Phase 4 — Retrieval: five retrievers behind one `Retriever` protocol, from plain vector
search to GraphRAG with Text2Cypher, measured on a reviewed golden set. On golden set v0
the default, R4 GraphRAG, reaches recall@5 0.889, MRR 0.904 and evidence recall 0.953,
against 0.649 / 0.633 / 0.649 for dense vector search
([eval/reports/2026-11-18/retrieval.md](eval/reports/2026-11-18/retrieval.md)).

### Added

- `ledgerlens.retrieval`, the retriever variants. Each returns cited chunks, graph facts
  and per-stage timings:
  - **R1** dense search, with full chunk text from Postgres and metadata filters;
  - **R2** α-weighted hybrid dense + BM25;
  - **R3** R2's candidates reranked by a cross-encoder (`BAAI/bge-reranker-base`);
  - **R4** GraphRAG: R3's documents expanded through parameterized Cypher templates
    (invoice neighbourhood, unpaid with PO, contract for PO, vendor ledger) into compact,
    citable facts, plus budgeted context assembly and a grounded answer prompt;
  - **R5** Text2Cypher: LLM-written read-only Cypher with guardrails and one retry, and
    R4 as the fallback.
- A query analyzer: document numbers, vendors, date ranges and intent, with an LLM
  fallback. Its filters and the graph template choice come from it.
- `ledgerlens.retrieval.factory`: builds any variant from `configs/`, and
  `build_retriever()` returns the default.
- `configs/retrieval.yaml`: default retriever, rerank model and graph expansion
  (`max_hops`, `per_seed`, `limit`).
- Evaluation:
  - golden set v0 (`data/golden/golden_set.jsonl`): 51 reviewed questions over the
    synthetic ledger in six categories, with a schema, a builder and CI validation;
  - `eval/benchmark_retrieval.py`: recall@k, MRR, evidence recall, per-category tables,
    latency and, with `--answers`, answer correctness, citations and cost;
  - `eval/tune_retrieval.py`: one-knob sweeps (hybrid α, rerank candidates, graph hops)
    with exact client-side hybrid scoring for Pinecone Local.
- Reports in `eval/reports/2026-11-18/`: `tuning_alpha.md`, `tuning_rerank_graph.md` and
  `retrieval.md`.

### Changed

- Tuned on golden set v0:
  - `pinecone.query_top_k` (rerank candidates) 20 → 50: R3 recall@5 0.836 → 0.868;
  - `retrieval.graph.max_hops` 2 → 1: same evidence, 4.5 → 2.6 facts per question;
  - `pinecone.hybrid_alpha` stays 0.7 (best for R3 and R4).
- Retrievers accept any `VectorIndex`, not only `PineconeStore`.
- Package version 0.4.0.

### Known issues

- R5 and the answer columns (correctness, citations, cost) are not measured yet: they need
  a working LLM key.
- Pinecone Local ignores sparse vectors, so the hybrid numbers come from exact
  client-side scoring (`--exact-hybrid`). Re-check them on the serverless index.
- Aggregation questions stay at recall@5 0.756 for every variant: totals come from the
  graph, but the golden evidence lists every invoice in the total.
- On CPU the cross-encoder dominates the latency tail (p95 ~7.5 s with 50 candidates).

## [0.3.0] - 2026-11-16

Phase 3 — Storage: every document is stored three ways, all keyed by the same
`document_id`: records in Postgres, chunk vectors in Pinecone and relationships in Neo4j.
One idempotent write path fills them. The synthetic ledger loads into all three with
matching counts, and Pinecone and Neo4j can be rebuilt from Postgres alone. This release
also contains the synthetic data milestone below, which was finished before Phase 3 closed.

### Storage (Phase 3)

#### Added

- PostgreSQL (`storage/postgres/`):
  - ORM models for documents, chunks, vendors, invoices, POs, payments with allocations,
    contracts, anomaly scores, reminders, `llm_calls` and `eval_runs`;
  - engine and session management, and Alembic with `make migrate` (also run by
    `make init`);
  - idempotent repositories: upserts whose update only fires on a real change;
  - payments matched to invoices by number, with invoice status refreshed from them;
  - Postgres-backed SHA-256 dedupe, and LLM call costs written to `llm_calls`.
- AP analytics in `analytics.sql`, with typed Python wrappers: AP aging
  (current / 0-30 / 31-60 / 61-90 / 90+), spend by vendor by month, unpaid totals and
  average days to pay. Every report is per currency and point-in-time.
- Pinecone (`storage/pinecone_store.py`):
  - batched upsert with filterable metadata (`doc_date` as `YYYYMMDD` ints);
  - α-weighted hybrid dense + sparse query;
  - delete by document prefix and stats;
  - one namespace per environment (`dev` / `eval` / `prod`);
  - tests against Pinecone Local.
- Neo4j (`storage/neo4j/`):
  - driver with read and write sessions, and a read-only runner for untrusted Cypher;
  - vendor resolution by tax id, then canonical name or alias, then fuzzy match;
  - a MERGE-based linker for Vendor → Invoice → PO ← Contract and Payment → Invoice;
  - `UnresolvedLink` nodes for references with no target yet.
- `ingestion/sinks.py` `StoresSink`, the write path:
  - Postgres transaction → Pinecone delete + upsert → Neo4j MERGE → `ready`;
  - any failure leaves the document `failed` with the step and the error, and the next
    run retries it;
  - re-ingesting is a no-op.
- `workers/ingest_worker.py`: a Redis (arq) queue worker for ingestion jobs (`make worker`,
  `make enqueue`).
- `scripts/reindex_pinecone.py` and `scripts/rebuild_graph.py` rebuild the vectors (BM25
  refit, stale vectors pruned) and the graph (aliases kept, `--wipe`) from Postgres.
- `scripts/load_synthetic.py` (`storage/bulk_load.py`):
  - records go straight into Postgres and Neo4j;
  - rendered PDFs go through the pipeline and attach to those records;
  - the script exits non-zero if store counts disagree.
- `scripts/seed_demo_data.py`: a small demo slice of the ledger with every invoice state
  and anomaly kind.
- `eval/reports/2026-11-16/storage.md`: the full synthetic ledger loaded with matching
  counts across stores (14,291 records, 950 documents, 1,667 chunks = 1,667 vectors),
  with per-stage timings and full-scale rebuilds.

#### Changed

- The pipeline hands results to a `StorageSink` instead of writing JSONL; `JsonlSink`
  keeps the offline mode.
- `IngestionPipeline.process(doc_type=...)` skips classification when the type is known.
- `StoresSink` has `record_id` / `resolve_vendor` hooks, used by the bulk loader.
- Package version 0.3.0.

#### Known issues

- Payments that name a number shared by exact-duplicate invoices stay unmatched: 171 of
  4,491 allocations on the synthetic set. Left for the reconciler.
- The rules classifier puts the compact invoice layout ("BILL NO.") below its threshold.

### Synthetic data milestone (Phase 6b: GAN + diffusion)

A labelled synthetic ledger, rendered to real-looking documents and degraded scans, so
ingestion, extraction and (next) anomaly detection can be measured against known truth.

#### Added

- `ml/synthetic/seed.py`: GAN seed rows from CORD receipts and a hand-made vendor master
  (50 vendors, 10 categories, 7 countries) with seed POs.
- Tabular **WGAN-GP** (`ml/synthetic/gan/`):
  - reversible preprocessing (log amounts, min-max, one-hot);
  - a generator with Gumbel-softmax categorical heads and a critic with gradient penalty;
  - optional `vendor_category` conditioning;
  - a training loop with checkpoints and `run.json`, plus a Colab notebook.
- `gan/sample.py`: samples with business-rule post-processing (currency rounding,
  `total = subtotal × (1 + tax_rate)` recomputed). `gan/evaluate.py`: per-column KS and
  Wasserstein, TVD, correlation-matrix difference, all against a real-vs-real baseline.
- `ml/synthetic/link.py`, `anomalies.py`, `build.py`:
  - a contract (rate card, terms) per vendor, one PO per invoice, and payments in paid /
    partial / unpaid / over-paid states;
  - six labelled anomaly types at exact rates: exact and near duplicate, price inflation,
    round amount, unknown vendor, rush terms;
  - `python -m ml.synthetic.build` writes `data/synthetic/records.jsonl`.
- `ml/synthetic/render/`:
  - Jinja2 templates: three invoice layouts, purchase order, remittance advice and contract;
  - rendering to PDF (WeasyPrint, or PyMuPDF's Story on Windows) plus page images;
  - output is deterministic, and exact duplicates are byte-identical.
- `ml/synthetic/diffusion/`:
  - classical scan augmentations (albumentations; one severity knob);
  - Stable Diffusion img2img degradation (SD 1.5, fp16, strength 0.2–0.45, fixed seeds)
    with a Colab notebook;
  - an OCR validity filter that keeps a variant only if every key field is read exactly.
- Evaluation:
  - `eval/extraction_synthetic.py` ingests rendered PDFs and scores extraction per type,
    layout and field;
  - `eval/ocr_robustness.py` measures CER, extraction F1 and key-field readability on
    clean vs degraded scans per OCR engine, with a strength → CER/F1 curve.
- Reports in `eval/reports/`:
  - GAN quality: 0 % rule violations; the conditional GAN's amounts are at the real-data
    baseline;
  - extraction on 950 rendered PDFs: 0 ingestion failures;
  - degraded-scan CER table: clean 0.011 → 0.288 at full severity.
- Extras: `synthetic` (jinja2, albumentations; installed in CI); matplotlib in `eval`.

#### Fixed

- Text-layer extraction expands ligatures. "ﬁles" now reads as "files", as label and
  vendor matching expect.
- Line grouping compares row centre lines instead of growing boxes. A tall heading no
  longer merges the two lines beside it ("Northwind 1 Market Street Traders …").
- `first_line_vendor` stops at a document title printed on the name's line. PO
  `buyer_name` F1 went from 0.00 to 1.00 on rendered POs.

#### Changed

- OCR preprocessing runs the median denoise only on speckled pages
  (`ocr.preprocess.denoise_min_speckle`), and before deskew. Synthetic invoice CER went
  from 0.050 to 0.011 on clean pages and from 0.426 to 0.288 at full severity. The
  speckled receipt fixture went from 0.053 to 0.064, within its 0.10 budget.

#### Not done yet

- The Stable Diffusion run on a GPU, and with it the SD strength → CER curve.
- PaddleOCR in the robustness benchmark: no wheels for the local environment.

## [0.1.0] - 2026-11-12

Phase 1 — Ingestion + extraction baseline: a PDF or scanned image goes in, a typed
record, retrieval chunks and their vectors come out, with the time each step took.

Version numbers follow roadmap phases, and Phase 2 (0.2.0, the model router) was built
first so extraction could use it; this release therefore comes after 0.2.0, and the
package version stays `0.2.0`.

### Added

- Domain schemas: `Document` (status lifecycle) and the extracted records `Invoice`,
  `Receipt`, `PurchaseOrder`, `Payment`, `Contract`, as one `ExtractedDocument` union keyed
  on `doc_type`; validation rejects malformed values, `consistency_issues()` reports
  arithmetic that does not add up.
- `ingestion/loaders.py`: SHA-256 dedupe (the same bytes under any name are ingested once),
  MIME sniffed from content, page counts, size/type limits from `ingestion.yaml`.
- Five sample documents that tell one story (contract → PO → invoice → payment, plus a
  scanned cafe receipt), each with the JSON it should extract to.
- `ingestion/layout.py`: per-page text-layer detection; words → lines → blocks → tables
  (markdown) by geometry, shared by the text-layer and OCR paths; scanned pages rasterized
  at 300 DPI.
- OCR: `OCREngine` protocol, Tesseract (deskew → denoise → upscale before OCR; CER 0.053
  on the scanned receipt), optional PaddleOCR, engine chosen by `ocr.engine`.
- `ingestion/classify.py`: doc_type from weighted keyword rules, LLM (`fast` alias) when
  unsure.
- `ingestion/chunking.py`: header + line-item chunks for invoices/receipts/POs, clause
  chunks for contracts (512 tokens, 50 overlap, embedding-model tokenizer), one chunk for
  payments; every chunk prefixed with `[INVOICE INV-889 | Vendor: … | Date: …]`.
- `ingestion/embedding.py`: dense `bge-base-en-v1.5` (normalized, query instruction,
  cached model, length-sorted batches) and BM25 sparse vectors with persisted params;
  `EmbeddedChunk` records ready for Pinecone.
- Extraction: `Extractor` protocol; normalizers for dates (day-first configurable),
  amounts, currencies and vendor names (`rapidfuzz` matching); the regex baseline
  `RulesExtractor`; `LLMExtractor` with the versioned `extract_invoice` prompt and
  structured output through the router.
- `ingestion/pipeline.py`: load → layout/OCR → classify → extract → chunk → embed with
  per-step timings; `python -m ledgerlens.ingestion.pipeline <folder>` / `make ingest`
  writes `documents.jsonl` and `chunks.jsonl`.
- Evaluation: character error rate and an OCR benchmark runner, a SROIE adapter, and
  field-level extraction F1 with JSON-validity rate.

### Changed

- PyMuPDF, Pillow, NumPy, OpenCV (headless), pytesseract and rapidfuzz are core
  dependencies; `pinecone-text` is no longer used (BM25 is implemented in-house without
  NLTK downloads).
- CI installs Tesseract and CPU-only PyTorch, so OCR and a tiny embedding model run for
  real in tests.
- `ocr.preprocess.binarize` defaults to `false` (Tesseract binarizes better itself).

## [0.2.0] - 2026-11-11

Phase 2 — Model router: one interface for every LLM. Callers ask for an alias (`smart`,
`fast`, `local`, `extractor`, `judge`) and the router falls back across OpenAI, Gemini,
Mistral, AWS Bedrock and Ollama, pricing and recording every call. Built before Phase 1
(see the daily plan) so extraction can use it from the start.

### Added

- `ledgerlens.llm` types: `Message`, `LLMRequest`, `LLMResponse`, `Usage`, the
  `LLMProvider` protocol, and errors split by what the router does next (retry, fall back,
  or raise).
- Provider adapters with recorded-response contract tests:
  - OpenAI and Mistral (Chat Completions, strict JSON-schema structured output),
  - Gemini (`generateContent`, `responseJsonSchema`, thought tokens billed as output),
  - Ollama (local Llama 3, `format` JSON schema, $0),
  - AWS Bedrock (Converse API via boto3, standard credential chain, structured output
    with the schema rewritten to Bedrock's subset).
- `LLMClient.complete()`: alias → ordered fallback chain; transient errors retried with the
  backoff from `models.yaml` (`Retry-After` honoured, capped at 30 s), provider errors fall
  back, validation errors are raised.
- `LLMClient.structured()`: returns a validated Pydantic instance, with one repair attempt
  on the model that answered. The same schema parses from all five providers.
- `LLMClient.stream()`: text deltas with retry/fallback until the first chunk; usage,
  cost and latency recorded when the stream ends.
- `configs/models.yaml`: aliases across all five providers and a USD-per-1M-token price
  table (checked against provider pricing pages on 2026-11-11).
- `observability/cost.py`: in-process cost ledger recording every call (alias, provider,
  model, tokens, cost, latency, request id) to `data/interim/llm_calls.jsonl`.
- `scripts/llm_smoke.py`: live smoke test — the same extraction and streaming prompts
  through every configured model, with a latency / tokens / $ table.

### Changed

- Package version bumped to `0.2.0`.
- `boto3` added to the `dev` extra so CI runs the Bedrock contract tests.

## [0.0.0] - 2026-11-04

Phase 0 — Foundation: an empty-but-working skeleton where every service starts with one
command and reports its health.

### Added

- Project scaffold: package layout, design docs, layer guides and ADR-0001 (Pinecone).
- `requirements.txt` (app) and `requirements-ml.txt` (GPU training) dependency sets.
- Typed configuration: `.env` settings plus `configs/*.yaml` validated into Pydantic
  sections (unknown keys rejected) via `get_config()`.
- Structured logging (pretty in dev, JSON elsewhere) with a per-request id.
- Stable ids shared by all stores: UUID4 `document_id` and `{document_id}#c{n}` vector ids.
- `docker-compose.yml` with Postgres, Neo4j, Redis, Pinecone Local and Ollama, all with
  healthchecks; `POSTGRES_PORT` to avoid clashing with a local Postgres.
- `GET /health` (per-dependency status, latency and errors) and `GET /ready` (503 until
  every required dependency is up).
- `scripts/create_pinecone_index.py`: idempotent index creation for Pinecone cloud or
  Pinecone Local; refuses to touch an index with a different shape.
- `schema.cypher` constraints/indexes and `scripts/init_neo4j_schema.py`.
- `scripts/download_datasets.py` for FUNSD, CUAD, CORD v2 and SROIE (manual instructions),
  with a `data/raw/manifest.json` record; interrupted downloads resume instead of being
  mistaken for complete datasets.
- Tooling: pre-commit (ruff, ruff-format, mypy `--strict`), Makefile with `make help`.
- CI: lint, format check, strict type check, unit tests with coverage, and an integration
  job against the compose services.

### Changed

- Package version is read from `src/ledgerlens/__init__.py` (single source) and reset to
  `0.0.0` to match the roadmap's `v0.<phase>.0` scheme.

[Unreleased]: https://github.com/Sam15b/LedgerLens/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/Sam15b/LedgerLens/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Sam15b/LedgerLens/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Sam15b/LedgerLens/compare/v0.1.0...v0.3.0
[0.1.0]: https://github.com/Sam15b/LedgerLens/compare/v0.2.0...v0.1.0
[0.2.0]: https://github.com/Sam15b/LedgerLens/compare/v0.0.0...v0.2.0
[0.0.0]: https://github.com/Sam15b/LedgerLens/releases/tag/v0.0.0
