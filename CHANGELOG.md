# Changelog

All notable changes to LedgerLens are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions
follow the roadmap: `v0.<phase>.0` marks the end of each phase in
[docs/04-implementation-roadmap.md](docs/04-implementation-roadmap.md).

## [Unreleased]

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

[Unreleased]: https://github.com/Sam15b/LedgerLens/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Sam15b/LedgerLens/compare/v0.2.0...v0.1.0
[0.2.0]: https://github.com/Sam15b/LedgerLens/compare/v0.0.0...v0.2.0
[0.0.0]: https://github.com/Sam15b/LedgerLens/releases/tag/v0.0.0
