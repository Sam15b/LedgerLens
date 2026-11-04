# LedgerLens

**Agentic GraphRAG for invoices, purchase orders and contracts.**

LedgerLens reads invoices, POs, payments and contracts (PDFs and scans), links them in a
knowledge graph, answers questions about them, flags suspicious or duplicate invoices,
and drafts payment reminders.

> 🚧 Early development: Phase 0 (foundation). See the [roadmap](docs/04-implementation-roadmap.md).

## Architecture at a glance

| Layer | What it does | Tech |
|-------|--------------|------|
| Ingestion | OCR, chunking, embeddings | Tesseract / PaddleOCR, PyMuPDF, Hugging Face transformers |
| Storage | Vectors, records, relationships | **Pinecone**, PostgreSQL, Neo4j |
| Retrieval | Hybrid vector search + Cypher graph traversal | GraphRAG |
| Agents | Router → extractor / reconciler / reminder-writer | LangGraph, MCP |
| Model router | One interface over many LLMs | GPT-4o, Gemini, Llama 3 (Ollama), Mistral, AWS Bedrock |
| Fine-tuning | Field extraction model | Llama 3 8B + LoRA (PyTorch, PEFT, CUDA) |
| Synthetic data | Invoice records + degraded scans | PyTorch GAN, Stable Diffusion img2img |
| Anomaly detection | Duplicate / suspicious invoices | TensorFlow / Keras autoencoder |
| Evaluation | Faithfulness, recall@k, F1, latency, cost | RAGAS, LangSmith / Phoenix |
| Deployment | API, dashboard, CI/CD, cloud | FastAPI, Next.js, Docker, GitHub Actions, AWS / GCP |

Full design: **[docs/](docs/README.md)**

## Project structure

```text
src/ledgerlens/   runtime package (ingestion, extraction, storage, retrieval, llm, agents, mcp_server, anomaly, api)
ml/               offline training: fine-tuning, synthetic data, anomaly model
eval/             golden set, RAGAS, retrieval benchmarks, reports
configs/          YAML config (models, Pinecone, ingestion, agents, eval)
scripts/          ops scripts (datasets, index creation, rebuilds)
tests/            unit / integration / e2e
infra/            Dockerfiles, AWS + GCP IaC
web/              Next.js dashboard
docs/             architecture, roadmap, layer guides, ADRs
```

Details: [docs/03-folder-structure.md](docs/03-folder-structure.md)

## Quick start

```bash
git clone https://github.com/Sam15b/LedgerLens.git
cd LedgerLens
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # app deps (API, ingestion, storage, LLMs, agents, eval)
pip install -e . --no-deps         # make the ledgerlens package importable
cp .env.example .env               # fill in API keys

docker compose up -d               # Postgres, Neo4j, Redis, Ollama
uvicorn ledgerlens.api.main:app --reload
curl http://localhost:8000/health
pytest
```

ML training dependencies (LoRA fine-tuning, GAN, diffusion, Keras autoencoder) are in
`requirements-ml.txt` and are meant for a CUDA GPU on Google Colab or Kaggle:

```bash
pip install -r requirements-ml.txt
```

## Development

```bash
make setup          # .venv + app/dev deps + editable install + git hooks
make help           # list every target
```

| Target | What it does |
|--------|--------------|
| `make up` / `make down` | Start / stop local services (Docker) |
| `make api` | FastAPI with reload on `:8000` |
| `make test` | Unit tests |
| `make cov` | Unit tests with a coverage report |
| `make lint` | `ruff check`, `ruff format --check`, `mypy --strict` (src + tests) |
| `make format` | Auto-format and auto-fix lint issues |
| `make clean` | Remove caches and build artifacts |

Targets for later phases (`init`, `data`, `ingest`, `mcp`, `web`, `test-int`, `eval`, `eval-ci`)
already exist and print `not implemented yet` until their phase lands.

On Windows, install GNU make with `choco install make` (or run the commands from Git Bash);
the Makefile picks `.venv/Scripts/python.exe` automatically.

### Pre-commit hooks

`make setup` installs them; otherwise run `pre-commit install` once per clone.
Every commit then runs ruff (lint + format), mypy `--strict`, YAML/TOML checks, a
large-file guard and a private-key detector. Run them on the whole repo with:

```bash
pre-commit run --all-files
```

CI (`.github/workflows/ci.yml`) runs the same lint, format, type and test checks on every
push and pull request.

### Configuration

Secrets and per-environment values come from `.env` (see `.env.example`). Tunable values —
model aliases, chunk sizes, thresholds — live in `configs/*.yaml` and are validated at load
time (`ledgerlens.core.config.get_config()`); unknown keys are rejected.

## Datasets

[SROIE](https://rrc.cvc.uab.es/?ch=13), [CORD](https://github.com/clovaai/cord),
[FUNSD](https://guillaumejaume.github.io/FUNSD/), [CUAD](https://www.atticusprojectai.org/cuad),
plus synthetic linked invoice/PO/payment data.

## Results

Metrics will be filled in from `eval/reports/` as phases complete.

| Metric | Target | Measured |
|--------|--------|----------|
| Faithfulness (RAGAS) | ≥ 0.85 | — |
| Context recall@5 | ≥ 0.90 | — |
| Extraction F1 (SROIE) | ≥ 0.90 | — |
| Anomaly PR-AUC | ≥ 0.85 | — |
| p95 query latency | < 4 s | — |
| Cost per document | < $0.01 | — |
