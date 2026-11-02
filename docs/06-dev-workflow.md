# 06 · Developer Workflow

## Prerequisites

| Tool | Why |
|------|-----|
| Python 3.11+ and [`uv`](https://docs.astral.sh/uv/) | Backend + ML |
| Node 20+ and `pnpm` | Next.js dashboard |
| Docker Desktop | Postgres, Neo4j, Pinecone Local, Ollama |
| Tesseract OCR | OCR engine (`choco install tesseract` on Windows) |
| Pinecone account | API key for cloud index (Pinecone Local is used for tests) |
| Google Colab / Kaggle | CUDA GPU for fine-tuning + diffusion |

## First-time setup

```bash
git clone <repo> && cd LedgerLens
cp .env.example .env            # fill in keys
uv sync                         # Python deps
pre-commit install
make up                         # docker compose up -d
make init                       # create Pinecone index, Neo4j constraints, run Alembic migrations
make data                       # download SROIE / CORD / FUNSD
make ingest DATASET=sroie       # first ingestion run
```

## Environment variables (`.env.example`)

```dotenv
# --- App ---
APP_ENV=dev
LOG_LEVEL=INFO

# --- PostgreSQL ---
DATABASE_URL=postgresql+psycopg://ledger:ledger@localhost:5432/ledgerlens

# --- Job queue ---
REDIS_URL=redis://localhost:6379/0

# --- Neo4j ---
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=change-me

# --- Pinecone ---
PINECONE_API_KEY=
PINECONE_INDEX=ledgerlens-chunks
PINECONE_CLOUD=aws               # aws | gcp
PINECONE_REGION=us-east-1
PINECONE_NAMESPACE=dev
PINECONE_HOST=                   # set to http://localhost:5081 to use Pinecone Local

# --- LLM providers ---
OPENAI_API_KEY=
GOOGLE_API_KEY=
MISTRAL_API_KEY=
OLLAMA_BASE_URL=http://localhost:11434
AWS_REGION=us-east-1             # Bedrock uses the standard AWS credential chain

# --- Embeddings ---
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DIM=768

# --- Observability ---
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=ledgerlens
PHOENIX_COLLECTOR_ENDPOINT=
```

## Makefile targets

| Target | Does |
|--------|------|
| `make up` / `make down` | Start / stop Docker services |
| `make init` | Pinecone index, Neo4j schema, Alembic migrations |
| `make data` | Download public datasets into `data/raw` |
| `make ingest DATASET=…` | Run ingestion pipeline |
| `make api` | Run FastAPI with reload |
| `make mcp` | Run MCP server (stdio or HTTP) |
| `make web` | Run Next.js dev server |
| `make test` | Unit tests |
| `make test-int` | Integration tests (needs `make up`) |
| `make eval` | Full evaluation → `eval/reports/<date>/` |
| `make lint` | ruff + mypy + eslint |

## Git workflow

- `main` is always deployable and protected (PR + green CI required).
- Branch naming: `feat/<layer>-<short>`, `fix/…`, `exp/…` (experiments), `docs/…`.
  e.g. `feat/retrieval-graphrag`, `exp/finetune-qlora-r16`.
- **Conventional commits:** `feat(retrieval): add cypher expansion for PO lookups`.
- One phase checklist item ≈ one PR. Keep PRs < ~400 lines where possible.
- Tag releases `v0.<phase>.0` at the end of each roadmap phase.

## CI/CD pipeline

```mermaid
flowchart LR
    PR[Pull request] --> L[Lint + mypy]
    L --> UT[Unit tests]
    UT --> IT[Integration tests<br/>compose: postgres, neo4j, pinecone-local]
    IT --> EV{Touches retrieval /<br/>agents / prompts?}
    EV -- yes --> EVAL[Golden-set eval subset<br/>fail on regression]
    EV -- no --> OK[✓]
    EVAL --> OK
    M[Merge to main] --> B[Build + push images]
    T[Tag v*] --> D[Deploy AWS or GCP]
```

## Coding conventions

- **Typed everything**: Pydantic models at boundaries, `mypy --strict` on `src/`.
- **Interfaces first**: each layer exposes a `Protocol` (e.g. `OCREngine`, `Extractor`, `Retriever`, `LLMProvider`) so implementations are swappable and testable.
- **No hard-coded model names / thresholds** — read from `configs/*.yaml`.
- **Prompts are files** in `agents/prompts/`, versioned and referenced by name.
- **Every LLM call** goes through `ledgerlens.llm.router` (cost + tracing).
- **Every DB write is idempotent** (upsert / MERGE).
- **Tests mirror source paths**: `src/ledgerlens/retrieval/graph.py` → `tests/unit/retrieval/test_graph.py`.
- External APIs are mocked in unit tests; integration tests use Docker services and Pinecone Local.

## Experiment tracking

ML experiments (`ml/`) log params + metrics to a simple `ml/<track>/runs/<run_id>.json`
(or Weights & Biases / MLflow if preferred). Every result quoted in the README must link
to a run file or an `eval/reports/` entry.
