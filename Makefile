# LedgerLens developer commands. Run `make help` for the list.
# Works with GNU make on Linux/macOS and on Windows (choco install make, or Git Bash).

ifeq ($(OS),Windows_NT)
VENV_PY := .venv/Scripts/python.exe
else
VENV_PY := .venv/bin/python
endif
# Use the project venv when it exists, otherwise whatever `python` is on PATH.
PY ?= $(if $(wildcard $(VENV_PY)),$(VENV_PY),python)

DATASET ?= sroie
MCP_PORT ?= 8765
EVAL_ARGS ?=

# Placeholder for targets whose implementation lands in a later roadmap phase.
todo = @echo make $@: not implemented yet - $(1)

.DEFAULT_GOAL := help
.PHONY: help setup install-ml up down init migrate data ingest worker enqueue api mcp mcp-inspect web \
        test test-int cov lint format eval eval-ci clean

help:             ## Show this help
	@$(PY) -c "import re; [print(f'{n:<12} {d}') for n, d in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open('Makefile').read(), re.M)]"

# --- Setup ---------------------------------------------------------------------------
setup:            ## Create .venv, install app + dev deps, install git hooks
	python -m venv .venv
	$(VENV_PY) -m pip install -r requirements.txt
	$(VENV_PY) -m pip install -e . --no-deps
	$(VENV_PY) -m pre_commit install

install-ml:       ## Install GPU/ML training deps (Colab / Kaggle)
	$(PY) -m pip install -r requirements-ml.txt

# --- Local services ------------------------------------------------------------------
up:               ## Start local services (Postgres, Neo4j, Redis, Ollama)
	docker compose up -d

down:             ## Stop local services
	docker compose down

init: migrate     ## Run DB migrations, create Pinecone index and Neo4j schema
	$(PY) scripts/create_pinecone_index.py
	$(PY) scripts/init_neo4j_schema.py

migrate:          ## Upgrade Postgres to the latest Alembic migration
	$(PY) -m alembic upgrade head

data:             ## Download public datasets into data/raw (DATASETS="funsd cuad" for a subset)
	$(PY) scripts/download_datasets.py $(DATASETS)

ingest:           ## Ingest data/raw/$(DATASET) into the stores (ARGS="--limit 100" or "--sink jsonl")
	$(PY) -m ledgerlens.ingestion.pipeline data/raw/$(DATASET) $(ARGS)

worker:           ## Run the ingestion worker (Redis queue; needs make up && make init)
	$(PY) -m ledgerlens.workers.ingest_worker run

enqueue:          ## Queue data/raw/$(DATASET) for the ingestion worker
	$(PY) -m ledgerlens.workers.ingest_worker enqueue data/raw/$(DATASET)

# --- Run -----------------------------------------------------------------------------
api:              ## Run FastAPI with reload on :8000
	$(PY) -m uvicorn ledgerlens.api.main:app --reload --port 8000 --loop asyncio:SelectorEventLoop

mcp:              ## Run the MCP server over streamable HTTP on :$(MCP_PORT)/mcp (MCP_PORT=8765)
	$(PY) -m ledgerlens.mcp_server.server --transport http --port $(MCP_PORT)

mcp-inspect:      ## Open the MCP Inspector on the server over stdio (needs Node.js / npx)
	npx @modelcontextprotocol/inspector $(PY) -m ledgerlens.mcp_server.server

web:              ## Run the Next.js dashboard
	$(call todo,Phase 8: web/)

# --- Quality -------------------------------------------------------------------------
test:             ## Unit tests
	$(PY) -m pytest tests/unit

test-int:         ## Integration tests (needs `make up`; PINECONE_HOST=http://localhost:5080 for Pinecone Local)
	$(PY) -m pytest -m integration tests/integration

cov:              ## Unit tests with coverage report
	$(PY) -m pytest tests/unit --cov --cov-report=term-missing

lint:             ## ruff check + format check + mypy --strict
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
	$(PY) -m mypy

format:           ## Auto-format and auto-fix lint issues
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

eval:             ## Full evaluation suite -> eval/reports/<date>/summary.md (EVAL_ARGS="--no-answers")
	$(PY) -m eval.runner --set full $(EVAL_ARGS)

eval-ci:          ## 40-question CI subset -> eval/reports/<date>/summary_ci.md
	$(PY) -m eval.runner --set ci $(EVAL_ARGS)

clean:            ## Remove caches and build artifacts
	$(PY) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for pat in ('.pytest_cache', '.mypy_cache', '.ruff_cache', 'htmlcov', 'build', 'dist', '**/__pycache__') for p in pathlib.Path('.').glob(pat) if '.venv' not in p.parts]"
