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

# Placeholder for targets whose implementation lands in a later roadmap phase.
todo = @echo make $@: not implemented yet - $(1)

.DEFAULT_GOAL := help
.PHONY: help setup install-ml up down init data ingest api mcp web \
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

init:             ## Create Pinecone index, Neo4j schema, run DB migrations
	$(call todo,Phase 0/3: index + schema scripts and Alembic migrations)

data:             ## Download public datasets into data/raw
	$(call todo,Phase 0: scripts/download_datasets.py)

ingest:           ## Run ingestion pipeline (DATASET=sroie)
	$(call todo,Phase 1: python -m ledgerlens.ingestion.pipeline data/raw/$(DATASET))

# --- Run -----------------------------------------------------------------------------
api:              ## Run FastAPI with reload on :8000
	$(PY) -m uvicorn ledgerlens.api.main:app --reload --port 8000

mcp:              ## Run the MCP server
	$(call todo,Phase 5: ledgerlens.mcp_server)

web:              ## Run the Next.js dashboard
	$(call todo,Phase 8: web/)

# --- Quality -------------------------------------------------------------------------
test:             ## Unit tests
	$(PY) -m pytest tests/unit

test-int:         ## Integration tests (needs `make up`)
	$(call todo,Phase 0: integration marker + tests/integration)

cov:              ## Unit tests with coverage report
	$(PY) -m pytest tests/unit --cov --cov-report=term-missing

lint:             ## ruff check + format check + mypy --strict
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
	$(PY) -m mypy

format:           ## Auto-format and auto-fix lint issues
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

eval:             ## Full evaluation suite -> eval/reports/<date>/
	$(call todo,Phase 7: eval runner)

eval-ci:          ## 40-question CI evaluation subset
	$(call todo,Phase 7: eval runner)

clean:            ## Remove caches and build artifacts
	$(PY) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for pat in ('.pytest_cache', '.mypy_cache', '.ruff_cache', 'htmlcov', 'build', 'dist', '**/__pycache__') for p in pathlib.Path('.').glob(pat) if '.venv' not in p.parts]"
