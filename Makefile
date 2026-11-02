.PHONY: setup install up down api test lint format

setup:            ## Create venv and install dev deps
	python -m venv .venv
	.venv/bin/pip install -e ".[dev]"

install:          ## Install all optional dependency groups
	pip install -e ".[dev,ingestion,storage,llm,agents,eval]"

up:               ## Start local services
	docker compose up -d

down:
	docker compose down

api:              ## Run FastAPI with reload
	uvicorn ledgerlens.api.main:app --reload --port 8000

test:
	pytest

lint:
	ruff check src tests
	mypy src

format:
	ruff format src tests
	ruff check --fix src tests
