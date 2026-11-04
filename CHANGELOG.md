# Changelog

All notable changes to LedgerLens are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions
follow the roadmap: `v0.<phase>.0` marks the end of each phase in
[docs/04-implementation-roadmap.md](docs/04-implementation-roadmap.md).

## [Unreleased]

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

[Unreleased]: https://github.com/Sam15b/LedgerLens/compare/v0.0.0...HEAD
[0.0.0]: https://github.com/Sam15b/LedgerLens/releases/tag/v0.0.0
