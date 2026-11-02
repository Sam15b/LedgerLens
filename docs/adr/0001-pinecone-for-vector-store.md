# ADR-0001 · Use Pinecone as the vector store

- **Status:** Accepted
- **Date:** 2026-11-02
- **Supersedes:** original plan to store embeddings in pgvector

## Context
The original design stored embeddings in **pgvector** inside PostgreSQL. LedgerLens needs:
hybrid (semantic + keyword) search because invoices are number- and code-heavy
(`INV-889`, `PO-1042`); metadata filtering by vendor, doc type and date; isolation between
dev / eval / prod data; and minimal operational work during a short build.

## Decision
Use **Pinecone serverless** for chunk embeddings:
- One index `ledgerlens-chunks`, dimension 768 (`bge-base-en-v1.5`), metric `dotproduct`.
- Dense + sparse vectors per chunk for hybrid search.
- Namespaces for environment isolation (`dev`, `eval`, `prod`; later per tenant).
- **PostgreSQL remains the system of record** (records, chunk text, analytics); **Neo4j** remains the graph store.
- Pinecone Local emulator for tests/CI.

## Consequences

**Positive**
- Managed, scales without tuning HNSW/IVF parameters.
- Native sparse-dense hybrid queries and metadata filters.
- Namespaces give clean eval isolation.
- Cloud-agnostic: works with both the AWS and GCP deployment paths.
- Adds "Pinecone" (a widely listed vector DB) to the skill set.

**Negative / mitigations**
| Trade-off | Mitigation |
|-----------|-----------|
| Data lives in two places (Postgres + Pinecone) → consistency risk | Shared `document_id`; idempotent write path; `reindex_pinecone.py` rebuilds from Postgres |
| No SQL joins between vectors and records | Store filterable fields in metadata; join in application code by `document_id` |
| External dependency + API key | Pinecone Local for offline dev/CI; retries in `pinecone_store.py` |
| Metadata size limits | Keep metadata small; full text in Postgres |
| Cost at scale | Serverless free/starter tier sufficient for demo volumes; monitor usage |

## Alternatives considered
- **pgvector** — simplest (one DB) but hybrid search and filtering require more manual work.
- **Neo4j vector index** — would co-locate with the graph, but couples retrieval scaling to the graph DB.
- **Qdrant / Weaviate** — capable, but self-hosting adds ops work.
