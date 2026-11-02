# LedgerLens Documentation

LedgerLens is an **agentic GraphRAG system for invoices, purchase orders and contracts**.
It reads documents, links them in a knowledge graph, answers questions about them,
flags suspicious invoices, and drafts payment reminders.

> **Vector store decision:** LedgerLens uses **Pinecone** (serverless) for embeddings
> instead of pgvector. PostgreSQL is still used for relational data and SQL analytics,
> and Neo4j holds the document graph. See [ADR-0001](adr/0001-pinecone-for-vector-store.md).

---

## Reading order

| # | Document | What it answers |
|---|----------|-----------------|
| 1 | [Project overview](01-project-overview.md) | What we're building, scope, success metrics, JD keyword map |
| 2 | [Architecture](02-architecture.md) | Components, tech stack, how the stores relate |
| 3 | [Folder structure](03-folder-structure.md) | Where every piece of code lives and why |
| 4 | [Implementation roadmap](04-implementation-roadmap.md) | Phase-by-phase build order, milestones, exit criteria |
| 5 | [Data & request flows](05-data-flow.md) | End-to-end sequence of ingest, query, anomaly, reminder |
| 6 | [Developer workflow](06-dev-workflow.md) | Local setup, env vars, git flow, CI/CD, conventions |

### Layer guides (implementation detail)

| Layer | Guide |
|-------|-------|
| Ingestion (OCR, chunking, embeddings) | [layers/01-ingestion.md](layers/01-ingestion.md) |
| Storage (Pinecone, PostgreSQL, Neo4j) | [layers/02-storage.md](layers/02-storage.md) |
| Retrieval (Vector + GraphRAG) | [layers/03-retrieval.md](layers/03-retrieval.md) |
| Agents (LangGraph + MCP) | [layers/04-agents-mcp.md](layers/04-agents-mcp.md) |
| Model router | [layers/05-model-router.md](layers/05-model-router.md) |
| Fine-tuning (Llama 3 8B LoRA) | [layers/06-fine-tuning.md](layers/06-fine-tuning.md) |
| Synthetic data (GAN + diffusion) | [layers/07-synthetic-data.md](layers/07-synthetic-data.md) |
| Anomaly detection (Keras autoencoder) | [layers/08-anomaly-detection.md](layers/08-anomaly-detection.md) |
| Evaluation (RAGAS, tracing, cost) | [layers/09-evaluation.md](layers/09-evaluation.md) |
| Deployment (Docker, CI/CD, AWS/GCP) | [layers/10-deployment.md](layers/10-deployment.md) |

### Architecture Decision Records

| ADR | Title |
|-----|-------|
| [0001](adr/0001-pinecone-for-vector-store.md) | Use Pinecone as the vector store |

---

## Docs conventions

- One topic per file; link instead of duplicating.
- Diagrams use **Mermaid** (renders on GitHub).
- Every layer guide follows the same template: *Purpose → Inputs/Outputs → Components → Implementation steps → Interfaces → Config → Tests → Definition of done*.
- Significant design choices get an ADR in `docs/adr/` (`NNNN-short-title.md`).
- Numbers in docs are **targets** until measured; measured results live in `eval/reports/`.
