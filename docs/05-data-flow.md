# 05 · Data & Request Flows

Four end-to-end flows cover everything the system does.

## Flow 1 — Document ingestion

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant API as FastAPI
    participant OBJ as Object storage
    participant Q as Job queue
    participant W as Ingest worker
    participant MR as Model router
    participant PG as PostgreSQL
    participant PC as Pinecone
    participant NJ as Neo4j
    participant AE as Anomaly scorer

    U->>API: POST /documents (PDF / image)
    API->>API: SHA-256 → already ingested? return existing id
    API->>OBJ: store raw file
    API->>PG: documents(status=queued)
    API->>Q: enqueue ingest(document_id)
    API-->>U: 202 {document_id}
    Q->>W: ingest(document_id)
    W->>W: text layer? → else OCR (Tesseract/Paddle)
    W->>W: classify doc_type (invoice / po / payment / contract / receipt)
    W->>MR: extract fields (fine-tuned Llama or GPT-4o, JSON schema)
    MR-->>W: typed record
    W->>W: normalize (dates, currency, vendor canonical name)
    W->>PG: upsert record + line items + chunks text
    W->>W: chunk → dense + sparse embed
    W->>PC: upsert vectors (id={document_id}#c{n}, metadata)
    W->>NJ: MERGE Vendor/Invoice/PO/Payment/Contract + edges
    W->>AE: score invoice
    AE-->>W: score + reasons
    W->>PG: anomaly_scores, documents(status=ready)
```

**Failure handling:** each step writes `documents.status` + `error`; re-running is safe
because Postgres uses upserts, Pinecone upserts overwrite by id, and Neo4j uses `MERGE`.

## Flow 2 — Question answering (GraphRAG)

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant API as /chat (SSE)
    participant R as Router agent
    participant QA as QA agent
    participant MCP as MCP tools
    participant PC as Pinecone
    participant NJ as Neo4j
    participant MR as Model router

    U->>API: "Which Acme invoices reference POs with no payment?"
    API->>R: LedgerState(messages)
    R->>MR: classify intent → qa (needs_graph=true)
    R->>QA: hand-off
    QA->>MCP: search_documents(query, filter vendor=Acme)
    MCP->>PC: hybrid query top_k=20 → rerank → top 5
    PC-->>QA: chunks + document_ids
    QA->>MCP: graph_query(template=unpaid_po_invoices, seeds=document_ids)
    MCP->>NJ: Cypher
    NJ-->>QA: subgraph rows
    QA->>MR: answer with context + cite [doc ids]
    MR-->>API: streamed tokens
    API-->>U: answer + citations
```

**Routing inside QA:**

| Question shape | Retrieval strategy |
|----------------|--------------------|
| Lookup in text ("payment terms in the Acme contract?") | Hybrid vector search → rerank |
| Relationship / multi-hop ("invoices whose PO has no payment") | Vector seeds → Cypher template expansion |
| Aggregate ("total unpaid by vendor in Q3") | Text2Cypher or SQL analytics tool |

## Flow 3 — Reconciliation & anomaly review

```mermaid
flowchart LR
    A[Invoice ingested] --> B{PO reference found?}
    B -- no --> F1[Flag: missing PO]
    B -- yes --> C[Fetch PO + contract via graph]
    C --> D{Qty / unit price / total<br/>within tolerance?}
    D -- no --> F2[Flag: mismatch + diff report]
    D -- yes --> E{Payment exists?}
    E -- yes, amount ≠ total --> F3[Flag: partial / over-payment]
    E -- no, past due --> G[Add to overdue list]
    E -- yes, matches --> OK[Matched ✓]
    A --> H[Autoencoder + rules score]
    H -- above threshold --> F4[Flag: suspicious / duplicate]
    F1 & F2 & F3 & F4 --> R[Dashboard review queue]
```

## Flow 4 — Payment reminder (human-in-the-loop)

```mermaid
sequenceDiagram
    autonumber
    actor C as Collections user
    participant R as Router agent
    participant RW as Reminder writer
    participant MCP as MCP tools
    participant MR as Model router
    participant PG as PostgreSQL

    C->>R: "Draft reminders for invoices 30+ days overdue"
    R->>RW: hand-off
    RW->>MCP: list_overdue(min_days=30)
    MCP-->>RW: invoices + vendor contacts + history
    loop each invoice
        RW->>MR: draft (tone by days overdue, cite invoice/PO numbers)
        RW->>MCP: draft_reminder(save as status=pending_approval)
    end
    RW-->>C: interrupt() — review drafts
    C->>PG: approve / edit / reject (dashboard)
    Note over PG: v1 marks approved; sending is stubbed
```

## Data lifecycle

| Stage | Location | Retention |
|-------|----------|-----------|
| Raw upload | Object storage (`raw/{document_id}`) | Keep |
| OCR text + page images | `data/interim` (local) / object storage | Regenerable |
| Structured record | PostgreSQL | Source of truth |
| Chunk vectors | Pinecone namespace | Rebuildable from Postgres |
| Graph | Neo4j | Rebuildable from Postgres |
| Traces | LangSmith / Phoenix | 30–90 days |
