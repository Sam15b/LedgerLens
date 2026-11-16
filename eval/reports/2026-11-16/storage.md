# Storage: synthetic ingest counts and timings — 2026-11-16

This is the Phase 3 exit run. It loads the full synthetic ledger into Postgres, Neo4j and Pinecone, checks that the three stores agree, and rebuilds the derived stores from Postgres.

**Data:** `data/synthetic/records.jsonl` (GAN run `20261113-131707-seed1`) holds 50 vendors, 50 contracts, 4,750 POs, 5,000 invoices and 4,491 payments. The 950 PDFs listed in `pdf/manifest.jsonl` were rendered from those records: 300 invoices, 300 POs, 300 payment advices and 50 contracts.

**Stores:** `docker compose` Postgres 16, Neo4j 5 and Pinecone Local (namespace `dev`).

**Machine:** a 12-thread AMD laptop CPU, with no GPU.

**Dense model:** `BAAI/bge-base-en-v1.5` (768-d) on the CPU. BM25 was fit on the loaded chunks.

```
docker compose up -d --wait && make init
PINECONE_HOST=http://localhost:5080 python scripts/load_synthetic.py --report eval/reports/2026-11-16/storage.json
python scripts/rebuild_graph.py
PINECONE_HOST=http://localhost:5080 python scripts/reindex_pinecone.py
```

## Parity

Postgres rows match Neo4j nodes, and Postgres chunks match Pinecone vectors (`describe_index_stats`):

| | Postgres | Neo4j | Pinecone |
|---|---:|---:|---:|
| Vendors | 58 | 58 | |
| Contracts | 50 | 50 | |
| Purchase orders | 4,750 | 4,750 | |
| Invoices | 5,000 | 5,000 | |
| Payments | 4,491 | 4,491 | |
| Line items | (in `*_line_items`) | 20,075 | |
| Documents (all `ready`) | 950 | | |
| Chunks / vectors | 1,667 | | 1,667 |

- **Vendors:** 58 = the 50 master vendors plus 8 created for the `unknown_vendor` invoices and payments. The loader resolved those 8 the same way ingestion does, by tax id, then name, then fuzzy match.
- **Graph links:** every reference found its target, so there are 0 `UnresolvedLink` nodes. There are 4,950 `REFERENCES` edges (the 50 unknown-vendor invoices have no PO) and 4,491 `SETTLES` edges.
- **Document attachment:** every rendered document attached to the record the structured pass wrote. 300 invoices, 300 POs, 300 payments and 50 contracts now have a `document_id`. None of them got a second copy.

## Timings

### Structured pass

Records go straight to Postgres and Neo4j, with no OCR and no extraction:

| Stage | First load | Re-run (no-op) |
|---|---:|---:|
| Plan vendors (in memory) | 0.04 s | 0.04 s |
| Postgres: 14,291 records in 500-record transactions | 93.0 s | 95.0 s |
| Neo4j: 58 vendors, then 14,291 `link` calls | 136.5 s | 171.7 s |

The re-run did the same work and changed nothing: every write is an upsert or a MERGE whose update only fires on a real change.

### Rendered documents

950 PDFs go through the pipeline, with the known record standing in for extraction:

| Stage | Wall time | Median per document |
|---|---:|---:|
| Load (sha256 dedupe against Postgres) | | 14.6 ms |
| Layout (text layer, no OCR needed) | | 4.2 ms |
| Chunk | | 0.1 ms |
| Process total (all of the above) | 24.5 s | |
| Embed: dense, 1,635 chunks, CPU | 243.2 s | 256.8 ms |
| Store: Postgres, then Pinecone delete + upsert, then Neo4j | 85.8 s | |
| **Total** | **353.7 s** | |

20 PDFs had been loaded by a trial run beforehand. 16 of them came back as sha256 duplicates. The other 4 were re-processed (see "Findings"). On the CPU, embedding takes about 70 % of the document path.

### Rebuilds from Postgres alone

Day 31's scripts, run at full scale:

| Script | Result | Time |
|---|---|---:|
| `rebuild_graph.py` | 58 vendors and 14,291 records merged. Nodes 34,424 → 34,424 and relationships 39,316 → 39,316, identical by label and type. | 197.5 s |
| `reindex_pinecone.py` | 950 documents re-embedded into 1,667 vectors, 0 pruned. BM25 refit on the 1,667 stored chunks. | 301.8 s |

In the integration tests, `rebuild_graph` restored a dropped graph exactly, and `reindex_pinecone` restored dropped vectors exactly (`tests/integration/test_write_path.py`).

## Findings

- **Payment matching by number is ambiguous for duplicate invoices.** 171 of 4,491 payment allocations stay unmatched (`invoice_id` is NULL). Exact-duplicate invoices reuse their source's number, so 100 (vendor, invoice number) pairs belong to two invoices, and `PaymentRepository` refuses to guess between them.
  - As a result, Postgres invoice statuses (3,857 paid, 463 partially paid, 680 unpaid) differ from the generator's labels (4,009 paid or overpaid, 482 partial, 509 unpaid).
  - This is the case the reconciler (Phase 5) has to resolve. It is not a loader bug.
- **The rules classifier misses the compact invoice layout.** On text alone, all 83 `invoice_compact` invoices fall below the confidence threshold and are classified `unknown`: the layout prints "BILL NO." and "AMOUNT PAYABLE", never "INVOICE". The other two layouts are classified correctly (217/217), and so are all POs, payments and contracts.
  - The bulk loader now passes the known type (`IngestionPipeline.process(doc_type=...)`), so it is unaffected.
  - Plain ingestion without an LLM would still drop these to `unknown`. This is a follow-up for `ingestion/classify.py`.
- **Pinecone Local is dense-only.** It stores sparse values but does not score them, so hybrid weighting cannot be measured on this run (docs/layers/02-storage.md). The vector counts are unaffected.
