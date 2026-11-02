# 01 · Project Overview

## Problem

Accounts-payable teams juggle invoices, purchase orders (POs), payments and contracts
spread across PDFs, scans and email attachments. Answering "Has vendor X been paid for
PO-1042?" or "Which invoices exceed their contract rate?" requires manually cross-referencing
several documents. Duplicate and fraudulent invoices slip through, and overdue payments
are chased by hand.

## What LedgerLens does

1. **Ingests** PDFs and scanned documents → OCR → chunks → embeddings.
2. **Extracts** structured fields (vendor, invoice no., dates, totals, line items, PO refs).
3. **Links** documents in a knowledge graph: `Vendor → Invoice → PO → Payment`, `Contract → PO`.
4. **Answers questions** using GraphRAG (Pinecone vector search + Neo4j Cypher traversal).
5. **Reconciles** invoices against POs and payments (3-way match).
6. **Flags** duplicate / suspicious invoices with an autoencoder + rules.
7. **Drafts** payment reminders for overdue invoices (human-approved before sending).

## Scope

| In scope | Out of scope (v1) |
|----------|-------------------|
| English invoices, receipts, POs, contracts | Multi-language OCR beyond what datasets require |
| Single-tenant demo with namespace-ready design | Full multi-tenant billing / auth provider integration |
| Drafting reminders | Actually emailing vendors (stubbed behind approval) |
| Batch + on-demand ingestion | Real-time ERP integration (SAP, NetSuite) |

## Users & core use cases

| User | Use case | Agent path |
|------|----------|------------|
| AP clerk | "Show unpaid invoices from Acme over $5k" | Router → QA (GraphRAG) |
| AP clerk | Upload a new invoice and get its fields | Router → Extractor |
| Finance controller | "Does INV-889 match its PO and contract rate?" | Router → Reconciler |
| Finance controller | Review flagged invoices | Anomaly service → dashboard |
| Collections | "Draft reminders for invoices 30+ days overdue" | Router → Reminder writer |

## Datasets

| Dataset | Content | Used for |
|---------|---------|----------|
| **SROIE** (ICDAR 2019) | ~1k scanned receipts, 4 key fields | OCR + field-extraction F1 |
| **CORD v2** | ~1k receipts with line-item labels | Line-item extraction, fine-tuning data |
| **FUNSD** | ~200 noisy scanned forms | Layout/OCR robustness, key-value linking |
| **CUAD** *(recommended add)* | ~500 annotated commercial contracts | Contract clause retrieval & QA |
| **Synthetic** (GAN + templates) | Invoices, POs, payments with known links | Graph, reconciliation, anomaly labels |

> SROIE/CORD/FUNSD are receipts/forms, not full AP document sets. The synthetic generator
> ([layers/07](layers/07-synthetic-data.md)) produces linked invoice/PO/payment records with
> ground-truth relationships and injected anomalies, which the graph, reconciler and
> anomaly detector need.

## Success metrics (targets → measured in `eval/reports/`)

| Metric | Target | Measured by |
|--------|--------|-------------|
| Answer faithfulness | ≥ 0.85 | RAGAS |
| Context recall@5 | ≥ 0.90 | Golden set (≈200 Qs) |
| GraphRAG vs vector-only (multi-hop Qs) | +15 pts answer correctness | Retrieval benchmark |
| Field extraction F1 (SROIE) | ≥ 0.90 (fine-tuned) | `eval/extraction` |
| Anomaly detection | PR-AUC ≥ 0.85 on injected anomalies | `ml/anomaly` eval |
| p95 query latency | < 4 s | LangSmith / Phoenix traces |
| Cost per document ingested | < $0.01 | Token + API cost logger |

## JD keyword coverage

| Layer | Keywords |
|-------|----------|
| Ingestion | Python, NLP, transformer architecture, unstructured data |
| Storage | Vector databases (**Pinecone**), Neo4j, SQL |
| Retrieval | RAG, GraphRAG |
| Agents | Agentic workflows, LangGraph, MCP |
| Model router | GPT-4o, Gemini, Llama 3 (Ollama), Mistral, AWS Bedrock |
| Fine-tuning | PyTorch, CUDA, PEFT/LoRA, deep learning |
| Synthetic data | GANs, diffusion models |
| Anomaly detection | TensorFlow, Keras |
| Evaluation | RAGAS, LangSmith/Phoenix, quantified metrics |
| Deployment | Docker, Git, GitHub Actions, AWS / GCP |
