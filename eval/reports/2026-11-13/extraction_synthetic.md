# Extraction on rendered synthetic PDFs — 2026-11-13

**Data.** `records.jsonl` was built from the Day 21 conditional GAN
(`python -m ml.synthetic.build --checkpoint <conditional run>/final.pt --invoices 5000`).
`python -m ml.synthetic.render.render_documents --limit 300` rendered 300 invoices, 300
POs and 300 remittance advices, spread evenly over the file, plus all 50 contracts:
**950 PDFs**, with 150-dpi page images. They were rendered through PyMuPDF's Story
(WeasyPrint needs GTK, which this Windows machine lacks), taking 29 s in all. The
invoices include 25 anomalous ones: 8 price inflation, 6 exact and 6 near duplicates, and
2 round-amount, 2 unknown-vendor and 1 rush-terms.

**Run.** `python -m eval.extraction_synthetic data/synthetic` sends every PDF through the
normal `IngestionPipeline` (dedupe, text layer, classify, **rules** extractor, chunk; no
embedding, no LLM calls). Each document's extracted scalar fields are scored against the
record it was rendered from.

Rendered documents by template: {'contract': 50, 'invoice_classic': 96, 'invoice_compact': 83, 'invoice_modern': 121, 'payment_advice': 300, 'purchase_order': 300}.

## Ingestion

| Documents | Ingested | Byte-identical duplicates (deduped) | Failed | Median ms / document |
|---:|---:|---:|---:|---:|
| 950 | 950 | 0 | 0 | 5 |

## Classification — accuracy 91.3%

| True \ predicted | contract | invoice | payment | purchase_order | unknown |
|---|---:|---:|---:|---:|---:|
| contract | 50 | 0 | 0 | 0 | 0 |
| invoice | 0 | 217 | 0 | 0 | 83 |
| payment | 0 | 0 | 300 | 0 | 0 |
| purchase_order | 0 | 0 | 0 | 300 | 0 |

## Extraction (rules extractor)

| Group | Docs | Schema-valid | Micro P | Micro R | Micro F1 |
|---|---:|---:|---:|---:|---:|
| contract | 50 | 100.0% | 0.895 | 0.895 | 0.895 |
| invoice | 300 | 32.0% | 1.000 | 0.567 | 0.724 |
| payment | 300 | 100.0% | 1.000 | 0.983 | 0.991 |
| purchase_order | 300 | 100.0% | 1.000 | 0.980 | 0.990 |
| invoice · invoice_classic | 96 | 100.0% | 1.000 | 0.987 | 0.993 |
| invoice · invoice_compact | 83 | 0.0% | 0.000 | 0.000 | 0.000 |
| invoice · invoice_modern | 121 | 0.0% | 1.000 | 0.623 | 0.768 |

### F1 per field: contract

| Field | Support | contract |
|---|---:|---:|
| contract_number | 50 | 1.000 |
| currency | 50 | 1.000 |
| customer_name | 50 | 0.160 |
| end_date | 50 | 1.000 |
| payment_terms_days | 50 | 1.000 |
| start_date | 50 | 1.000 |
| title | 50 | 1.000 |
| vendor_name | 50 | 1.000 |

### F1 per field: invoice

| Field | Support | invoice | invoice_classic | invoice_compact | invoice_modern |
|---|---:|---:|---:|---:|---:|
| currency | 300 | 0.839 | 1.000 | 0.000 | 1.000 |
| customer_name | 300 | 0.485 | 1.000 | 0.000 | 0.000 |
| due_date | 300 | 0.485 | 1.000 | 0.000 | 0.000 |
| invoice_date | 300 | 0.839 | 1.000 | 0.000 | 1.000 |
| invoice_number | 300 | 0.485 | 1.000 | 0.000 | 0.000 |
| po_number | 298 | 0.840 | 1.000 | 0.000 | 1.000 |
| subtotal | 298 | 0.483 | 1.000 | 0.000 | 0.000 |
| tax | 298 | 0.840 | 1.000 | 0.000 | 1.000 |
| total | 300 | 0.839 | 1.000 | 0.000 | 1.000 |
| vendor_name | 300 | 0.839 | 1.000 | 0.000 | 1.000 |
| vendor_tax_id | 298 | 0.766 | 0.921 | 0.000 | 0.924 |

### F1 per field: payment

| Field | Support | payment |
|---|---:|---:|
| amount | 300 | 1.000 |
| currency | 300 | 1.000 |
| method | 300 | 0.936 |
| paid_at | 300 | 1.000 |
| payer_name | 300 | 1.000 |
| payment_reference | 300 | 1.000 |
| vendor_name | 300 | 1.000 |

### F1 per field: purchase_order

| Field | Support | purchase_order |
|---|---:|---:|
| buyer_name | 300 | 1.000 |
| contract_number | 300 | 1.000 |
| currency | 300 | 1.000 |
| delivery_date | 300 | 1.000 |
| order_date | 300 | 1.000 |
| po_number | 300 | 1.000 |
| subtotal | 300 | 1.000 |
| tax | 300 | 1.000 |
| total | 300 | 1.000 |
| vendor_name | 300 | 1.000 |
| vendor_tax_id | 300 | 0.879 |

## Findings

**Done-when met:** 950 rendered PDFs (≥ 500) were ingested with **0 failures**, at a
median of 5 ms each. None of the six exact duplicates in this sample had its source in the sample, so
dedupe caught nothing here. The unit test renders a duplicate together with its source
and checks that the second copy is dropped as byte-identical.

**Fixed along the way** (separate commits on this branch). Both were real ingestion bugs
that the ground truth exposed:

| Bug | Symptom | Before → after |
|---|---|---|
| Text layer kept ligatures (`ﬁ`) | "Archive files" read as "Archive ﬁles" | line descriptions and labels with fi/fl now match |
| A tall title chained the two lines beside it into one | "Northwind 1 Market Street Traders PURCHASE ORDER" | PO `buyer_name` F1 0.000 → 1.000 (with the rules fix that drops a title sharing the name's line); PO micro F1 0.898 → 0.990 |

**What the rules baseline cannot do.** The three invoice layouts were written to differ:

- **classic** ("Label: value" on one line): F1 0.993. This is the layout the regexes were
  written for.
- **modern** (header row of labels with the values in the row below): F1 0.768. The
  invoice number, due date, subtotal and customer are missed because each label and its
  value land on different text lines. That makes the invoice schema-invalid
  (`invoice_number` is required) for every modern document.
- **compact** (till-style "BILL NO. / AMOUNT PAYABLE"): never contains the word "invoice",
  so the rules classifier calls all 83 `unknown`. No extraction runs, giving F1 0 and 91.3 %
  classification accuracy overall. The pipeline's LLM classifier fallback is off here
  (no LLM calls).
- **contract** `customer_name` (F1 0.16): the "between A (the Supplier) and B (the
  Customer)" sentence wraps across two lines, which the single-line regex cannot match.

These are the cases the LLM and fine-tuned extractors (Phase 6a) have to beat.
`eval.extraction_synthetic.evaluate(root, records, extractor=...)` re-scores the same
950 PDFs with any extractor. This run made no paid calls.
