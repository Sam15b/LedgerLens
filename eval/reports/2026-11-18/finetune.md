# Extraction model comparison

- **date:** 2026-11-18
- **test set (held out):** `data/processed/finetune/test.jsonl` (130 documents: 100 cord, 30 synthetic)
- **command:** `python -m ml.finetune.eval_extraction --models rules --report eval/reports/2026-11-18/finetune.md`

## Overall (micro F1 over every field)

Only the rules baseline has run so far; the other rows are the
docs/layers/06-fine-tuning.md comparison still to fill (see "Not run yet" below).

| model | F1 | precision | recall | JSON valid | $ / 1k docs | p50 ms | p95 ms | tokens in / out |
|---|---|---|---|---|---|---|---|---|
| rules | 0.601 | 0.992 | 0.431 | n/a | 0.00 | 0 | 0 | n/a / n/a |
| GPT-4o (`smart`) | not run | | | | | | | |
| Llama 3 8B (base) | not run | | | | | | | |
| **Llama 3 8B + LoRA** | not run | | | | | | | |

## F1 by source

| model | cord (n=100) | synthetic (n=30) |
|---|---|---|
| rules | 0.352 | 0.807 |

## F1 by field

| field | support | rules |
|---|---|---|
| subtotal | 95 | 0.333 |
| tax | 71 | 0.776 |
| total | 125 | 0.897 |
| line_items | 311 | 0.000 |
| invoice_number | 30 | 0.776 |
| vendor_name | 30 | 1.000 |
| vendor_tax_id | 30 | 0.909 |
| customer_name | 30 | 0.571 |
| invoice_date | 30 | 1.000 |
| due_date | 30 | 0.776 |
| po_number | 30 | 1.000 |
| currency | 30 | 1.000 |

## Reading the numbers

- **The rules baseline is precise but misses half the fields** (P 0.992, R 0.431). It
  never extracts line items (311 gold items, recall 0), and on CORD's subtotals it scores
  0.333. Line items are the main thing the fine-tuned model has to add.
- **By source:** synthetic invoices 0.807 (the rules were written for these layouts) vs
  CORD receipts 0.352. The total itself is found well everywhere (field F1 0.897).
- Latency is ~0 ms (regular expressions on CPU), and there are no tokens and no cost.

## Not run yet

| row | needs | command |
|---|---|---|
| GPT-4o | a working OpenAI key (the configured one returns HTTP 401; Bedrock's token is invalid; no Gemini key). About 130 paid calls. | `python -m ml.finetune.eval_extraction --models smart --out eval/reports/<date>/finetune_smart.jsonl` |
| Llama 3 8B base | a GPU and the gated weights | `python -m ml.finetune.eval_extraction --models base` |
| Llama 3 8B + LoRA | the run-1 adapter (`ml/finetune/runs/run-1/NOTES.md`) | `python -m ml.finetune.eval_extraction --models lora --adapter $DRIVE/ledgerlens/finetune/run-1/adapter` |

Run them on the same `test.jsonl` (sha256 in `ml/finetune/runs/run-1/dataset.json`), and
merge with `--models rules,smart,base,lora --report ...` to regenerate this table in one
go. SROIE is not in this test set: it needs a manual download.
