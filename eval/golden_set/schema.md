# Golden set

`data/golden/golden_set.jsonl` holds questions about the synthetic ledger with known
answers. The retrieval benchmark (`eval/benchmark_retrieval.py`) scores every retriever
variant on them. One JSON object per line; `eval/golden_set/schema.py` defines and
validates the format.

```json
{"id": "q021", "category": "multi_hop",
 "question": "Which payment settled invoice INV-005742?",
 "ground_truth_answer": "Invoice INV-005742 was settled by remittance PAY-2025-000255 on 2025-02-28.",
 "relevant_document_ids": ["<invoice document id>", "<remittance document id>"],
 "requires_graph": true, "difficulty": "medium",
 "expected": {"number": "PAY-2025-000255", "date": "2025-02-28"},
 "source": "synthetic", "reviewed": true}
```

## Fields

| Field | Type | Meaning |
|-------|------|---------|
| `id` | `q` + 3 or more digits | Stable id. Ids are never reused for a different question. |
| `category` | see below | Question shape, the unit of the per-category benchmark breakdown. |
| `question` | text ending in `?` | As a user would ask it, with no hints about where the answer is. |
| `ground_truth_answer` | text | A complete, correct answer in one or two sentences. |
| `relevant_document_ids` | list of `documents.id` | Documents whose text or graph records hold the evidence, most important first. Recall@k and MRR are computed against this list. Empty only for `unanswerable`. |
| `requires_graph` | bool | The answer depends on links between records, or on a total over many records. Vector search alone is not expected to find it. |
| `difficulty` | `easy` / `medium` / `hard` | `easy`: one fact from one document. `medium`: one hop or one clause. `hard`: several documents, a computation, or a comparison. |
| `expected` | object | Key facts a correct answer must state, for automatic answer checks. Possible keys: `amount` (as printed, `"12,656.66"`), `currency`, `date` (ISO), `number`, `numbers`, `count`, `days`, `vendor`, `match`, `items`, `duplicate`, `answerable`. |
| `source` | text | `synthetic` for v0. Hand-written questions on real documents will use `sroie` / `cuad`. |
| `reviewed` | bool | The answer was checked against the evidence (see Review below). |

## Categories (v0 counts)

| Category | n | Shapes | Evidence |
|----------|---|--------|----------|
| `lookup` | 12 | total, due date or PO of an invoice; total of a PO; amount and vendor of a remittance | one document |
| `clause` | 8 | payment terms, a rate on the rate card, the end date of a contract | the contract |
| `multi_hop` | 12 | which payment settled an invoice; which contract governs the PO on an invoice; what is still owed on a partly paid invoice; a vendor's invoices with a PO but no payment | invoice + payment / contract / PO documents |
| `aggregation` | 9 | invoice count per vendor-year; amount invoiced per vendor-quarter; outstanding balance with a vendor; top vendor by spend per currency-year | the rendered documents among the counted invoices |
| `reconciliation` | 6 | invoice unit prices against the contract rate card (2 inflated, 2 clean); invoices submitted twice | invoice + contract / the other copy |
| `unanswerable` | 4 | invoices, POs and contracts that do not exist | none: the right answer is "not in the records" |

27 of the 51 questions have `requires_graph`.

## How v0 was built

```bash
python -m eval.golden_set.build_golden_set --seed 39 --out data/golden/golden_set.draft.jsonl
```

The builder reads the stores' view of the ledger from Postgres: records, payment
allocations, and which record came from which document. It takes the generator's anomaly
labels (`price_inflation`, `exact_duplicate`) from `data/synthetic/records.jsonl`.

- Every question is about **rendered documents**, so the evidence exists as text and in
  the vector index.
- Questions avoid **invoice numbers that two records share** (the synthetic exact
  duplicates). The stores resolve those ambiguously: Postgres leaves the payment
  allocation unlinked, while the graph links it to one of the two copies.
- Dates in questions are **absolute** (`Q2 2026`, `in 2025`), so answers do not change
  as time passes.

## Review

1. **Single-document answers** (lookup, clause): the builder drops a draft unless the
   amount, number, date or day count it states appears in the document's own text.
   Amounts are matched with or without thousands separators; dates are matched in any
   printed format.
2. **Multi-document answers** (multi_hop, aggregation, reconciliation): each one is
   derived a second time, independently, from Neo4j (SETTLES amounts, REFERENCES /
   GOVERNS links, contract rate cards). All 31 agree with the builder.
3. **Store check**: `python -m eval.golden_set.schema --store` confirms every
   `relevant_document_ids` entry is a ready document in Postgres.

v0 was reviewed by these checks plus a read-through of every question and answer for
wording. It has **not yet had a human domain review**. Do one before quoting results
externally, as `docs/layers/09-evaluation.md` asks.

## Synthetic extension (q052–q155)

Added on 2026-11-21 with the same builder. It now has more question shapes per category
(see its docstring), and every new question carries a `check` recipe:

```bash
python -m eval.golden_set.build_golden_set --extend data/golden/golden_set.jsonl --seed 63 \
    --out <drafts>
python -m eval.golden_set.review <drafts> --accept-verified
python -m eval.golden_set.verify                  # 104/104 checks agree
```

- **Grounding**: single-document drafts must state facts found in their document's text,
  as in v0.
- **Graph check**: `verify.py` derives every answer a second time from Neo4j, using
  Cypher over SETTLES amounts, REFERENCES / GOVERNS paths, rate cards, counts and totals.
  For a clause missing from a contract it uses SQL over the contract's text. All 104
  agree. When the answers were deliberately altered, the check failed every one.
- **Read-through**: every question and answer was read for wording.

Like v0, these questions have **not** had a human domain review. They are template
drafts over known ground truth. LLM-drafted questions must always go through the
interactive review (`review.py` without `--accept-verified`).

The synthetic targets are lookup 25, clause 15, multi_hop 45, aggregation 30,
reconciliation 20 and unanswerable 20 (`SYNTHETIC_TARGETS`). Hand-written SROIE lookups
and CUAD clause questions fill the rest of the ~200.

## Validating

```bash
python -m eval.golden_set.schema                 # format, ids, duplicates
python -m eval.golden_set.schema --store         # + every document exists (needs Postgres)
```

CI runs the format checks in `tests/unit/eval/test_golden_set.py`.

## Changing the set

- Never change the meaning of an existing id. Retire it and add a new one.
- After a reload of the synthetic data (new document ids), rebuild with the same seed and
  re-run the review.
- The target is about 200 questions (`docs/layers/09-evaluation.md`). Hand-written
  questions on SROIE/CUAD documents come next, with `source` set accordingly.
