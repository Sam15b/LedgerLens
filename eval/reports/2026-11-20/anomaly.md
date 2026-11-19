# Anomaly detection comparison

- **date:** 2026-11-20
- **data:** synthetic ledger `data/synthetic/records.jsonl` (5,000 invoices, 450 with an injected anomaly), features from `python -m ml.anomaly.build_features` (19 features, no embedding PCA)
- **split:** by time; **test** = the latest 750 invoices (2026-07-04 .. 2026-10-30, 66 anomalies), never used to train or to pick a threshold. **val** = the 750 before them (62 anomalies), used to pick thresholds.
- **autoencoder:** `models/anomaly/ae-20261119-160552` (Day 50, Keras 3.15 on PyTorch)
- **command:** `python -m ml.anomaly.evaluate --model models/anomaly/ae-20261119-160552 [--split val] --save-combined`

## Methods

| Method | Score (for ranking) | Flag |
|---|---|---|
| Rules | number of different rules that fired (0-5) | any rule hit |
| Isolation Forest | `-score_samples`, 300 trees fit on train normals (same scaled features) | cut chosen on val like the AE's: lowest score with precision ≥ 0.8 (0.656) |
| Autoencoder | reconstruction error | ≥ 0.342, its saved threshold (precision ≥ 0.8 on val) |
| Rules + AE | `anomaly/scorer.py`: a rule hit scores 0.5 + a/2, other invoices a/2, with a = error / (error + threshold) | any rule hit, or AE error ≥ threshold |
| Rules + AE (combined cut) | same | any rule hit, or AE error ≥ **0.188**: the threshold re-chosen on val with the rule hits already flagged (precision ≥ 0.8) |

The rules run over the whole ledger in arrival order (invoice date, then id). Each invoice is checked against the invoices received before it, its PO (minus what earlier invoices billed against it), its contract and the vendor master, as ingestion would see them.

## Test split (750 invoices, 66 anomalies)

| Method | PR-AUC | ROC-AUC | P@50 | Flagged | Precision | Recall | exact_duplicate | near_duplicate | price_inflation | round_amount | rush_terms | unknown_vendor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Rules | 0.661 | 0.879 | 0.84 | 60 | 0.85 | 0.77 | 13/13 | 9/17 | 15/15 | 6/6 | 0/7 | 8/8 |
| Isolation Forest | 0.200 | 0.641 | 0.30 | 1 | 0.00 | 0.00 | 0/13 | 0/17 | 0/15 | 0/6 | 0/7 | 0/8 |
| Autoencoder | 0.394 | 0.690 | 0.46 | 13 | 0.77 | 0.15 | 1/13 | 0/17 | 0/15 | 3/6 | 0/7 | 6/8 |
| Rules + AE | **0.830** | **0.952** | 0.84 | 63 | 0.81 | 0.77 | 13/13 | 9/17 | 15/15 | 6/6 | 0/7 | 8/8 |
| **Rules + AE (combined cut)** | **0.830** | **0.952** | 0.84 | 71 | 0.75 | **0.80** | 13/13 | 9/17 | 15/15 | 6/6 | 2/7 | 8/8 |

## Validation split (750 invoices, 62 anomalies)

The thresholds were chosen on this split, so these numbers are optimistic. They are shown to check that test behaves the same way.

| Method | PR-AUC | ROC-AUC | P@50 | Flagged | Precision | Recall | exact_duplicate | near_duplicate | price_inflation | round_amount | rush_terms | unknown_vendor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Rules | 0.702 | 0.890 | 0.88 | 55 | 0.89 | 0.79 | 11/11 | 9/15 | 16/16 | 4/5 | 0/6 | 9/9 |
| Isolation Forest | 0.135 | 0.570 | 0.14 | 1 | 1.00 | 0.02 | 0/11 | 0/15 | 0/16 | 0/5 | 1/6 | 0/9 |
| Autoencoder | 0.366 | 0.641 | 0.44 | 7 | 0.86 | 0.10 | 0/11 | 0/15 | 0/16 | 1/5 | 1/6 | 4/9 |
| Rules + AE | 0.861 | 0.960 | 0.90 | 57 | 0.88 | 0.81 | 11/11 | 9/15 | 16/16 | 4/5 | 1/6 | 9/9 |
| Rules + AE (combined cut) | 0.861 | 0.960 | 0.90 | 65 | 0.80 | 0.84 | 11/11 | 9/15 | 16/16 | 5/5 | 2/6 | 9/9 |

## Findings

1. **The combination beats each method alone.** Test PR-AUC is 0.830 for rules + AE, against 0.661 for the rules, 0.394 for the autoencoder and 0.200 for Isolation Forest. The rules give a coarse ranking (0, 1 or 2 rules), and the autoencoder orders the invoices inside each band. That ordering is what lifts PR-AUC and ROC-AUC, even though the flags barely change at the AE's own threshold.
2. **The autoencoder adds recall where no rule applies.** With the combined cut (0.188, chosen on val), rules + AE catches 2/7 rush-terms invoices on test, which no rule covers, for 8 more flags. It lifts recall from 0.77 to 0.80 and costs precision 0.81 → 0.75 on test (on val it was set to 0.80). The AE's own 0.342 threshold adds almost nothing on top of the rules: its few catches (round amounts, unknown vendors) are already rule hits.
3. **Isolation Forest is not a useful baseline here.** On test it ranks barely better than chance (ROC-AUC 0.64), and at the val-chosen cut it flags one invoice, which is wrong. The anomalies are small shifts in a few features, such as a duplicated total or a slightly inflated unit price. Random axis-aligned splits do not isolate them faster than ordinary variety between vendors.
4. **Almost every rule miss and rule false alarm is the same labelling effect, not a rule error.** The ledger shifts a near duplicate's date by up to 7 days either way. When the copy is dated *before* the invoice it copies, it arrives first and has nothing to match. The clean original, arriving later, is then flagged instead: as a near duplicate of the copy, and as over its PO, which the copy already billed. On test this explains **8 of the 9** rule false alarms and **all 8** missed duplicates; on val, **6 of 6** and **6 of 6**. The pair is still caught, but on the other invoice. Per pair, the rules catch every duplicate in both splits.
5. **Near-duplicate rule, Day 49 carry-over:** over the whole ledger, the rules flag 50 clean invoices. **43** are originals of these swapped pairs (each also over its PO). The other **7 of 4,550** are genuine false alarms (recurring identical orders with adjacent invoice numbers), as found on Day 49. The rule stays as it is. A real feed has the same order problem (a resubmission can carry an earlier date than the invoice it copies), so the reviewer sees the pair whichever side is flagged, through the `POSSIBLE_DUPLICATE` edge.

## Decision: serving threshold

`anomaly/scorer.py` flags **any rule hit, or AE error ≥ the combined cut** (0.188 for `ae-20261119-160552`). The combined cut is written into the model's `metadata.json` under `combined` by `--save-combined`, and the scorer prefers it over the AE's standalone threshold. Rule hits rank first in the review list (score ≥ 0.5). The autoencoder orders invoices within each band, and alone it flags only invoices whose error reaches the combined cut.

## Not covered

- The PCA text features from header-chunk embeddings are still not built (Days 46-49 carry-over), so every row here uses the 19 numeric features only.
- The labels are synthetic: one injected anomaly type per invoice. Real invoices can carry several issues, or issues that no type here describes.
