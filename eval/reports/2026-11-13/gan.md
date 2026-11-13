# GAN quality — 2026-11-13

Two WGAN-GP runs on the seed table (`python -m ml.synthetic.seed`): 1,462 rows, 500
vendor-master PO invoices + 962 CORD receipts. Each was trained for 5,000 generator steps
with the default config (Adam 2e-4, betas 0.5/0.9, 5 critic steps per generator step,
batch 256) **on CPU**, about 9 minutes per run. 1.5k rows do not need the Colab GPU
notebook (`ml/synthetic/gan/notebooks/train_gan.ipynb`); it is there for larger seed sets.

| Run | Conditioning | Seconds | Final critic loss | Final W-distance | Final GP | Run JSON |
|---|---|---:|---:|---:|---:|---|
| `20261113-131707-seed0` | none | 553 | -0.142 | 0.182 | 0.0040 | [gan_run_unconditional.json](gan_run_unconditional.json) |
| `20261113-131707-seed1` | `vendor_category` | 559 | -0.040 | 0.077 | 0.0037 | [gan_run_conditional.json](gan_run_conditional.json) |

Each checkpoint was sampled for 5,000 rows (`--seed 0`, the conditional one with the real
category mix) and scored with
`python -m ml.synthetic.gan.evaluate <run>/final.pt -n 5000`. **Baseline** = two random
halves of the real rows scored against each other: the distance sample size alone gives.
KS and TVD run 0–1; W1 is on the GAN's scaled space (log amounts, every column min-max
scaled to [-1, 1]). Weights (`*.pt`) are git-ignored; only the run JSON is committed.

## Summary

| Metric | Unconditional | Conditional | Baseline |
|---|---:|---:|---:|
| Mean KS (numeric) | 0.169 | 0.150 | 0.035 |
| Mean W1 (numeric, scaled) | 0.042 | 0.036 | 0.020 |
| Mean TVD (categorical) | 0.017 | 0.013 | 0.033 |
| Joint TVD (category × currency) | 0.059 | 0.039 | 0.090 |
| Correlation diff, mean abs | **0.023** | 0.059 | 0.039 |
| Correlation diff, max abs | 0.061 (tax_rate × total) | 0.281 (qty_mean × days_to_due) | 0.121 |
| Rows breaking business rules | **0.0 %** | **0.0 %** | — |

## Per column

| Column | KS uncond. | KS cond. | KS baseline | W1 uncond. | W1 cond. | W1 baseline |
|---|---:|---:|---:|---:|---:|---:|
| line_count | 0.051 | 0.096 | 0.026 | 0.020 | 0.029 | 0.011 |
| qty_mean | 0.479 | 0.498 | 0.041 | 0.012 | 0.015 | 0.009 |
| unit_price_mean | 0.173 | 0.049 | 0.041 | 0.038 | 0.018 | 0.023 |
| subtotal | 0.160 | **0.031** | 0.040 | 0.051 | 0.014 | 0.017 |
| tax_rate | 0.159 | 0.258 | 0.025 | 0.054 | 0.067 | 0.018 |
| total | 0.158 | **0.033** | 0.037 | 0.051 | 0.014 | 0.016 |
| days_to_due | 0.111 | 0.154 | 0.036 | 0.052 | 0.052 | 0.039 |
| day_of_month | 0.065 | 0.078 | 0.038 | 0.061 | 0.081 | 0.025 |

| Column | TVD uncond. | TVD cond. | TVD baseline |
|---|---:|---:|---:|
| vendor_category | 0.020 | 0.010 | 0.047 |
| currency | 0.013 | 0.015 | 0.019 |

## Reading the numbers

- **Business rules hold for every row.** `total` is recomputed as
  `subtotal × (1 + tax_rate)` after rounding to the currency step, so no sampled row
  breaks it; this is enforced by `ml/synthetic/gan/sample.py`, not learned.
- **Amounts:** the conditional GAN matches the real `subtotal`, `total` and
  `unit_price_mean` distributions about as closely as a second real sample does (KS
  0.03–0.05 vs a 0.04 baseline). The unconditional run is clearly worse on amounts (KS
  ~0.16): without the category it has to learn a five-orders-of-magnitude, multi-modal
  (IDR vs GBP) amount distribution from noise alone.
- **Categories and currencies** are at or below the sampling-noise floor for both runs,
  including the joint category × currency mix, so the generator does not invent pairings
  such as a marketing agency billing in IDR more often than the data has them.
- **`qty_mean` and `tax_rate` have point masses that the GAN smooths over.** 53 % of real
  rows have `qty_mean` exactly 1.0, and `tax_rate` takes a handful of exact values (0,
  0.08, 0.10, 0.11, 0.20). The generator outputs continuous values near those spikes. KS
  catches this (0.48–0.50 for `qty_mean`), but the mass is in the right place, so W1 is
  close to the baseline (0.012 vs 0.009). The usual fix is CTGAN-style mode-specific
  normalization, or snapping `tax_rate` to the country VAT table when records are linked
  to vendors (Day 22).
- **Correlations:** the unconditional run keeps the real correlation structure better
  than the baseline split (mean abs diff 0.023). The conditional run has one weak pair,
  `qty_mean × days_to_due` (0.28): within a category, receipts (qty 1, due 0) and
  invoices (larger quantities, due 7–60 days) are not separated as sharply as in the
  real rows.

**Choice for Day 22:** link records from the **conditional** checkpoint. Its amount
distributions are at baseline level, and linking already overrides the weak columns
(`days_to_due` comes from the vendor contract; `tax_rate` from the vendor's country).
