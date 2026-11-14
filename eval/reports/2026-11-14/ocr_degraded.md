# OCR robustness: clean vs degraded scans — 2026-11-14

Phase 6b exit benchmark. **Pages:** the 300 rendered synthetic invoice pages from Day 23
(150 dpi grayscale, three layouts), each degraded by the classical augmentations of
Day 24 at severity 0.25 / 0.5 / 0.75 / 1.0, giving **1,500 page images**. **Engine:**
Tesseract 5.5 (`--psm 6`, min confidence 40), run in a local Docker image with the same
apt Tesseract as CI.

```
python -m ml.synthetic.diffusion.augment data/synthetic/pages/invoice --severities 0.25 0.5 0.75 1.0
python -m eval.ocr_robustness data/synthetic     --variants data/synthetic/degraded/classical/manifest.jsonl --engines tesseract paddle     --report eval/reports/2026-11-14/ocr_degraded.md --json ... --plot ...
```

**Scores.**
- **CER** is measured against the page's own PDF text layer, read through ingestion's
  line grouping so both sides share a reading order, with separator lines dropped.
- **Extraction F1** is the rules extractor on the OCR text, given the true document type
  (so it moves with OCR quality only), against the ledger record.
- **Key fields readable** is the share of pages whose invoice number, PO number and total
  are all read exactly. This is the Day 24 validity filter.

## Results (tuned preprocessing)

| Engine | Method | Strength | Pages | Corpus CER | Median CER | Extraction F1 | F1 vs clean | Key fields readable | Median s/page | Failed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tesseract | clean | 0.00 | 300 | 0.011 | 0.005 | 0.874 | 100% | 100% | 1.57 | 0 |
| tesseract | classical | 0.25 | 300 | 0.020 | 0.018 | 0.852 | 97% | 98% | 1.54 | 0 |
| tesseract | classical | 0.50 | 300 | 0.064 | 0.026 | 0.829 | 95% | 89% | 1.52 | 0 |
| tesseract | classical | 0.75 | 300 | 0.161 | 0.055 | 0.727 | 83% | 67% | 1.35 | 0 |
| tesseract | classical | 1.00 | 300 | 0.288 | 0.226 | 0.555 | 63% | 33% | 1.31 | 0 |
| paddle | — | — | — | — | — | — | — | — | — | skipped: paddleocr is not installed; `pip install paddlepaddle paddleocr` or set ocr.engine: tesseract |

![strength -> CER and extraction F1](ocr_degraded_curve.png)

Machine-readable numbers: [ocr_degraded.json](ocr_degraded.json).

- **Clean OCR is near-perfect** (CER 0.011, every key field readable on all 300 pages).
- **Graceful up to severity 0.5** (±1.5° skew, σ 0.75 px blur, JPEG q60, 1 % speckle):
  CER 0.064, extraction F1 95 % of clean.
- **The cliff is between 0.75 and 1.0.** At full severity a third of pages still yield
  every key field exactly, so a production pipeline should route such scans to the LLM
  extractor or to human review rather than trust them.
- Median CER stays far below corpus CER at 0.75–1.0. The damage concentrates in a minority
  of pages: the modern layout's white-on-dark banner and "Amount due" bar are the first
  thing lost.

## What the tuning changed (`fix(ocr)` on this branch)

Same 1,500 images, previous preprocessing (deskew → median denoise on every page) vs
tuned (median denoise only when the page has isolated-pixel speckle, and before deskew):

| Severity | Corpus CER | Extraction F1 | Key fields readable |
|---|---:|---:|---:|
| clean | 0.050 → **0.011** | 0.776 → **0.874** | 77% → **100%** |
| 0.25 | 0.019 → 0.020 (same) | 0.852 → 0.852 | 97% → 98% |
| 0.5 | 0.115 → **0.064** | 0.782 → **0.829** | 72% → **89%** |
| 0.75 | 0.222 → **0.161** | 0.660 → **0.727** | 47% → **67%** |
| 1.0 | 0.426 → **0.288** | 0.412 → **0.555** | 14% → **33%** |

Numbers for the previous pipeline: [ocr_degraded_before.json](ocr_degraded_before.json).

- The median filter ate thin strokes on clean 150-dpi pages. Running it only on speckled
  pages took clean CER from 0.050 to 0.011.
- Deskewing first smeared each speck into a grey blob the median could no longer remove.
  Denoising first cut CER by about a third at severities 0.5–1.0.
- Also measured on a 40-page subset and rejected: Otsu binarization (worse everywhere)
  and upscaling 150-dpi pages ×2 (better at 0.5, but worse on clean pages and at
  0.75–1.0).
- Cost: the speckled receipt fixture goes from CER 0.053 to 0.064, within its 0.10 budget.

## Not covered by this run

- **Stable Diffusion img2img variants:** not generated. This machine has no GPU, and the
  Colab notebook (`ml/synthetic/diffusion/notebooks/degrade.ipynb`) has not been run yet.
  Once it has, pass its `validity.jsonl` as a second `--variants` (with `--kept-only`) and
  the SD strength curve joins this table and plot.
- **PaddleOCR:** reported as skipped. It has no wheels for this environment, so the
  per-engine comparison is Tesseract only.
