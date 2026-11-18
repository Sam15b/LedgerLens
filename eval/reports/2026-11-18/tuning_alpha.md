# Retrieval tuning: hybrid alpha

- **date:** 2026-11-18
- **golden set:** `data\golden\golden_set.jsonl` (51 questions)
- **k:** 5
- **hybrid scoring:** exact, client-side (`ExactHybridStore`): Pinecone Local ignores sparse vectors
- **filters:** query analyzer (same for every value)
- **command:** `python -m eval.tune_retrieval alpha --report C:\Users\manna\AppData\Local\Temp/alpha.md`

## Picks

| knob | variant | was | picked |
|---|---|---|---|
| alpha | R2 | 0.7 | 0.6 |
| alpha | R3 | 0.7 | 0.7 |

## Sweeps

### alpha on R2 (picked by recall@5)

| alpha | recall@5 | MRR | evidence recall | facts / question | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| 0 | 0.772 | 0.809 | 0.772 | 0.0 | 124 | 178 |
| 0.1 | 0.772 | 0.809 | 0.772 | 0.0 | 71 | 97 |
| 0.2 | 0.772 | 0.809 | 0.772 | 0.0 | 67 | 90 |
| 0.3 | 0.772 | 0.809 | 0.772 | 0.0 | 73 | 88 |
| 0.4 | 0.777 | 0.809 | 0.777 | 0.0 | 70 | 98 |
| 0.5 | 0.777 | 0.809 | 0.777 | 0.0 | 69 | 86 |
| **0.6** (picked) | 0.777 | 0.809 | 0.777 | 0.0 | 69 | 87 |
| 0.7 (was) | 0.777 | 0.798 | 0.777 | 0.0 | 68 | 87 |
| 0.8 | 0.766 | 0.787 | 0.766 | 0.0 | 69 | 84 |
| 0.9 | 0.719 | 0.696 | 0.719 | 0.0 | 69 | 91 |
| 1 | 0.649 | 0.633 | 0.649 | 0.0 | 63 | 80 |

### alpha on R3 (picked by recall@5)

| alpha | recall@5 | MRR | evidence recall | facts / question | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| 0 | 0.836 | 0.826 | 0.836 | 0.0 | 616 | 3,256 |
| 0.1 | 0.836 | 0.826 | 0.836 | 0.0 | 592 | 3,363 |
| 0.2 | 0.836 | 0.826 | 0.836 | 0.0 | 599 | 3,347 |
| 0.3 | 0.836 | 0.826 | 0.836 | 0.0 | 557 | 3,385 |
| 0.4 | 0.836 | 0.826 | 0.836 | 0.0 | 560 | 3,454 |
| 0.5 | 0.836 | 0.826 | 0.836 | 0.0 | 531 | 3,306 |
| 0.6 | 0.836 | 0.837 | 0.836 | 0.0 | 529 | 3,281 |
| **0.7** (picked) (was) | 0.836 | 0.848 | 0.836 | 0.0 | 540 | 3,178 |
| 0.8 | 0.826 | 0.826 | 0.826 | 0.0 | 514 | 3,320 |
| 0.9 | 0.783 | 0.805 | 0.783 | 0.0 | 549 | 3,562 |
| 1 | 0.698 | 0.699 | 0.698 | 0.0 | 545 | 3,617 |

## Reading the numbers

- **Hybrid beats dense once the sparse half is scored.** Day 40's R2 = R1 came from
  Pinecone Local dropping sparse values. Scored exactly, BM25 lifts R2 recall@5 from
  0.649 (alpha 1, dense only) to 0.777, and R3 from 0.698 to 0.836. Document numbers
  (`INV-000123`, `PO-...`) are exact tokens that dense embeddings blur and BM25 matches.
- **The curve is flat from 0.4 to 0.7** and falls off above 0.8. Below 0.4 BM25
  dominates: recall holds, but ties at zero sparse overlap fall back to the dense order.
- **One `hybrid_alpha` serves both variants**, and the default retriever builds on R3's
  ranking. R3's best is **0.7** (MRR 0.848; 0.6 gives 0.837). R2 alone prefers 0.6 by
  MRR only (0.809 vs 0.798, same recall), so `hybrid_alpha` stays at **0.7**.
- Latency does not depend on alpha: every value is one query. The differences in the
  p50 column are noise and cache warm-up. R3's p95 is the CPU cross-encoder.
