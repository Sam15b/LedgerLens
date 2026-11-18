# Retrieval tuning: rerank candidates and graph hops

- **date:** 2026-11-18
- **golden set:** `data/golden/golden_set.jsonl` v0 (51 questions)
- **k:** 5
- **hybrid scoring:** exact, client-side (`ExactHybridStore`): Pinecone Local ignores sparse vectors; `hybrid_alpha` 0.7 ([tuning_alpha.md](tuning_alpha.md))
- **filters:** query analyzer (same for every value)
- **commands:**
  - `python -m eval.tune_retrieval candidates hops --report eval/reports/2026-11-18/tuning_rerank_graph.md` (candidates 5-50, hops with 50 candidates)
  - `python -m eval.tune_retrieval candidates --candidates 40,50,75,100` (the grid's best, 50, was at its edge)

## Picks

| knob | variant | was | sweep's pick | set to |
|---|---|---|---|---|
| candidates (`pinecone.query_top_k`) | R3 | 20 | 75 | **50** |
| hops (`retrieval.graph.max_hops`) | R4 | 2 | 0 | **1** |

## Sweeps

### candidates on R3 (picked by recall@5, ties to fewer)

| candidates | recall@5 | MRR | evidence recall | p50 ms | p95 ms |
|---|---|---|---|---|---|
| 5 | 0.777 | 0.809 | 0.777 | 576 | 1,127 |
| 10 | 0.836 | 0.848 | 0.836 | 655 | 1,893 |
| 20 (was) | 0.836 | 0.848 | 0.836 | 585 | 3,635 |
| 30 | 0.836 | 0.848 | 0.836 | 596 | 4,459 |
| 40 | 0.847 | 0.869 | 0.847 | 594 | 6,245 |
| **50** (set) | 0.868 | 0.901 | 0.868 | 559 | 7,336 |
| 75 (sweep's pick) | 0.879 | 0.922 | 0.879 | 630 | 10,941 |
| 100 | 0.879 | 0.922 | 0.879 | 640 | 15,759 |

50 appears in both runs with the same scores (p95 7,336 / 7,492 ms).

### hops on R4 (picked by evidence recall, ties to fewer facts), 50 candidates

| hops | recall@5 | MRR | evidence recall | facts / question | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| 0 (sweep's pick) | 0.889 | 0.904 | 0.953 | 2.4 | 686 | 7,130 |
| **1** (set) | 0.889 | 0.904 | 0.953 | 2.6 | 649 | 7,825 |
| 2 (was) | 0.889 | 0.904 | 0.953 | 4.5 | 707 | 7,295 |

## Reading the numbers

- **More rerank candidates find more.** The cross-encoder can only promote what R2 hands
  it: recall@5 climbs from 0.836 (10-30 candidates) to 0.868 at 50 and 0.879 at 75, then
  stops (100 = 75). MRR follows (0.848 -> 0.901 -> 0.922).
- **The cost is the tail.** p50 hardly moves: most questions carry analyzer filters that
  leave fewer chunks than the candidate count. The unfiltered ones rerank every candidate,
  so p95 grows with the count: 3.6 s at 20, 7.3 s at 50, 10.9 s at 75 on this CPU.
  **50** is the knee: +0.032 recall@5 over 20, and 75 adds +0.011 more for +3.6 s at p95.
  On a GPU the cross-encoder is far cheaper and 75 would be the better setting.
- **Hops do not change what the golden set measures.** All three hop limits reach the
  same evidence (0.953); fewer hops just give fewer facts. Two hops add contract fan-out
  (every invoice under a seed contract: 4.5 facts per question vs 2.6), which contract
  questions already get from the `contract_for_po` template. **1 hop** keeps the
  payment -> invoice and PO -> invoice walks that "which invoice did PAY-123 settle?" needs;
  golden set v0 has no such question yet, so 0 would win only by default.
- recall@5 is higher for R4 (0.889) than for R3 (0.868) here: on questions whose chunks
  cover fewer than five documents, a document cited by a graph fact lands in the top 5.
