# Run 1: QLoRA r=16 on Llama 3 8B Instruct

**Status: not trained yet.** The config and data are pinned (`config.json`, `dataset.json`).
The run needs a 16 GB GPU and access to the gated Llama 3 weights, and neither is
available on the machine that prepared it. Nothing in this folder is a training result
yet.

## Setup

| | |
|---|---|
| base | `meta-llama/Meta-Llama-3-8B-Instruct`, 4-bit NF4 + double quantization |
| LoRA | r 16, α 32, dropout 0.05, all attention + MLP projections |
| optimiser | paged 8-bit AdamW, lr 2e-4, cosine, 3 % warmup |
| batch | 2 × 8 gradient accumulation = 16 |
| length | 2,048 tokens (longest example: 1,111) |
| precision | `auto` → fp16 on a T4, gradient checkpointing on |
| schedule | 3 epochs ≈ 191 steps (1,016 train examples / 16) |
| validation | 130 val examples every 50 steps; stop after 3 evaluations without improvement |
| data | `dataset.json`: CORD 778 / 98 / 100 and synthetic 238 / 32 / 30 (train / val / test) |

## How to run it

1. Build the data and check it matches `dataset.json` (same sha256 per split):
   `python -m ml.finetune.prepare_dataset --out data/processed/finetune`.
2. Upload `data/processed/finetune/` to Drive (`MyDrive/ledgerlens/finetune-data`).
3. Open `ml/finetune/notebooks/colab_qlora_llama3.ipynb` on a T4 runtime.
4. Run the 10-step dry run (`RUN_ID = "dry-run"`) first.
5. Then set `RUN_ID = "run-1"` and run the full-run cell.
6. If the session times out, rerun the setup cells and the same cell. It resumes from the
   last checkpoint in `MyDrive/ledgerlens/finetune/run-1/`.
7. Copy `ml/finetune/runs/run-1/run.json` and `metrics.jsonl` back here, and commit them.

## What to look for

- **Done when** `run.json` has `"status": "finished"`, `val_loss.went_down` is `true`
  and the adapter is in `MyDrive/ledgerlens/finetune/run-1/adapter`.
- **Expected memory**: ~5.5 GB of 4-bit weights plus activations at micro-batch 2 ×
  2,048 tokens, which should sit well under 15 GB. If it runs out of memory, set
  `--set per_device_train_batch_size=1 --set gradient_accumulation_steps=16`; the
  effective batch stays 16.
- **Expected time**: about 2–3 h on a T4. That is more than one Colab session can be
  relied on for, so expect to resume at least once.
- **Next**: Day 45 scores the adapter on the 130 test documents with
  `python -m ml.finetune.eval_extraction --models rules,base,lora --adapter ...`.

## What the CPU smoke tests showed (not a result)

The pipeline was exercised on CPU with a 2-layer random Llama that uses the real Llama 3
tokenizer and chat template, on the real prepared data:

- training steps run and the adapter saves;
- a run resumes from its last checkpoint;
- validation runs every `eval_steps`, and its loss reaches `run.json`;
- the best checkpoint is the one kept.

The loss values from those runs (≈ 11.7, i.e. ln of the vocabulary size) say nothing
about the real model.
