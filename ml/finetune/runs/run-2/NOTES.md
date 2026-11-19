# Run 2: QLoRA r=32, lr 1e-4, 2 epochs (compared with run 1)

**Status: not trained yet, and neither is run 1.** Both configs and the data are pinned
(`config.json`, `dataset.json` in each folder). Training needs a 16 GB GPU and access to the
gated Llama 3 weights, and neither is available on the machine that prepared this. The
comparison table below has no numbers yet: it is filled from the two runs' `run.json` and
from `eval_extraction.py`, not by hand.

## What changes from run 1

| | run 1 | run 2 | why |
|---|---|---|---|
| LoRA r / α | 16 / 32 | **32 / 64** | more adapter capacity for line items, the field the rules baseline does worst on; α / r stays 2, so the update scale is unchanged |
| learning rate | 2e-4 | **1e-4** | twice the rank at the same scale moves the weights further per step; halving the rate keeps the step size about the same |
| epochs | 3 (≈ 191 steps) | **2 (≈ 127 steps)** | 1,016 training examples with a fixed output format: the format is learned in well under an epoch, and a third pass mostly memorises receipts |
| everything else | | same | 4-bit NF4, all 7 projections, dropout 0.05, effective batch 16, 2,048 tokens, cosine with 3 % warmup, seed 13, the same split files (identical sha256) |

Only rank and schedule change, so a difference between the runs can be put down to them.
Validation and early stopping work as in run 1 (every 50 steps, patience 3), so run 2 also
keeps its best checkpoint, not its last.

## If run 1's curves point elsewhere

These values were chosen before run 1 existed. Once its `metrics.jsonl` is in, adjust
`configs/run-2.yaml` before training run 2:

- **val loss bottoms out early and rises** (best checkpoint at step 50–100): keep r 32 but
  train 1 epoch, or raise `lora_dropout` to 0.1.
- **val loss still falling at the end of epoch 3**: keep lr 2e-4 and 3 epochs, change only
  r → 32.
- **F1 good but JSON validity < 98 %**: the format is not learned; this is a data problem
  (check `prepare_dataset.py`'s targets), not one more epochs fix.
- **line-item F1 is the gap and the rest is close to GPT-4o**: r 64 / α 128 is the next
  step up; ≈ 168 M trainable parameters still fit a T4.

## Comparison (filled after both runs)

| | run 1 | run 2 |
|---|---|---|
| trainable parameters | ≈ 42 M | ≈ 84 M |
| steps run / planned | – / 191 | – / 127 |
| best val loss (step) | – | – |
| stopped early | – | – |
| train time (T4) | – | – |
| test F1, all fields | – | – |
| test F1, line items | – | – |
| JSON valid % | – | – |
| p95 latency, Q4_K_M via Ollama | – | – |

*(Trainable parameters: r × (d_in + d_out) summed over the 7 projections of the 32 layers
of Llama 3 8B: 2.62 M per unit of r.)* The rest comes from:

```bash
# per run: training curves and early stopping
jq '{global_step, stopped_early, val_loss, train_seconds}' ml/finetune/runs/run-2/run.json
# test F1 and JSON validity, once per adapter (GPU; the default --data is the test split)
python -m ml.finetune.eval_extraction --models lora \
    --adapter $DRIVE/ledgerlens/finetune/run-2/adapter --out run-2.test.jsonl
```

The run with the higher test F1 is the one exported. On a tie, run 1 is exported, because
its setup is the documented default.

## How to run it

As run 1 (`../run-1/NOTES.md`), with the run-2 config:

```bash
python -m ml.finetune.train_lora --config ml/finetune/configs/run-2.yaml \
    --set checkpoint_dir=/content/drive/MyDrive/ledgerlens/finetune
```

Expected time is about two thirds of run 1 (127 instead of 191 steps; the larger adapter
adds little next to the 8B forward and backward passes), so ≈ 1.5–2 h on a T4. If memory
runs short, use micro-batch 1 × 16 accumulation, as for run 1.

## Then: merge, GGUF, Ollama

```bash
python -m ml.finetune.merge_and_export \
    --adapter /content/drive/MyDrive/ledgerlens/finetune/run-2/adapter \
    --out /content/drive/MyDrive/ledgerlens/export/run-2 --llama-cpp llama.cpp
# on the machine that serves it, with the export folder copied over:
ollama create ledgerlens-llama3-lora -f export/run-2/Modelfile
ollama run ledgerlens-llama3-lora "Extract ... Document text: ..."
```

The merge needs ~16 GB for the float16 base: on Colab's T4 it offloads to `--offload-dir`;
Kaggle's 30 GB of RAM does it with `--device-map cpu`. The Q4_K_M file is ~4.9 GB.

### What was checked without a GPU (not a result)

- The merge step on a tiny random Llama with the real Llama 3 tokenizer and a random
  adapter on every projection: the merged checkpoint reloads, has no LoRA weights, and
  gives the same logits as base + adapter (`tests/unit/ml/test_merge_and_export.py`
  checks this on CPU).
- `ollama create` (docker-compose Ollama) accepts `Modelfile`, with FROM pointed at a
  small public GGUF, and `ollama show` lists the same system prompt, template and
  parameters.
- **Not run:** llama.cpp's converter and `llama-quantize`. The GGUF step is covered by
  unit tests with a fake command runner only. The first real conversion is the run-2
  export above. This Ollama build cannot import safetensors on Linux (it needs MLX), so
  the GGUF is the only way in.
