"""Merge a trained LoRA adapter into Llama 3 8B Instruct, for serving without PEFT.

    python -m ml.finetune.merge_and_export \\
        --adapter /content/drive/MyDrive/ledgerlens/finetune/run-2/adapter \\
        --out /content/drive/MyDrive/ledgerlens/export/run-2

The adapter was trained on a 4-bit base, but it is merged into the **16-bit** base: folding
the LoRA product ``B @ A * alpha / r`` into NF4 weights would add a second rounding on top
of the 4-bit one, and the GGUF export quantizes again anyway. 8B parameters in float16 are
~16 GB, more than a T4 (15 GB) or Colab's CPU RAM (12.7 GB) holds alone, so the base loads
with ``device_map="auto"`` and offloads what does not fit to ``--offload-dir``; PEFT moves
each offloaded layer in to merge it. Kaggle's 30 GB of RAM merges on CPU without
offloading.

The merged folder (``<out>/merged``) is a plain Hugging Face checkpoint: safetensors shards
and the tokenizer, with the Llama 3 chat template.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ml.finetune.prepare_dataset import DEFAULT_BASE_MODEL
from ml.finetune.train_lora import torch_dtype

logger = logging.getLogger(__name__)

DEFAULT_DTYPE = "float16"
"""The merged weights' dtype: float16 everywhere (a T4 has no bfloat16), and what the GGUF
converter writes as ``f16``."""
MAX_SHARD_SIZE = "2GB"


def merge_adapter(
    adapter: str | Path,
    out: Path,
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    dtype: str = DEFAULT_DTYPE,
    device_map: str | None = "auto",
    offload_dir: Path | None = None,
) -> Path:
    """Fold ``adapter`` into ``base_model``'s weights and save the plain model with its
    tokenizer to ``out``. Returns ``out``."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out.mkdir(parents=True, exist_ok=True)
    load: dict[str, Any] = {"low_cpu_mem_usage": True}
    if device_map is not None:
        load["device_map"] = device_map
        if offload_dir is not None:
            offload_dir.mkdir(parents=True, exist_ok=True)
            load["offload_folder"] = str(offload_dir)
    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch_dtype(dtype), **load)
    model: Any = PeftModel.from_pretrained(base, str(adapter))
    merged = model.merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True, max_shard_size=MAX_SHARD_SIZE)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(out)
    logger.info("merged %s into %s -> %s", adapter, base_model, out)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.finetune.merge_and_export",
        description="Merge a LoRA adapter into the base model.",
    )
    parser.add_argument("--adapter", required=True, help="adapter folder from train_lora.py")
    parser.add_argument("--out", type=Path, required=True, help="export folder")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE, help="float16 | bfloat16 | float32")
    parser.add_argument(
        "--device-map",
        default="auto",
        help="'auto' (GPU, then CPU, then --offload-dir) or 'cpu'",
    )
    parser.add_argument("--offload-dir", type=Path, help="disk for layers that fit nowhere")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    merge_adapter(
        args.adapter,
        args.out / "merged",
        base_model=args.base_model,
        dtype=args.dtype,
        device_map=args.device_map,
        offload_dir=args.offload_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
