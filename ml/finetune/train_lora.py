"""QLoRA fine-tuning of Llama 3 8B Instruct for field extraction, on one 16 GB GPU.

Runs on a free Colab / Kaggle T4 (see ``notebooks/colab_qlora_llama3.ipynb``); the GPU
libraries (torch, transformers, peft, bitsandbytes) come from ``requirements-ml.txt`` and
are imported inside the functions, so this module imports anywhere.

The base model is loaded in 4-bit NF4 with double quantization (bitsandbytes): ~5.5 GB of
weights instead of ~16 GB in fp16, which leaves room for activations on a T4.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ml.finetune.prepare_dataset import DEFAULT_BASE_MODEL

logger = logging.getLogger(__name__)

LLAMA3_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class TrainConfig:
    """Everything a run depends on (docs/layers/06-fine-tuning.md, "Training setup")."""

    base_model: str = DEFAULT_BASE_MODEL
    # 4-bit NF4 base (QLoRA)
    load_in_4bit: bool = True
    quant_type: str = "nf4"
    double_quant: bool = True
    compute_dtype: str = "float16"
    """``float16`` on a T4 (no bfloat16 support), ``bfloat16`` on A100 / L4."""
    target_modules: list[str] = field(default_factory=lambda: list(LLAMA3_TARGET_MODULES))


def torch_dtype(name: str) -> Any:
    """``"float16"`` -> ``torch.float16`` (and ``bfloat16``, ``float32``)."""
    import torch

    dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if name not in dtypes:
        raise ValueError(f"unknown dtype {name!r}; choose from {', '.join(dtypes)}")
    return dtypes[name]


def quantization_config(config: TrainConfig) -> Any:
    """The bitsandbytes 4-bit config for the base weights (``None``: load unquantized)."""
    if not config.load_in_4bit:
        return None
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.quant_type,
        bnb_4bit_use_double_quant=config.double_quant,
        bnb_4bit_compute_dtype=torch_dtype(config.compute_dtype),
    )


def load_tokenizer(config: TrainConfig) -> Any:
    """The base model's tokenizer, padding on the right with the end-of-text token
    (Llama 3 ships no pad token)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(config: TrainConfig) -> Any:
    """The base model in 4-bit NF4 on the GPU, made ready for k-bit training (norms in
    fp32, input gradients on for gradient checkpointing)."""
    from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM

    quantization = quantization_config(config)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=quantization,
        torch_dtype=torch_dtype(config.compute_dtype),
        device_map="auto",
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing
    if quantization is not None:
        model = prepare_model_for_kbit_training(model)
    footprint = model.get_memory_footprint() / 2**30
    logger.info("loaded %s: %.1f GiB", config.base_model, footprint)
    return model
