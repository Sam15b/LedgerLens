"""Merging a LoRA adapter into the base and exporting it, on a tiny random Llama (CPU)."""

from pathlib import Path
from typing import Any

import pytest

from ml.finetune.merge_and_export import build_parser, merge_adapter

VOCAB = ["<unk>", "<s>", "</s>", "total", "date", "vendor", "1.00", "{", "}", ":"]


def tiny_base(path: Path) -> Path:
    """A 2-layer random Llama with a word-level tokenizer, saved like a Hub checkpoint."""
    pytest.importorskip("torch")
    tokenizers: Any = pytest.importorskip("tokenizers")
    transformers: Any = pytest.importorskip("transformers")

    config = transformers.LlamaConfig(
        vocab_size=len(VOCAB),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    transformers.LlamaForCausalLM(config).save_pretrained(path)
    model = tokenizers.models.WordLevel({w: i for i, w in enumerate(VOCAB)}, unk_token="<unk>")
    tokenizer = tokenizers.Tokenizer(model)
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", bos_token="<s>", eos_token="</s>"
    ).save_pretrained(path)
    return path


def tiny_adapter(base: Path, path: Path) -> Path:
    """A LoRA adapter with random (non-zero) B, so merging changes the weights."""
    peft: Any = pytest.importorskip("peft")
    transformers: Any = pytest.importorskip("transformers")

    model = transformers.AutoModelForCausalLM.from_pretrained(base)
    lora = peft.LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj", "down_proj"],
        init_lora_weights=False,
        task_type="CAUSAL_LM",
    )
    peft.get_peft_model(model, lora).save_pretrained(path)
    return path


def logits(model: Any) -> Any:
    import torch

    with torch.no_grad():
        return model(input_ids=torch.tensor([[1, 3, 9, 6, 2]])).logits


def test_merged_model_matches_base_plus_adapter(tmp_path: Path) -> None:
    base = tiny_base(tmp_path / "base")
    adapter = tiny_adapter(base, tmp_path / "adapter")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = merge_adapter(
        adapter, tmp_path / "merged", base_model=str(base), dtype="float32", device_map=None
    )

    with_adapter = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base), adapter)
    merged = AutoModelForCausalLM.from_pretrained(out)
    plain = AutoModelForCausalLM.from_pretrained(base)
    assert logits(merged).allclose(logits(with_adapter), atol=1e-5)
    assert not logits(merged).allclose(logits(plain), atol=1e-3)
    # a plain checkpoint: no PEFT config or LoRA weights, the tokenizer next to it
    assert not (out / "adapter_config.json").exists()
    assert not any("lora" in name for name in merged.state_dict())
    assert list(out.glob("*.safetensors"))
    assert AutoTokenizer.from_pretrained(out).convert_tokens_to_ids("total") == 3


def test_cli_defaults() -> None:
    args = build_parser().parse_args(["--adapter", "a", "--out", "x"])
    assert (args.dtype, args.device_map, args.offload_dir) == ("float16", "auto", None)
    assert args.base_model == "meta-llama/Meta-Llama-3-8B-Instruct"
