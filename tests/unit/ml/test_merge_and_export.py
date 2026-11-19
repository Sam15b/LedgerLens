"""Merging a LoRA adapter into the base and exporting it, on a tiny random Llama (CPU)."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from ml.finetune.merge_and_export import (
    LlamaCpp,
    build_parser,
    export_gguf,
    gguf_paths,
    main,
    merge_adapter,
    parse_steps,
    write_manifest,
)
from ml.finetune.train_lora import sha256

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
    assert (args.steps, args.quant, args.keep_f16) == (["merge", "gguf"], "Q4_K_M", False)


class FakeRun:
    """Records commands and writes each one's output file, like the real tools would."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, command: Sequence[str]) -> None:
        self.commands.append(list(command))
        out = command[command.index("--outfile") + 1] if "--outfile" in command else command[2]
        Path(out).write_bytes(b"GGUF" + bytes(len(self.commands)))


def checkout(tmp_path: Path) -> LlamaCpp:
    root = tmp_path / "llama.cpp"
    (root / "build" / "bin").mkdir(parents=True)
    (root / "convert_hf_to_gguf.py").write_text("", encoding="utf-8")
    (root / "build" / "bin" / "llama-quantize").write_text("", encoding="utf-8")
    return LlamaCpp(root, python="python")


def merged_dir(tmp_path: Path) -> Path:
    merged = tmp_path / "export" / "merged"
    merged.mkdir(parents=True)
    (merged / "config.json").write_text("{}", encoding="utf-8")
    return merged


def test_gguf_export_converts_to_f16_then_quantizes(tmp_path: Path) -> None:
    llama_cpp, merged, out = checkout(tmp_path), merged_dir(tmp_path), tmp_path / "export"
    run = FakeRun()

    result = export_gguf(merged, out, llama_cpp=llama_cpp, run=run)

    f16, q4 = gguf_paths(out)
    assert q4.name == "ledgerlens-llama3-lora.Q4_K_M.gguf"
    convert, quantize = run.commands
    assert convert == [
        "python",
        str(llama_cpp.converter),
        str(merged),
        "--outfile",
        str(f16),
        "--outtype",
        "f16",
    ]
    assert quantize == [
        str(llama_cpp.root / "build" / "bin" / "llama-quantize"),
        str(f16),
        str(q4),
        "Q4_K_M",
    ]
    # the f16 intermediate is removed; the quantized file is recorded
    assert not f16.exists() and q4.exists()
    assert result["quant"] == "Q4_K_M" and result["llama_cpp_commit"] is None
    assert result["files"] == {
        "gguf": {"path": q4.name, "bytes": 6, "sha256": sha256(q4)},
    }
    assert len(result["commands"]) == 2


def test_gguf_export_options_and_missing_inputs(tmp_path: Path) -> None:
    llama_cpp, merged, out = checkout(tmp_path), merged_dir(tmp_path), tmp_path / "export"
    llama_cpp.quantize_bin = Path("/opt/llama/llama-quantize")
    run = FakeRun()

    result = export_gguf(
        merged, out, llama_cpp=llama_cpp, name="m", quant="Q8_0", keep_f16=True, run=run
    )

    assert run.commands[1][0] == str(Path("/opt/llama/llama-quantize"))
    assert run.commands[1][-1] == "Q8_0"
    assert set(result["files"]) == {"gguf", "gguf_f16"}
    assert (out / "m.f16.gguf").exists() and (out / "m.Q8_0.gguf").exists()
    with pytest.raises(FileNotFoundError, match="merged checkpoint"):
        export_gguf(tmp_path / "nowhere", out, llama_cpp=llama_cpp, run=run)
    with pytest.raises(FileNotFoundError, match="--llama-cpp"):
        export_gguf(merged, out, llama_cpp=LlamaCpp(tmp_path / "missing"), run=run)


def test_quantizer_falls_back_to_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="--quantize-bin"):
        LlamaCpp(tmp_path).quantizer()
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    assert LlamaCpp(tmp_path).quantizer() == "/usr/bin/llama-quantize"


def test_manifest_accumulates_steps(tmp_path: Path) -> None:
    write_manifest(tmp_path, {"adapter": "a", "dtype": "float16"})
    path = write_manifest(tmp_path, {"quant": "Q4_K_M"})

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert {k: manifest[k] for k in ("adapter", "dtype", "quant")} == {
        "adapter": "a",
        "dtype": "float16",
        "quant": "Q4_K_M",
    }
    assert "updated_at" in manifest


def test_steps_option() -> None:
    assert parse_steps("gguf") == ["gguf"]
    assert parse_steps("gguf, merge") == ["merge", "gguf"]  # always merge first
    with pytest.raises(argparse.ArgumentTypeError):
        parse_steps("merge,upload")
    with pytest.raises(SystemExit):
        main(["--out", "x", "--steps", "merge"])  # merge needs --adapter
