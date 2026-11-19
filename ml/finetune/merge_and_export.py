"""Merge a trained LoRA adapter into Llama 3 8B Instruct and export it to GGUF for Ollama.

    python -m ml.finetune.merge_and_export \\
        --adapter /content/drive/MyDrive/ledgerlens/finetune/run-2/adapter \\
        --out /content/drive/MyDrive/ledgerlens/export/run-2 --llama-cpp llama.cpp

Two steps (``--steps merge,gguf``, both by default):

**merge**: the adapter was trained on a 4-bit base, but it is merged into the **16-bit**
base: folding the LoRA product ``B @ A * alpha / r`` into NF4 weights would add a second
rounding on top of the 4-bit one, and the GGUF export quantizes again anyway. 8B parameters
in float16 are ~16 GB, more than a T4 (15 GB) or Colab's CPU RAM (12.7 GB) holds alone, so
the base loads with ``device_map="auto"`` and offloads what does not fit to
``--offload-dir``; PEFT moves each offloaded layer in to merge it. Kaggle's 30 GB of RAM
merges on CPU without offloading. The merged folder (``<out>/merged``) is a plain Hugging
Face checkpoint: safetensors shards and the tokenizer, with the Llama 3 chat template.

**gguf**: llama.cpp's ``convert_hf_to_gguf.py`` writes the merged model as a float16 GGUF,
then ``llama-quantize`` turns it into ``Q4_K_M`` (4-bit k-quants, with the attention value
and FFN down projections of half the layers kept at 6-bit): ~4.9 GB for 8B, which runs on a
laptop CPU or an 8 GB GPU. ``<out>/<name>.Q4_K_M.gguf`` is what the Ollama ``Modelfile``
loads; the step copies ``ml/finetune/Modelfile`` next to it. llama.cpp is not a pip
package; on Colab::

    git clone --depth 1 https://github.com/ggml-org/llama.cpp
    pip install -e llama.cpp/gguf-py sentencepiece
    cmake -S llama.cpp -B llama.cpp/build
    cmake --build llama.cpp/build -j --target llama-quantize

(or unpack a release build and pass ``--quantize-bin``). ``<out>/export.json`` records the
inputs, the commands, the llama.cpp commit and each file's size and sha256.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml.finetune.prepare_dataset import DEFAULT_BASE_MODEL
from ml.finetune.train_lora import sha256, torch_dtype

logger = logging.getLogger(__name__)

DEFAULT_DTYPE = "float16"
"""The merged weights' dtype: float16 everywhere (a T4 has no bfloat16), and what the GGUF
converter writes as ``f16``."""
MAX_SHARD_SIZE = "2GB"
MODEL_NAME = "ledgerlens-llama3-lora"
DEFAULT_QUANT = "Q4_K_M"
STEPS = ("merge", "gguf")
MODELFILE = Path(__file__).with_name("Modelfile")

Runner = Callable[[Sequence[str]], None]
"""Runs one command and raises if it fails (tests pass a fake)."""


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


# --------------------------------------------------------------------------- GGUF


def run_command(command: Sequence[str]) -> None:
    logger.info("$ %s", subprocess.list2cmdline(list(command)))
    subprocess.run(list(command), check=True)


@dataclass
class LlamaCpp:
    """Where llama.cpp's tools are: a checkout (for ``convert_hf_to_gguf.py``) and a
    ``llama-quantize`` binary (default: the checkout's CMake build, then ``PATH``)."""

    root: Path
    quantize_bin: Path | None = None
    python: str = field(default=sys.executable)

    @property
    def converter(self) -> Path:
        return self.root / "convert_hf_to_gguf.py"

    def quantizer(self) -> str:
        if self.quantize_bin is not None:
            return str(self.quantize_bin)
        build = self.root / "build" / "bin"
        for folder in (build, build / "Release"):
            for name in ("llama-quantize", "llama-quantize.exe"):
                if (folder / name).is_file():
                    return str(folder / name)
        found = shutil.which("llama-quantize")
        if found is None:
            raise FileNotFoundError(
                f"llama-quantize is not in {build} or on PATH: build it (see "
                "ml/finetune/merge_and_export.py) or pass --quantize-bin"
            )
        return found

    def convert_command(self, merged: Path, outfile: Path, outtype: str = "f16") -> list[str]:
        return [
            self.python,
            str(self.converter),
            str(merged),
            "--outfile",
            str(outfile),
            "--outtype",
            outtype,
        ]

    def quantize_command(self, source: Path, outfile: Path, quant: str) -> list[str]:
        return [self.quantizer(), str(source), str(outfile), quant]

    def commit(self) -> str | None:
        """The checkout's git commit (the converter changes often), if it is a git repo."""
        try:
            result = subprocess.run(
                ["git", "-C", str(self.root), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip() or None


def gguf_paths(out: Path, name: str = MODEL_NAME, quant: str = DEFAULT_QUANT) -> tuple[Path, Path]:
    """``(<out>/<name>.f16.gguf, <out>/<name>.<quant>.gguf)``."""
    return out / f"{name}.f16.gguf", out / f"{name}.{quant}.gguf"


def export_gguf(
    merged: Path,
    out: Path,
    *,
    llama_cpp: LlamaCpp,
    name: str = MODEL_NAME,
    quant: str = DEFAULT_QUANT,
    keep_f16: bool = False,
    run: Runner = run_command,
) -> dict[str, Any]:
    """Convert the merged checkpoint to a float16 GGUF and quantize it to ``quant``.
    Returns what was run and written (the GGUF part of ``export.json``)."""
    if not (merged / "config.json").is_file():
        raise FileNotFoundError(f"{merged} is not a merged checkpoint (no config.json)")
    if not llama_cpp.converter.is_file():
        raise FileNotFoundError(f"{llama_cpp.converter} not found: pass --llama-cpp <checkout>")
    out.mkdir(parents=True, exist_ok=True)
    f16, quantized = gguf_paths(out, name, quant)
    commands = [
        llama_cpp.convert_command(merged, f16),
        llama_cpp.quantize_command(f16, quantized, quant),
    ]
    for command in commands:
        run(command)
    files = {"gguf": file_record(quantized)}
    if keep_f16:
        files["gguf_f16"] = file_record(f16)
    else:
        f16.unlink(missing_ok=True)
    return {
        "quant": quant,
        "llama_cpp_commit": llama_cpp.commit(),
        "commands": [subprocess.list2cmdline(c) for c in commands],
        "files": files,
    }


def write_modelfile(out: Path, gguf: str, template: Path = MODELFILE) -> Path:
    """Copy the Ollama ``Modelfile`` to ``<out>/Modelfile``, loading ``gguf`` (a file name
    in ``out``)."""
    lines = template.read_text(encoding="utf-8").splitlines()
    froms = [i for i, line in enumerate(lines) if line.startswith("FROM ")]
    if len(froms) != 1:
        raise ValueError(f"{template} must have exactly one FROM line")
    lines[froms[0]] = f"FROM ./{gguf}"
    path = out / "Modelfile"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def file_record(path: Path) -> dict[str, Any]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_manifest(out: Path, values: dict[str, Any]) -> Path:
    """Merge ``values`` into ``<out>/export.json`` (an export may run one step at a time)."""
    path = out / "export.json"
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    current.update(values, updated_at=datetime.now(UTC).isoformat())
    path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- CLI


def parse_steps(value: str) -> list[str]:
    steps = {s.strip() for s in value.split(",") if s.strip()}
    if not steps or steps - set(STEPS):
        raise argparse.ArgumentTypeError(f"--steps takes {', '.join(STEPS)}, got {value!r}")
    return [s for s in STEPS if s in steps]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.finetune.merge_and_export",
        description="Merge a LoRA adapter into the base model and export it to GGUF.",
    )
    parser.add_argument("--adapter", help="adapter folder from train_lora.py (merge step)")
    parser.add_argument("--out", type=Path, required=True, help="export folder")
    parser.add_argument(
        "--steps", type=parse_steps, default=list(STEPS), help="merge,gguf (default: both)"
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE, help="float16 | bfloat16 | float32")
    parser.add_argument(
        "--device-map",
        default="auto",
        help="'auto' (GPU, then CPU, then --offload-dir) or 'cpu'",
    )
    parser.add_argument("--offload-dir", type=Path, help="disk for layers that fit nowhere")
    parser.add_argument("--llama-cpp", type=Path, default=Path("llama.cpp"), help="checkout")
    parser.add_argument("--quantize-bin", type=Path, help="llama-quantize binary")
    parser.add_argument("--quant", default=DEFAULT_QUANT, help="llama-quantize type")
    parser.add_argument("--name", default=MODEL_NAME, help="GGUF file name stem")
    parser.add_argument("--keep-f16", action="store_true", help="keep the float16 GGUF")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if "merge" in args.steps and not args.adapter:
        parser.error("the merge step needs --adapter")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    merged = args.out / "merged"
    if "merge" in args.steps:
        merge_adapter(
            args.adapter,
            merged,
            base_model=args.base_model,
            dtype=args.dtype,
            device_map=args.device_map,
            offload_dir=args.offload_dir,
        )
        write_manifest(
            args.out,
            {"base_model": args.base_model, "adapter": str(args.adapter), "dtype": args.dtype},
        )
    if "gguf" in args.steps:
        gguf = export_gguf(
            merged,
            args.out,
            llama_cpp=LlamaCpp(args.llama_cpp, args.quantize_bin),
            name=args.name,
            quant=args.quant,
            keep_f16=args.keep_f16,
        )
        modelfile = write_modelfile(args.out, gguf["files"]["gguf"]["path"])
        write_manifest(args.out, {"name": args.name, **gguf, "modelfile": modelfile.name})
        logger.info("next: ollama create %s -f %s", MODEL_NAME, modelfile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
