"""QLoRA fine-tuning of Llama 3 8B Instruct for field extraction, on one 16 GB GPU.

Runs on a free Colab / Kaggle T4 (see ``notebooks/colab_qlora_llama3.ipynb``); the GPU
libraries (torch, transformers, peft, bitsandbytes) come from ``requirements-ml.txt`` and
are imported inside the functions, so this module imports anywhere.

The base model is loaded in 4-bit NF4 with double quantization (bitsandbytes): ~5.5 GB of
weights instead of ~16 GB in fp16, which leaves room for activations on a T4. The rest of
the T4 budget: fp16 compute (a T4 has no bfloat16), gradient checkpointing (activations
recomputed in the backward pass instead of stored), the paged 8-bit AdamW optimizer
(optimizer state paged to CPU under memory pressure) and sequences cut at 2,048 tokens. LoRA
adapters (r=16, alpha=32, dropout 0.05) go on every attention and MLP projection; only they
train. The data is the output of ``prepare_dataset.py``, tokenized with the Llama 3 chat
template and the loss on the response tokens only::

    python -m ml.finetune.train_lora --config ml/finetune/configs/run.yaml --set max_steps=10

Hyperparameters come from the YAML config (every key a :class:`TrainConfig` field), and
``--set key=value`` overrides any of them for one run.

Free GPU sessions end without warning, so checkpoints go to persistent storage
(``--checkpoint-dir``: a mounted Google Drive folder on Colab, ``/kaggle/working`` on
Kaggle) under the run id, and a rerun with the same ``--run-id`` resumes from the last one.
Each run is logged to ``ml/finetune/runs/<run_id>/``: ``run.json`` (config, environment,
dataset files and hashes, final metrics) and ``metrics.jsonl`` (every logged step). Those
are small JSON files for git; adapter weights stay in the checkpoint folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ml.finetune.prepare_dataset import (
    DEFAULT_BASE_MODEL,
    DEFAULT_MAX_LENGTH,
    DEFAULT_SEED,
    IGNORE_INDEX,
    ChatTokenizer,
    Example,
    encode,
)

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
    compute_dtype: str = "auto"
    """``auto``: ``bfloat16`` on GPUs that have it (A100, L4: compute capability 8+),
    ``float16`` on a T4 / P100, ``float32`` without a GPU."""
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    """Paged 8-bit AdamW (bitsandbytes); ``adamw_torch`` is used without a GPU."""
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: list(LLAMA3_TARGET_MODULES))
    # optimisation: effective batch = micro-batch x gradient accumulation = 16
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: float = 3.0
    max_steps: int = -1
    """Stop after this many optimizer steps (-1: run ``num_train_epochs``)."""
    max_length: int = DEFAULT_MAX_LENGTH
    logging_steps: int = 10
    seed: int = DEFAULT_SEED
    # data and outputs
    data_dir: str = "data/processed/finetune"
    checkpoint_dir: str = "outputs/finetune"
    """Persistent storage for checkpoints and the adapter (Drive / Kaggle output)."""
    runs_dir: str = "ml/finetune/runs"
    """Where the run's JSON log goes (committed; no weights)."""
    run_id: str = ""
    """Empty: a new id from the time and LoRA rank. Reuse an id to resume that run."""
    save_steps: int = 50
    save_total_limit: int = 3

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def to_json(self) -> dict[str, Any]:
        return asdict(self) | {"effective_batch_size": self.effective_batch_size}


DEFAULT_CONFIG = Path(__file__).parent / "configs" / "run.yaml"


def _coerce(name: str, current: Any, value: Any) -> Any:
    """``value`` (from YAML or ``--set``) as the type of the field's default."""
    if isinstance(value, str) and not isinstance(current, str):
        value = yaml.safe_load(value)  # "10" -> 10, "true" -> True, "[a, b]" -> [a, b]
    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{name}: expected true or false, got {value!r}")
        return value
    if isinstance(current, int | float) and not isinstance(value, bool):
        try:
            return type(current)(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}: expected a number, got {value!r}") from exc
    if isinstance(current, list):
        if not isinstance(value, list):
            raise ValueError(f"{name}: expected a list, got {value!r}")
        return [str(item) for item in value]
    return str(value) if isinstance(current, str) and value is not None else value


def parse_overrides(pairs: Sequence[str]) -> dict[str, str]:
    """``["max_steps=10", "run_id=run-1"]`` -> ``{"max_steps": "10", "run_id": "run-1"}``."""
    overrides = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--set expects key=value, got {pair!r}")
        overrides[key.strip()] = value.strip()
    return overrides


def load_train_config(
    path: Path | None = None, overrides: Mapping[str, Any] | None = None
) -> TrainConfig:
    """:class:`TrainConfig` from a YAML file (default: ``configs/run.yaml``), then
    ``overrides``. Unknown keys are an error, so a typo cannot silently fall back to a
    default."""
    path = path or DEFAULT_CONFIG
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must hold a mapping of TrainConfig fields")
    values = {**raw, **(overrides or {})}
    defaults = TrainConfig()
    known = {f.name for f in fields(TrainConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown TrainConfig field(s): {', '.join(unknown)}")
    return TrainConfig(
        **{name: _coerce(name, getattr(defaults, name), v) for name, v in values.items()}
    )


def make_run_id(config: TrainConfig, now: datetime | None = None) -> str:
    """``config.run_id``, or ``<UTC time>-qlora-r<rank>`` for a new run."""
    if config.run_id:
        return config.run_id
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-qlora-r{config.lora_r}"


def resolve_dtype(name: str, capability: tuple[int, int] | None = None) -> str:
    """``auto`` for the GPU at hand (``capability``: CUDA compute capability, ``None``: no
    GPU). bfloat16 needs Ampere (8.x) or newer; ``torch.cuda.is_bf16_supported()`` is not
    used because it also says yes on a T4, where bfloat16 is emulated and slow."""
    if name != "auto":
        return name
    if capability is None:
        return "float32"
    return "bfloat16" if capability[0] >= 8 else "float16"


def cuda_capability() -> tuple[int, int] | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(0)
    return int(major), int(minor)


def torch_dtype(name: str) -> Any:
    """``"float16"`` -> ``torch.float16`` (and ``bfloat16``, ``float32``; ``auto`` is
    resolved for this machine's GPU first)."""
    import torch

    name = resolve_dtype(name, cuda_capability())
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
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.gradient_checkpointing
        )
    footprint = model.get_memory_footprint() / 2**30
    logger.info("loaded %s: %.1f GiB", config.base_model, footprint)
    return model


def lora_config(config: TrainConfig) -> Any:
    """Adapters on every attention and MLP projection of the decoder."""
    from peft import LoraConfig

    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )


# --------------------------------------------------------------------------- data


def load_examples(path: Path) -> list[Example]:
    """A ``<split>.jsonl`` written by ``prepare_dataset.py``, back into examples."""
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        examples.append(
            Example(
                source=row["source"],
                doc_id=row["doc_id"],
                split=row["split"],
                doc_type=row["doc_type"],
                input=row["input"],
                target=json.loads(row["output"]),
            )
        )
    return examples


def tokenize(
    examples: Sequence[Example], tokenizer: ChatTokenizer, *, max_length: int
) -> list[dict[str, list[int]]]:
    """Encoded examples (loss on the response only); over-long ones are skipped."""
    encoded = [encode(example, tokenizer, max_length=max_length) for example in examples]
    kept = [item for item in encoded if item is not None]
    if len(kept) < len(encoded):
        logger.warning("skipped %d examples over %d tokens", len(encoded) - len(kept), max_length)
    return kept


class SFTCollator:
    """Pads a batch to its longest example: ids with the pad token, labels with
    :data:`~ml.finetune.prepare_dataset.IGNORE_INDEX` (no loss on padding)."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: Sequence[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        width = max(len(item["input_ids"]) for item in batch)

        def padded(key: str, value: int) -> Any:
            rows = [item[key] + [value] * (width - len(item[key])) for item in batch]
            return torch.tensor(rows, dtype=torch.long)

        return {
            "input_ids": padded("input_ids", self.pad_token_id),
            "attention_mask": padded("attention_mask", 0),
            "labels": padded("labels", IGNORE_INDEX),
        }


# --------------------------------------------------------------------------- training


def warmup_arguments(fields: set[str], ratio: float) -> dict[str, float]:
    """Warmup as a share of the steps: ``warmup_ratio`` up to transformers 4.x, a float
    ``warmup_steps`` (below 1: a ratio) from 5.0, which dropped ``warmup_ratio``."""
    return {"warmup_ratio": ratio} if "warmup_ratio" in fields else {"warmup_steps": ratio}


def training_arguments(config: TrainConfig, output_dir: Path) -> Any:
    import dataclasses

    from transformers import TrainingArguments

    names = {f.name for f in dataclasses.fields(TrainingArguments)}
    capability = cuda_capability()
    dtype = resolve_dtype(config.compute_dtype, capability)
    kwargs: dict[str, Any] = dict(
        output_dir=str(output_dir),
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        logging_steps=config.logging_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        fp16=capability is not None and dtype == "float16",
        bf16=capability is not None and dtype == "bfloat16",
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=config.optim if capability is not None else "adamw_torch",
        seed=config.seed,
        report_to=[],
    )
    kwargs.update(warmup_arguments(names, config.warmup_ratio))
    return TrainingArguments(**kwargs)


# --------------------------------------------------------------------------- run log


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_summary(data_dir: Path) -> dict[str, Any]:
    """Examples per split and source, and each file's hash: which data a run saw."""
    summary: dict[str, Any] = {"dir": str(data_dir), "splits": {}}
    for path in sorted(data_dir.glob("*.jsonl")):
        if path.name.endswith(".chat.jsonl"):
            continue
        sources: dict[str, int] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                source = json.loads(line)["source"]
                sources[source] = sources.get(source, 0) + 1
        summary["splits"][path.stem] = {
            "examples": sum(sources.values()),
            "by_source": sources,
            "sha256": sha256(path),
        }
    return summary


def _version(module: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(module)
    except PackageNotFoundError:
        return None


def environment() -> dict[str, Any]:
    """Library versions, the GPU and the git commit the run used."""
    env: dict[str, Any] = {
        "python": platform.python_version(),
        **{m: _version(m) for m in ("torch", "transformers", "peft", "bitsandbytes")},
    }
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            env["gpu"] = {"name": props.name, "memory_gib": round(props.total_memory / 2**30, 1)}
        else:
            env["gpu"] = None
    except ImportError:
        env["gpu"] = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        env["git_commit"] = commit.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        env["git_commit"] = None
    return env


class RunLog:
    """``runs/<run_id>/run.json`` and ``metrics.jsonl`` for one run."""

    def __init__(self, runs_dir: Path, run_id: str) -> None:
        self.dir = runs_dir / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_path = self.dir / "run.json"
        self.metrics_path = self.dir / "metrics.jsonl"

    def read(self) -> dict[str, Any]:
        if self.run_path.is_file():
            data: dict[str, Any] = json.loads(self.run_path.read_text(encoding="utf-8"))
            return data
        return {}

    def update(self, **values: Any) -> None:
        run = self.read() | values
        self.run_path.write_text(json.dumps(run, indent=2, default=str) + "\n", encoding="utf-8")

    def log(self, step: int, metrics: Mapping[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": step, **metrics}, default=str) + "\n")


def log_callback(run_log: RunLog) -> Any:
    """A Trainer callback that appends every logged metric dict to ``metrics.jsonl``."""
    from transformers import TrainerCallback

    class _Log(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kw: Any) -> None:
            if logs:
                run_log.log(state.global_step, logs)

    return _Log()


def last_checkpoint(output_dir: Path) -> str | None:
    """The newest ``checkpoint-<step>`` folder to resume from, if any."""
    if not output_dir.is_dir():
        return None
    from transformers.trainer_utils import get_last_checkpoint

    found: str | None = get_last_checkpoint(str(output_dir))
    return found


def build_trainer(config: TrainConfig, output_dir: Path) -> Any:
    """Base model + fresh LoRA adapters, the tokenized train / val splits and a Trainer."""
    from peft import get_peft_model
    from transformers import Trainer

    tokenizer = load_tokenizer(config)
    data = Path(config.data_dir)
    train = tokenize(load_examples(data / "train.jsonl"), tokenizer, max_length=config.max_length)
    val = tokenize(load_examples(data / "val.jsonl"), tokenizer, max_length=config.max_length)
    model = get_peft_model(load_base_model(config), lora_config(config))
    model.print_trainable_parameters()
    return Trainer(
        model=model,
        args=training_arguments(config, output_dir),
        train_dataset=train,
        eval_dataset=val,
        data_collator=SFTCollator(tokenizer.pad_token_id),
    )


def train(config: TrainConfig) -> Path:
    """Train (resuming from the run's last checkpoint if there is one), save the adapter to
    ``checkpoint_dir/<run_id>/adapter`` and log the run. Returns the adapter folder."""
    run_id = make_run_id(config)
    output_dir = Path(config.checkpoint_dir) / run_id
    run_log = RunLog(Path(config.runs_dir), run_id)
    resume = last_checkpoint(output_dir)
    run_log.update(
        run_id=run_id,
        status="running",
        compute_dtype=resolve_dtype(config.compute_dtype, cuda_capability()),
        started_at=run_log.read().get("started_at") or datetime.now(UTC).isoformat(),
        resumed_from=resume,
        config=config.to_json(),
        environment=environment(),
        dataset=dataset_summary(Path(config.data_dir)),
        checkpoint_dir=str(output_dir),
    )
    trainer = build_trainer(config, output_dir)
    trainer.add_callback(log_callback(run_log))
    start = time.perf_counter()
    try:
        result = trainer.train(resume_from_checkpoint=resume)
    except BaseException:
        run_log.update(status="failed")
        raise
    adapter = output_dir / "adapter"
    trainer.model.save_pretrained(adapter)
    run_log.update(
        status="finished",
        finished_at=datetime.now(UTC).isoformat(),
        train_seconds=round(time.perf_counter() - start, 1),
        global_step=result.global_step,
        metrics=result.metrics,
        adapter=str(adapter),
    )
    logger.info("adapter saved to %s; run logged to %s", adapter, run_log.dir)
    return adapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.finetune.train_lora",
        description="QLoRA fine-tuning of the extraction model.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="run YAML")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config field, e.g. --set max_steps=10 (repeatable)",
    )
    parser.add_argument(
        "--print-config", action="store_true", help="print the resolved config and exit"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_train_config(args.config, parse_overrides(args.overrides))
    if args.print_config:
        print(json.dumps(config.to_json(), indent=2))
        return 0
    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
