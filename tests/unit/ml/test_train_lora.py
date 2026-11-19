"""QLoRA training script: config loading, run ids, the collator and the run log.

The GPU path (4-bit load, training) runs on Colab; these cover what runs anywhere.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ml.finetune.prepare_dataset import IGNORE_INDEX
from ml.finetune.train_lora import (
    DEFAULT_CONFIG,
    LLAMA3_TARGET_MODULES,
    RunLog,
    SFTCollator,
    TrainConfig,
    check_steps,
    dataset_summary,
    load_examples,
    load_train_config,
    make_run_id,
    parse_overrides,
    resolve_dtype,
    training_arguments,
    val_loss_summary,
    warmup_arguments,
)


def test_repo_run_config_matches_the_documented_setup() -> None:
    config = load_train_config(DEFAULT_CONFIG)

    assert config.base_model == "meta-llama/Meta-Llama-3-8B-Instruct"
    assert (config.load_in_4bit, config.quant_type, config.double_quant) == (True, "nf4", True)
    assert (config.lora_r, config.lora_alpha, config.lora_dropout) == (16, 32, 0.05)
    assert config.target_modules == list(LLAMA3_TARGET_MODULES)
    assert (config.learning_rate, config.lr_scheduler, config.warmup_ratio) == (
        2e-4,
        "cosine",
        0.03,
    )
    assert config.effective_batch_size == 16
    assert config.max_length == 2048
    assert config.gradient_checkpointing is True
    assert (config.compute_dtype, config.optim) == ("auto", "paged_adamw_8bit")


def test_run_2_changes_only_rank_and_schedule_and_matches_its_record() -> None:
    run_1 = load_train_config(DEFAULT_CONFIG).to_json()
    run_2 = load_train_config(DEFAULT_CONFIG.with_name("run-2.yaml")).to_json()

    changed = {k: (run_1[k], run_2[k]) for k in run_1 if run_1[k] != run_2[k]}
    assert changed == {
        "lora_r": (16, 32),
        "lora_alpha": (32, 64),
        "learning_rate": (2e-4, 1e-4),
        "num_train_epochs": (3.0, 2.0),
        "run_id": ("", "run-2"),
    }
    runs = DEFAULT_CONFIG.parent.parent / "runs"
    recorded = json.loads((runs / "run-2" / "config.json").read_text(encoding="utf-8"))
    assert recorded == run_2 | {"checkpoint_dir": "/content/drive/MyDrive/ledgerlens/finetune"}
    # both runs train on the same files
    assert (runs / "run-1" / "dataset.json").read_text(encoding="utf-8") == (
        runs / "run-2" / "dataset.json"
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("name", "capability", "expected"),
    [
        ("auto", (7, 5), "float16"),  # T4: no native bfloat16
        ("auto", (6, 0), "float16"),  # P100
        ("auto", (8, 0), "bfloat16"),  # A100
        ("auto", (8, 9), "bfloat16"),  # L4
        ("auto", None, "float32"),  # CPU
        ("float16", (8, 0), "float16"),  # explicit wins
    ],
)
def test_resolve_dtype(name: str, capability: tuple[int, int] | None, expected: str) -> None:
    assert resolve_dtype(name, capability) == expected


def test_overrides_are_coerced_to_the_field_type(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text("lora_r: 16\nnum_train_epochs: 3\n", encoding="utf-8")

    config = load_train_config(
        path,
        parse_overrides(
            ["max_steps=10", "load_in_4bit=false", "target_modules=[q_proj, v_proj]", "run_id=7"]
        ),
    )

    assert config.max_steps == 10
    assert config.load_in_4bit is False
    assert config.target_modules == ["q_proj", "v_proj"]
    assert config.run_id == "7"
    assert config.num_train_epochs == 3.0 and isinstance(config.num_train_epochs, float)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"lr": "1"}, "unknown"),
        ({"max_steps": "ten"}, "number"),
        ({"load_in_4bit": "maybe"}, "true or false"),
        ({"target_modules": "q_proj"}, "list"),
    ],
)
def test_bad_values_are_rejected(overrides: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_train_config(DEFAULT_CONFIG, overrides)


def test_parse_overrides_needs_key_value() -> None:
    assert parse_overrides([" a = 1 ", "b=x=y"]) == {"a": "1", "b": "x=y"}
    with pytest.raises(ValueError, match="key=value"):
        parse_overrides(["max_steps"])


def test_run_id_is_kept_or_made_from_time_and_rank() -> None:
    now = datetime(2026, 11, 18, 9, 30, 5, tzinfo=UTC)

    assert make_run_id(TrainConfig(run_id="run-1"), now) == "run-1"
    assert make_run_id(TrainConfig(lora_r=8), now) == "20261118-093005-qlora-r8"


def test_warmup_argument_follows_the_transformers_version() -> None:
    assert warmup_arguments({"warmup_ratio", "warmup_steps"}, 0.03) == {"warmup_ratio": 0.03}
    assert warmup_arguments({"warmup_steps"}, 0.03) == {"warmup_steps": 0.03}


def test_collator_pads_ids_masks_and_labels() -> None:
    pytest.importorskip("torch")
    batch = [
        {"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1], "labels": [IGNORE_INDEX, 6, 7]},
        {"input_ids": [8], "attention_mask": [1], "labels": [8]},
    ]

    padded = SFTCollator(pad_token_id=0)(batch)

    assert padded["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert padded["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert padded["labels"].tolist() == [[IGNORE_INDEX, 6, 7], [8, IGNORE_INDEX, IGNORE_INDEX]]


def write_split(data: Path, split: str, sources: list[str]) -> None:
    rows = [
        {
            "id": f"{source}-{i}",
            "source": source,
            "doc_id": f"{source}-{i}",
            "split": split,
            "doc_type": "receipt",
            "instruction": "Extract",
            "input": f"text {i}",
            "output": json.dumps({"total": f"{i}.00"}),
        }
        for i, source in enumerate(sources)
    ]
    (data / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (data / f"{split}.chat.jsonl").write_text("{}\n")


def test_load_examples_round_trips_the_dataset(tmp_path: Path) -> None:
    write_split(tmp_path, "val", ["cord", "synthetic"])

    examples = load_examples(tmp_path / "val.jsonl")

    assert [e.doc_id for e in examples] == ["cord-0", "synthetic-1"]
    assert examples[1].target == {"total": "1.00"}
    assert examples[1].output == '{"total": "1.00"}'


def test_dataset_summary_counts_sources_and_hashes_files(tmp_path: Path) -> None:
    write_split(tmp_path, "train", ["cord", "cord", "synthetic"])
    write_split(tmp_path, "test", ["cord"])

    summary = dataset_summary(tmp_path)

    assert set(summary["splits"]) == {"train", "test"}  # chat files are not counted twice
    assert summary["splits"]["train"]["by_source"] == {"cord": 2, "synthetic": 1}
    assert summary["splits"]["train"]["examples"] == 3
    assert len(summary["splits"]["test"]["sha256"]) == 64


def test_run_log_merges_updates_and_appends_metrics(tmp_path: Path) -> None:
    log = RunLog(tmp_path, "run-1")

    log.update(status="running", config={"lora_r": 16})
    log.update(status="finished")
    log.log(10, {"loss": 1.5})
    log.log(20, {"eval_loss": 1.2})

    run = json.loads((tmp_path / "run-1" / "run.json").read_text(encoding="utf-8"))
    assert run == {"status": "finished", "config": {"lora_r": 16}}
    lines = (tmp_path / "run-1" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [
        {"step": 10, "loss": 1.5},
        {"step": 20, "eval_loss": 1.2},
    ]


def test_val_loss_summary() -> None:
    history = [
        {"loss": 2.0, "step": 10},
        {"eval_loss": 1.8, "step": 50},
        {"eval_loss": 1.5, "step": 100},
        {"eval_loss": 1.6, "step": 150},
    ]

    assert val_loss_summary(history) == {
        "evaluations": 3,
        "first": 1.8,
        "best": 1.5,
        "best_step": 100,
        "last": 1.6,
        "went_down": True,
    }
    assert val_loss_summary([{"loss": 2.0, "step": 10}]) is None


def test_save_steps_must_be_a_multiple_of_eval_steps() -> None:
    check_steps(TrainConfig(save_steps=100, eval_steps=50))
    with pytest.raises(ValueError, match="multiple"):
        check_steps(TrainConfig(save_steps=50, eval_steps=40))


def test_training_arguments_validate_and_keep_the_best_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    pytest.importorskip("accelerate")

    args = training_arguments(TrainConfig(eval_steps=25, save_steps=50), tmp_path)

    assert (args.eval_steps, args.save_steps) == (25, 50)
    assert str(args.eval_strategy).lower().endswith("steps")
    assert args.load_best_model_at_end and args.metric_for_best_model == "eval_loss"
    assert args.greater_is_better is False
    assert args.gradient_checkpointing is True
