"""The Ollama Modelfile serves the model with the prompt and decoding it was trained for."""

import re
from pathlib import Path

import pytest

from ml.finetune.infer import DEFAULT_MAX_NEW_TOKENS
from ml.finetune.merge_and_export import MODELFILE, gguf_paths, write_modelfile
from ml.finetune.prepare_dataset import SYSTEM_PROMPT


def directives(text: str) -> dict[str, list[str]]:
    """Modelfile instructions (``FROM``, ``PARAMETER``, ...) and their arguments; triple-quoted
    values may span lines."""
    found: dict[str, list[str]] = {}
    pattern = r'^(FROM|TEMPLATE|SYSTEM|PARAMETER) (?:"""(.*?)"""|([^\n]*))$'
    for match in re.finditer(pattern, text, re.M | re.S):
        found.setdefault(match[1], []).append(match[2] if match[2] is not None else match[3])
    return found


def parameters(text: str) -> dict[str, list[str]]:
    params: dict[str, list[str]] = {}
    for value in directives(text)["PARAMETER"]:
        name, _, arg = value.partition(" ")
        params.setdefault(name, []).append(arg)
    return params


TEXT = MODELFILE.read_text(encoding="utf-8")


def test_from_loads_the_exported_gguf() -> None:
    assert directives(TEXT)["FROM"] == [f"./{gguf_paths(Path('.'))[1].name}"]


def test_system_prompt_is_the_training_one() -> None:
    assert directives(TEXT)["SYSTEM"] == [SYSTEM_PROMPT]


def test_template_is_the_llama3_chat_format() -> None:
    (template,) = directives(TEXT)["TEMPLATE"]
    turns = [
        "<|start_header_id|>system<|end_header_id|>\n\n{{ .System }}<|eot_id|>",
        "<|start_header_id|>user<|end_header_id|>\n\n{{ .Prompt }}<|eot_id|>",
        "<|start_header_id|>assistant<|end_header_id|>\n\n{{ .Response }}<|eot_id|>",
    ]
    positions = [template.index(turn) for turn in turns]
    assert positions == sorted(positions)


def test_decoding_is_greedy_and_stops_at_end_of_turn() -> None:
    params = parameters(TEXT)
    assert params["temperature"] == ["0"] and params["top_k"] == ["1"]
    assert params["num_predict"] == [str(DEFAULT_MAX_NEW_TOKENS)]
    assert int(params["num_ctx"][0]) >= 2048  # the training length
    assert '"<|eot_id|>"' in params["stop"]


def test_write_modelfile_points_from_at_the_gguf(tmp_path: Path) -> None:
    path = write_modelfile(tmp_path, "m.Q8_0.gguf")

    written = path.read_text(encoding="utf-8")
    assert path == tmp_path / "Modelfile"
    assert directives(written)["FROM"] == ["./m.Q8_0.gguf"]
    # everything else is the repo's Modelfile
    assert written.replace("FROM ./m.Q8_0.gguf", "") == TEXT.replace(
        f"FROM ./{gguf_paths(Path('.'))[1].name}", ""
    )

    two = tmp_path / "Two"
    two.write_text("FROM a\nFROM b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="one FROM"):
        write_modelfile(tmp_path, "x.gguf", template=two)
