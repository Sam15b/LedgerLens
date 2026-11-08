from pathlib import Path

import pytest
from pydantic import ValidationError

from ledgerlens.core.config import (
    ENV_FILES,
    AppConfig,
    Settings,
    load_config,
    resolve_configs_dir,
    split_model_ref,
)


def test_repo_configs_load_and_validate() -> None:
    config = load_config()

    assert config.pinecone.metric == "dotproduct"
    assert config.pinecone.dimension == 768
    assert config.ingestion.ocr.engine in {"tesseract", "paddle"}
    assert config.agents.max_steps > 0
    assert config.models.retry.max_attempts >= 1


def test_missing_config_dir_falls_back_to_defaults(tmp_path: Path) -> None:
    assert load_config(tmp_path) == AppConfig()


def test_partial_yaml_overrides_only_given_keys(tmp_path: Path) -> None:
    (tmp_path / "ingestion.yaml").write_text("ocr:\n  dpi: 200\n", encoding="utf-8")

    config = load_config(tmp_path)

    assert config.ingestion.ocr.dpi == 200
    assert config.ingestion.ocr.engine == "tesseract"


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "pinecone.yaml").write_text("indx: typo\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(tmp_path)


def test_invalid_value_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "pinecone.yaml").write_text("hybrid_alpha: 1.5\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(tmp_path)


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "models.yaml").write_text("- just\n- a list\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        load_config(tmp_path)


def test_pricing_uses_in_out_keys(tmp_path: Path) -> None:
    (tmp_path / "models.yaml").write_text(
        "pricing_usd_per_1m_tokens:\n  openai/gpt-4o: { in: 2.5, out: 10 }\n", encoding="utf-8"
    )

    price = load_config(tmp_path).models.pricing_usd_per_1m_tokens["openai/gpt-4o"]

    assert (price.input, price.output) == (2.5, 10)


def test_resolve_alias(tmp_path: Path) -> None:
    (tmp_path / "models.yaml").write_text("aliases:\n  local: [ollama/llama3:8b]\n", "utf-8")
    models = load_config(tmp_path).models

    assert models.resolve("local") == ["ollama/llama3:8b"]
    with pytest.raises(KeyError, match="unknown model alias"):
        models.resolve("nope")


def test_agent_specific_extras_are_kept() -> None:
    reconciler = load_config().agents.agents["reconciler"]

    assert reconciler.model == "smart"
    assert reconciler.model_extra == {"price_tolerance_pct": 2.0}


def test_settings_work_without_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_ENV", raising=False)

    monkeypatch.setitem(Settings.model_config, "env_file", None)  # ignore any developer .env

    settings = Settings()

    assert settings.app_env == "dev"
    assert settings.embedding_dim == 768


def test_repo_env_file_is_found_from_any_directory() -> None:
    repo_env, cwd_env = ENV_FILES

    assert repo_env.is_absolute()
    assert repo_env.parent == resolve_configs_dir().parent  # same root as configs/
    assert Settings.model_config["env_file"] == ENV_FILES
    assert not cwd_env.is_absolute()  # resolved against the working directory at load time


def test_working_directory_env_file_overrides_repo_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    root_env = tmp_path / "root.env"
    root_env.write_text("APP_ENV=staging\nLOG_LEVEL=DEBUG\n", encoding="utf-8")
    (tmp_path / ".env").write_text("APP_ENV=prod\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    monkeypatch.setitem(Settings.model_config, "env_file", (root_env, ENV_FILES[1]))

    settings = Settings()

    assert settings.app_env == "prod"  # ./.env wins
    assert settings.log_level == "DEBUG"  # repo-root value kept where ./.env is silent


def test_configs_dir_override(tmp_path: Path) -> None:
    assert resolve_configs_dir(tmp_path) == tmp_path


def _models_yaml(tmp_path: Path, body: str) -> Path:
    (tmp_path / "models.yaml").write_text(body, encoding="utf-8")
    return tmp_path


def test_repo_models_yaml_aliases_and_prices_are_valid() -> None:
    models = load_config().models

    assert models.resolve("local") == ["ollama/llama3:8b"]
    assert models.timeout_s > 0
    for ref in models.pricing_usd_per_1m_tokens:
        split_model_ref(ref)


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("ollama/llama3:8b", ("ollama", "llama3:8b")),
        ("openai/gpt-4o", ("openai", "gpt-4o")),
        ("ollama/hf.co/org/model:q4", ("ollama", "hf.co/org/model:q4")),
    ],
)
def test_split_model_ref(ref: str, expected: tuple[str, str]) -> None:
    assert split_model_ref(ref) == expected


@pytest.mark.parametrize("ref", ["gpt-4o", "openai/", "/gpt-4o", "anthropc/claude"])
def test_split_model_ref_rejects_bad_refs(ref: str) -> None:
    with pytest.raises(ValueError):
        split_model_ref(ref)


def test_alias_with_unknown_provider_is_rejected(tmp_path: Path) -> None:
    _models_yaml(tmp_path, "aliases:\n  smart: [opneai/gpt-4o]\n")

    with pytest.raises(ValidationError, match="unknown provider 'opneai'"):
        load_config(tmp_path)


def test_alias_listing_a_model_twice_is_rejected(tmp_path: Path) -> None:
    _models_yaml(tmp_path, "aliases:\n  fast: [openai/gpt-4o-mini, openai/gpt-4o-mini]\n")

    with pytest.raises(ValidationError, match="more than once"):
        load_config(tmp_path)


def test_pricing_key_must_be_a_model_ref(tmp_path: Path) -> None:
    _models_yaml(tmp_path, "pricing_usd_per_1m_tokens:\n  gpt-4o: { in: 1, out: 2 }\n")

    with pytest.raises(ValidationError, match="provider/model"):
        load_config(tmp_path)


def test_resolve_accepts_explicit_model_ref(tmp_path: Path) -> None:
    models = load_config(_models_yaml(tmp_path, "aliases: {}\n")).models

    assert models.resolve("openai/gpt-4o") == ["openai/gpt-4o"]


def test_resolve_empty_alias_is_an_error(tmp_path: Path) -> None:
    models = load_config(_models_yaml(tmp_path, "aliases:\n  smart: []\n")).models

    with pytest.raises(ValueError, match="no models"):
        models.resolve("smart")


def test_resolve_returns_a_copy(tmp_path: Path) -> None:
    models = load_config(_models_yaml(tmp_path, "aliases:\n  local: [ollama/a]\n")).models

    models.resolve("local").append("ollama/b")

    assert models.resolve("local") == ["ollama/a"]


def test_retry_and_timeout_are_parsed(tmp_path: Path) -> None:
    body = "retry: { max_attempts: 2, backoff_s: [0.5, 3] }\ntimeout_s: 15\n"
    models = load_config(_models_yaml(tmp_path, body)).models

    assert models.retry.max_attempts == 2
    assert models.timeout_s == 15
    assert [models.retry.delay_for(n) for n in (1, 2, 3, 9)] == [0.5, 3, 3, 3]


@pytest.mark.parametrize(
    "body",
    [
        "retry: { max_attempts: 0 }\n",
        "retry: { backoff_s: [] }\n",
        "retry: { backoff_s: [1, -2] }\n",
        "timeout_s: 0\n",
    ],
)
def test_invalid_retry_or_timeout_is_rejected(tmp_path: Path, body: str) -> None:
    with pytest.raises(ValidationError):
        load_config(_models_yaml(tmp_path, body))
