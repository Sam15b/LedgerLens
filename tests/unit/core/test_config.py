from pathlib import Path

import pytest
from pydantic import ValidationError

from ledgerlens.core.config import AppConfig, Settings, load_config, resolve_configs_dir


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


def test_settings_work_without_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.delenv("APP_ENV", raising=False)

    settings = Settings()

    assert settings.app_env == "dev"
    assert settings.embedding_dim == 768


def test_configs_dir_override(tmp_path: Path) -> None:
    assert resolve_configs_dir(tmp_path) == tmp_path
