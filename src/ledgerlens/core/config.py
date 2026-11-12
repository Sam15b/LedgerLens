"""Application configuration.

Two sources, kept deliberately separate:

* ``Settings`` -- secrets and per-environment values from environment variables / ``.env``.
* ``AppConfig`` -- tunable, versioned values (model aliases, chunk sizes, thresholds) from
  ``configs/*.yaml``. Nothing tunable is hard-coded in Python; it lives in these files.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/ledgerlens/core/config.py -> repository root is three levels above the package dir.
_REPO_ROOT = Path(__file__).resolve().parents[3]

# The repo-root .env is found from any working directory (like configs/ is); a ./.env in the
# working directory is read after it, so its values win.
ENV_FILES = (_REPO_ROOT / ".env", Path(".env"))


def repo_path(path: str | Path) -> Path:
    """Resolve a path from a config file: relative paths are taken from the repo root."""
    path = Path(path)
    return path if path.is_absolute() else _REPO_ROOT / path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILES, env_file_encoding="utf-8", extra="ignore")

    # App
    app_env: str = "dev"
    log_level: str = "INFO"
    configs_dir: Path | None = None  # defaults to <repo>/configs, then ./configs
    health_check_timeout_s: float = 3.0

    # PostgreSQL
    database_url: str = "postgresql+psycopg://ledger:ledger@localhost:5432/ledgerlens"

    # Job queue
    redis_url: str = "redis://localhost:6379/0"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "change-me"

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index: str = "ledgerlens-chunks"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_namespace: str = "dev"
    pinecone_host: str = ""

    # LLM providers
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    google_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    mistral_api_key: str = ""
    mistral_base_url: str = "https://api.mistral.ai/v1"
    ollama_base_url: str = "http://localhost:11434"
    aws_region: str = "us-east-1"

    # Embeddings
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768

    # Observability
    langsmith_api_key: str = ""
    langsmith_project: str = "ledgerlens"
    phoenix_collector_endpoint: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()


# --------------------------------------------------------------------------- YAML models


class _Section(BaseModel):
    """Base for config sections: reject unknown keys so typos in YAML fail loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


# models.yaml ------------------------------------------------------------------------


LLM_PROVIDERS = frozenset({"openai", "gemini", "ollama", "mistral", "bedrock"})
"""Provider prefixes allowed in ``provider/model`` references."""


def split_model_ref(ref: str) -> tuple[str, str]:
    """Split ``provider/model`` (e.g. ``ollama/llama3:8b``) into its two parts.

    Only the first ``/`` separates them, so model ids may contain slashes themselves.
    """
    provider, sep, model = ref.partition("/")
    if not sep or not provider or not model:
        raise ValueError(f"model reference {ref!r} must look like 'provider/model'")
    if provider not in LLM_PROVIDERS:
        known = ", ".join(sorted(LLM_PROVIDERS))
        raise ValueError(f"unknown provider {provider!r} in {ref!r} (known: {known})")
    return provider, model


class ModelPrice(_Section):
    input: float | None = Field(default=None, alias="in", ge=0)
    output: float | None = Field(default=None, alias="out", ge=0)


class RetryConfig(_Section):
    max_attempts: int = Field(default=3, ge=1)
    """Attempts per provider before falling back to the next one."""
    backoff_s: list[Annotated[float, Field(ge=0)]] = Field(
        default_factory=lambda: [1.0, 2.0, 4.0], min_length=1
    )
    """Wait before retry n is ``backoff_s[n-1]``; the last value repeats."""

    def delay_for(self, retry: int) -> float:
        """Seconds to wait before the ``retry``-th retry (1-based)."""
        return self.backoff_s[min(max(retry, 1), len(self.backoff_s)) - 1]


class ModelsConfig(_Section):
    aliases: dict[str, list[str]] = Field(default_factory=dict)
    pricing_usd_per_1m_tokens: dict[str, ModelPrice] = Field(default_factory=dict)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    timeout_s: float = Field(default=60, gt=0)
    cost_log: str | None = None
    """JSONL file (relative to the repo root) that every priced call is appended to."""

    @field_validator("aliases")
    @classmethod
    def _check_aliases(cls, aliases: dict[str, list[str]]) -> dict[str, list[str]]:
        for name, refs in aliases.items():
            for ref in refs:
                split_model_ref(ref)
            if len(set(refs)) != len(refs):
                raise ValueError(f"alias {name!r} lists the same model more than once")
        return aliases

    @field_validator("pricing_usd_per_1m_tokens")
    @classmethod
    def _check_pricing_keys(cls, prices: dict[str, ModelPrice]) -> dict[str, ModelPrice]:
        for ref in prices:
            split_model_ref(ref)
        return prices

    def resolve(self, alias: str) -> list[str]:
        """Return the ordered ``provider/model`` fallback list for an alias.

        An explicit ``provider/model`` reference is accepted too and resolves to itself.
        """
        if alias in self.aliases:
            refs = self.aliases[alias]
            if not refs:
                raise ValueError(f"model alias {alias!r} has no models in models.yaml")
            return list(refs)
        if "/" in alias:
            split_model_ref(alias)
            return [alias]
        known = ", ".join(sorted(self.aliases)) or "none"
        raise KeyError(f"unknown model alias {alias!r} (known: {known})")


# pinecone.yaml ----------------------------------------------------------------------


class PineconeConfig(_Section):
    index: str = "ledgerlens-chunks"
    dimension: int = Field(default=768, gt=0)
    metric: Literal["dotproduct", "cosine", "euclidean"] = "dotproduct"
    spec: Literal["serverless", "pod"] = "serverless"
    namespaces: list[str] = Field(default_factory=lambda: ["dev", "eval", "prod"])
    upsert_batch_size: int = Field(default=100, gt=0)
    hybrid_alpha: float = Field(default=0.7, ge=0, le=1)
    query_top_k: int = Field(default=20, gt=0)
    metadata_text_max_bytes: int = Field(default=1024, gt=0)


# ingestion.yaml ---------------------------------------------------------------------


class LoaderConfig(_Section):
    max_file_mb: float = Field(default=25, gt=0)
    allowed_mime_types: list[str] = Field(
        default_factory=lambda: ["application/pdf", "image/png", "image/jpeg", "image/tiff"]
    )


class PreprocessConfig(_Section):
    deskew: bool = True
    denoise: bool = True
    binarize: bool = False


class OCRConfig(_Section):
    engine: Literal["tesseract", "paddle"] = "tesseract"
    dpi: int = Field(default=300, gt=0)
    min_confidence: float = Field(default=40, ge=0, le=100)
    tesseract_psm: int = Field(default=6, ge=0, le=13)
    tesseract_cmd: str | None = None
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)


class LayoutConfig(_Section):
    text_layer_min_chars: int = Field(default=50, ge=0)


class ChunkSpec(_Section):
    max_tokens: int = Field(default=512, gt=0)
    overlap: int = Field(default=50, ge=0)


class ChunkingConfig(_Section):
    context_header: bool = True
    contract: ChunkSpec = Field(default_factory=ChunkSpec)


class EmbeddingConfig(_Section):
    dense_model: str = "BAAI/bge-base-en-v1.5"
    batch_size: int = Field(default=32, gt=0)
    normalize: bool = True
    sparse: Literal["bm25"] = "bm25"
    sparse_params_path: str = "data/processed/bm25.json"


class ExtractionConfig(_Section):
    default: Literal["rules", "llm", "finetuned"] = "llm"
    model_alias: str = "extractor"
    day_first: bool = True  # read 03/04/2026 as 3 April (most of the world), not March 4


class IngestionConfig(_Section):
    loader: LoaderConfig = Field(default_factory=LoaderConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    layout: LayoutConfig = Field(default_factory=LayoutConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)


# agents.yaml ------------------------------------------------------------------------


class AgentConfig(BaseModel):
    """Per-agent settings; agent-specific knobs (e.g. tolerances) are allowed as extras."""

    model_config = ConfigDict(extra="allow", frozen=True)

    model: str


class AgentsConfig(_Section):
    max_steps: int = Field(default=8, gt=0)
    agents: dict[str, AgentConfig] = Field(default_factory=dict)


# eval.yaml --------------------------------------------------------------------------


class EvalConfig(_Section):
    golden_set: str = "data/golden/golden_set.jsonl"
    ci_subset_size: int = Field(default=40, gt=0)
    k: int = Field(default=5, gt=0)
    regression_threshold_pts: float = Field(default=3, ge=0)
    targets: dict[str, float] = Field(default_factory=dict)


# aggregate --------------------------------------------------------------------------


class AppConfig(_Section):
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    pinecone: PineconeConfig = Field(default_factory=PineconeConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)


_CONFIG_FILES = ("models", "pinecone", "ingestion", "agents", "eval")


def resolve_configs_dir(override: Path | None = None) -> Path:
    """Find the ``configs/`` directory: explicit override, repo root, then CWD."""
    if override is not None:
        return override
    for candidate in (_REPO_ROOT / "configs", Path.cwd() / "configs"):
        if candidate.is_dir():
            return candidate
    return Path.cwd() / "configs"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def load_config(configs_dir: Path | None = None) -> AppConfig:
    """Load and validate every ``configs/*.yaml`` file. Missing files fall back to defaults."""
    directory = resolve_configs_dir(configs_dir)
    raw = {name: _read_yaml(directory / f"{name}.yaml") for name in _CONFIG_FILES}
    return AppConfig.model_validate(raw)


@lru_cache
def get_config() -> AppConfig:
    return load_config(get_settings().configs_dir)
