"""Application settings loaded from environment variables / .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    app_env: str = "dev"
    log_level: str = "INFO"

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
    google_api_key: str = ""
    mistral_api_key: str = ""
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
