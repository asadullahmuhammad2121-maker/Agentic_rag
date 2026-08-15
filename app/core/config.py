"""Application settings loaded from environment variables."""

from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the RAG foundation application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="rag-foundation", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    app_env: Literal["development", "staging", "production", "test"] = Field(
        default="development",
        description="Runtime environment",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Structured logging level",
    )

    # HTTP / future ingestion limits
    request_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    max_upload_file_size_mb: int = Field(default=25, gt=0, le=500)

    # Groq LLM — keys may be empty in Phase 1A (generation not implemented yet)
    groq_api_key: SecretStr = Field(default=SecretStr(""), description="Groq API key")
    groq_model: str = Field(default="llama-3.3-70b-versatile")
    groq_timeout_seconds: float = Field(default=60.0, gt=0, le=600)

    # Hugging Face embeddings — keys may be empty in Phase 1A
    huggingface_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Hugging Face API key",
    )
    huggingface_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    huggingface_timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    embedding_dimension: int = Field(
        default=384,
        gt=0,
        description="Expected embedding vector dimensionality",
    )
    embedding_batch_size: int = Field(
        default=16,
        gt=0,
        le=256,
        description="Max texts per embedding batch",
    )

    # Chunking (Phase 1C)
    chunk_size: int = Field(
        default=500,
        gt=0,
        description="Target chunk size in characters",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        description="Character overlap between consecutive chunks",
    )

    # Retrieval (Phase 1D)
    retrieval_top_k: int = Field(
        default=5,
        gt=0,
        le=50,
        description="Default number of chunks to retrieve",
    )
    retrieval_score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional minimum similarity score for retrieval hits",
    )

    # LLM generation (Phase 1E)
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, gt=0, le=8192)

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection_name: str = Field(default="documents")
    qdrant_timeout_seconds: float = Field(default=10.0, gt=0, le=300)

    @field_validator("qdrant_url")
    @classmethod
    def validate_qdrant_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            msg = "QDRANT_URL must start with http:// or https://"
            raise ValueError(msg)
        return normalized

    @field_validator("groq_model", "huggingface_embedding_model", "qdrant_collection_name")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            msg = "Value must not be empty"
            raise ValueError(msg)
        return stripped

    @model_validator(mode="after")
    def validate_production_secrets(self) -> Self:
        """Require provider API keys when running in production."""
        if self.app_env == "production":
            missing: list[str] = []
            if not self.groq_api_key.get_secret_value().strip():
                missing.append("GROQ_API_KEY")
            if not self.huggingface_api_key.get_secret_value().strip():
                missing.append("HUGGINGFACE_API_KEY")
            if missing:
                msg = f"Missing required production secrets: {', '.join(missing)}"
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_chunking(self) -> Self:
        """Ensure overlap is strictly smaller than chunk size when overlap is used."""
        if self.chunk_overlap >= self.chunk_size:
            msg = "CHUNK_OVERLAP must be smaller than CHUNK_SIZE"
            raise ValueError(msg)
        return self

    @property
    def max_upload_file_size_bytes(self) -> int:
        return self.max_upload_file_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


def clear_settings_cache() -> None:
    """Clear the settings cache (useful in tests)."""
    get_settings.cache_clear()
