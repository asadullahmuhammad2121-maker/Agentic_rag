"""Application settings loaded from environment variables."""

from functools import lru_cache
from typing import Literal, Self

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
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
    request_timeout_seconds: float = Field(default=120.0, gt=0, le=600)
    max_upload_file_size_mb: int = Field(default=25, gt=0, le=500)
    app_port: int = Field(default=8000, gt=0, le=65535)
    uvicorn_workers: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Number of Uvicorn worker processes per container",
    )
    uvicorn_timeout_keep_alive: int = Field(default=5, gt=0, le=120)
    keyword_index_lock_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        le=300,
        description="Max seconds to wait for the BM25 index file lock",
    )

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

    # Chunking (Phase 1C / Phase 2B)
    chunking_strategy: Literal["fixed", "recursive", "semantic", "structure"] = Field(
        default="fixed",
        description="Active chunking strategy",
    )
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
    chunk_min_size: int = Field(
        default=20,
        gt=0,
        description="Minimum chunk size; smaller chunks are merged when possible",
    )
    chunk_max_size: int = Field(
        default=2000,
        gt=0,
        description="Maximum chunk size; larger chunks are split",
    )
    semantic_similarity_threshold: float = Field(
        default=0.72,
        ge=0.0,
        le=1.0,
        description="Cosine similarity breakpoint for semantic chunking",
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

    # Query transformation (Phase 2D)
    query_transformation_enabled: bool = Field(
        default=False,
        description="Rewrite user queries for retrieval using Groq",
    )
    query_transformation_max_tokens: int = Field(
        default=256,
        gt=0,
        le=1024,
        description="Max tokens for query rewriting completions",
    )

    # Multi-query retrieval (Phase 2E)
    multi_query_enabled: bool = Field(
        default=False,
        description="Generate multiple retrieval queries and combine results",
    )
    multi_query_count: int = Field(
        default=3,
        ge=2,
        le=10,
        description="Number of diverse retrieval queries to generate",
    )
    multi_query_max_tokens: int = Field(
        default=512,
        gt=0,
        le=2048,
        description="Max tokens for multi-query generation completions",
    )

    # Hybrid search (Phase 2F)
    hybrid_search_enabled: bool = Field(
        default=False,
        description="Combine vector and BM25 keyword retrieval",
    )
    vector_search_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for vector results in hybrid rank fusion",
    )
    keyword_search_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for keyword results in hybrid rank fusion",
    )
    hybrid_top_k: int = Field(
        default=10,
        gt=0,
        le=50,
        description="Default number of fused chunks to return when hybrid search is enabled",
    )
    keyword_index_path: str = Field(
        default="keyword_index/index.json",
        description="Filesystem path for the BM25 keyword index",
    )

    # Context optimization (Phase 2H)
    context_optimization_enabled: bool = Field(
        default=False,
        description="Optimize retrieved context before prompt construction",
    )
    context_max_chunks: int = Field(
        default=8,
        gt=0,
        le=50,
        description="Maximum number of chunks to include in optimized context",
    )
    context_max_tokens: int = Field(
        default=6000,
        gt=0,
        le=100_000,
        description="Estimated token budget for optimized context chunks",
    )
    context_min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum reliable retrieval score for chunk inclusion",
    )

    # Upload limits
    max_batch_upload_files: int = Field(
        default=20,
        gt=0,
        le=100,
        description="Maximum number of files allowed in one batch upload request",
    )

    # LLM generation (Phase 1E)
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, gt=0, le=8192)

    # Agent (Phase 3A / 3E)
    agent_max_steps: int = Field(
        default=2,
        ge=1,
        le=8,
        description="Maximum decide/act steps per agent run",
    )
    agent_routing_enabled: bool = Field(
        default=True,
        description="Use LLM routing to select agent tools",
    )
    agent_routing_max_tokens: int = Field(
        default=256,
        gt=0,
        le=1024,
        description="Maximum tokens for the routing LLM response",
    )
    agent_planning_enabled: bool = Field(
        default=True,
        description="Decompose hybrid queries into planned sub-tasks",
    )
    agent_planning_max_tokens: int = Field(
        default=512,
        gt=0,
        le=2048,
        description="Maximum tokens for the planning LLM response",
    )

    # Tavily web search (Phase 3D)
    tavily_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("tavily_enabled", "web_search_enabled"),
        description="Enable Tavily web search tool for the agent",
    )
    tavily_api_key: SecretStr = Field(default=SecretStr(""), description="Tavily API key")
    tavily_max_results: int = Field(
        default=5,
        gt=0,
        le=20,
        description="Default maximum Tavily search results to return",
    )
    tavily_search_depth: Literal["basic", "advanced"] = Field(
        default="basic",
        description="Tavily search depth tradeoff",
    )
    tavily_timeout_seconds: float = Field(default=30.0, gt=0, le=300)

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
        """Ensure chunking settings are internally consistent."""
        if self.chunk_overlap >= self.chunk_size:
            msg = "CHUNK_OVERLAP must be smaller than CHUNK_SIZE"
            raise ValueError(msg)
        if self.chunk_min_size > self.chunk_size:
            msg = "CHUNK_MIN_SIZE must not exceed CHUNK_SIZE"
            raise ValueError(msg)
        if self.chunk_max_size < self.chunk_size:
            msg = "CHUNK_MAX_SIZE must be greater than or equal to CHUNK_SIZE"
            raise ValueError(msg)
        return self

    @property
    def max_upload_file_size_bytes(self) -> int:
        return self.max_upload_file_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def tavily_configured(self) -> bool:
        """Return whether Tavily is enabled and has an API key."""
        return self.tavily_enabled and bool(self.tavily_api_key.get_secret_value().strip())


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


def clear_settings_cache() -> None:
    """Clear the settings cache (useful in tests)."""
    get_settings.cache_clear()
