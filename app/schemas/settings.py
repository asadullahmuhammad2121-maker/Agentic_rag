"""Public, non-secret application settings schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ToolStatusResponse(BaseModel):
    """Safe status for a registered agent tool."""

    name: str
    label: str
    enabled: bool
    configured: bool
    available: bool


class GeneralSettingsResponse(BaseModel):
    """General application settings."""

    app_name: str
    app_version: str
    environment: str
    log_level: str
    request_timeout_seconds: float


class RAGSettingsResponse(BaseModel):
    """RAG pipeline configuration (read-only)."""

    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int
    chunk_min_size: int
    chunk_max_size: int
    semantic_similarity_threshold: float
    retrieval_top_k: int
    retrieval_score_threshold: float | None
    hybrid_search_enabled: bool
    hybrid_top_k: int
    vector_search_weight: float
    keyword_search_weight: float
    query_transformation_enabled: bool
    multi_query_enabled: bool
    multi_query_count: int
    context_optimization_enabled: bool
    context_max_chunks: int
    context_max_tokens: int
    context_min_score: float
    reranking_enabled: bool = False


class AgentSettingsResponse(BaseModel):
    """Agent orchestration configuration (read-only)."""

    agent_enabled: bool = True
    agent_max_steps: int
    agent_routing_enabled: bool
    agent_planning_enabled: bool
    agent_runs_persistence_enabled: bool = True
    groq_model: str
    groq_configured: bool
    llm_temperature: float
    llm_max_tokens: int
    tools: list[ToolStatusResponse] = Field(default_factory=list)


class SearchSettingsResponse(BaseModel):
    """Search subsystem configuration (read-only)."""

    vector_search_enabled: bool = True
    bm25_enabled: bool
    web_search_enabled: bool
    web_search_configured: bool
    tavily_max_results: int
    tavily_search_depth: str
    embedding_model: str
    embedding_dimension: int
    qdrant_collection_name: str


class PublicSettingsResponse(BaseModel):
    """User-safe read-only settings exposed to the frontend."""

    read_only: bool = True
    general: GeneralSettingsResponse
    rag: RAGSettingsResponse
    agent: AgentSettingsResponse
    search: SearchSettingsResponse
