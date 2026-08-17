"""Retrieval explorer API schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.query import QueryRequest


class RetrievalExploreRequest(QueryRequest):
    """Request body for ``POST /retrieval/explore``."""


RetrievalMethod = Literal["vector", "bm25", "hybrid_fusion", "multi_query"]


class RetrievedChunkResponse(BaseModel):
    """A retrieved chunk exposed for retrieval exploration."""

    chunk_id: str
    text: str
    document_id: str
    filename: str
    file_type: str
    source: str
    page_number: int
    section: str | None = None
    chunk_index: int
    chunking_strategy: str
    score: float
    retrieval_method: RetrievalMethod


class PipelineStageResponse(BaseModel):
    """One stage in the retrieval pipeline."""

    id: str
    label: str
    enabled: bool
    executed: bool
    result_count: int | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class RetrievalConfigurationResponse(BaseModel):
    """Active retrieval configuration flags."""

    query_transformation_enabled: bool
    multi_query_enabled: bool
    hybrid_search_enabled: bool
    context_optimization_enabled: bool
    reranking_enabled: bool = False


class RetrievalExploreResponse(BaseModel):
    """Raw retrieval pipeline output for the Retrieval Explorer."""

    query: str
    retrieval_query: str
    generated_queries: list[str] | None = None
    configuration: RetrievalConfigurationResponse
    pipeline: list[PipelineStageResponse]
    vector_results: list[RetrievedChunkResponse] = Field(default_factory=list)
    bm25_results: list[RetrievedChunkResponse] = Field(default_factory=list)
    fused_results: list[RetrievedChunkResponse] | None = None
    results: list[RetrievedChunkResponse] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
