"""Query / RAG API schemas."""

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for ``POST /query``."""

    query: str = Field(min_length=1, max_length=4000, description="User question")
    top_k: int | None = Field(
        default=None,
        gt=0,
        le=50,
        description="Optional override for number of chunks to retrieve",
    )
    filters: dict[str, str | int] | None = Field(
        default=None,
        description="Optional exact-match metadata filters (e.g. document_id, filename)",
    )


class CitationResponse(BaseModel):
    """Citation metadata returned with an answer."""

    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    chunk_id: str = Field(description="Vector-store point / chunk reference")
    score: float
    label: str = Field(description="Source label used in the prompt (e.g. S1)")


class QueryResponse(BaseModel):
    """RAG answer with citations."""

    answer: str
    citations: list[CitationResponse] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
