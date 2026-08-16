"""Query / RAG API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.services.retrieval.filters import RetrievalFilters


class QueryRequest(BaseModel):
    """Request body for ``POST /query``."""

    query: str = Field(min_length=1, max_length=4000, description="User question")
    top_k: int | None = Field(
        default=None,
        gt=0,
        le=50,
        description="Optional override for number of chunks to retrieve",
    )
    document_ids: list[str] | None = Field(
        default=None,
        description="Optional document IDs to restrict retrieval",
    )
    filenames: list[str] | None = Field(
        default=None,
        description="Optional filenames to restrict retrieval",
    )
    file_types: list[str] | None = Field(
        default=None,
        description="Optional file types to restrict retrieval (e.g. pdf, txt)",
    )
    sections: list[str] | None = Field(
        default=None,
        description="Optional section names to restrict retrieval",
    )
    filters: dict[str, str | int] | None = Field(
        default=None,
        description="Deprecated exact-match metadata filters; prefer structured filter fields",
    )

    @field_validator("document_ids", "filenames", "file_types", "sections")
    @classmethod
    def normalize_filter_lists(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = [item.strip() for item in value if item and item.strip()]
        if not cleaned:
            msg = "Filter list must not contain empty values"
            raise ValueError(msg)
        return cleaned

    def build_retrieval_filters(self) -> RetrievalFilters | None:
        """Convert request fields into validated retrieval filters."""
        return RetrievalFilters.from_query(
            document_ids=self.document_ids,
            filenames=self.filenames,
            file_types=self.file_types,
            sections=self.sections,
            legacy_filters=self.filters,
        )


class CitationResponse(BaseModel):
    """Citation metadata returned with an answer."""

    document_id: str
    filename: str
    file_type: str
    source: str
    page_number: int
    section: str | None = None
    chunk_index: int
    chunk_id: str = Field(description="Vector-store point / chunk reference")
    score: float
    label: str = Field(description="Source label used in the prompt (e.g. S1)")


class QueryResponse(BaseModel):
    """RAG answer with citations."""

    answer: str
    citations: list[CitationResponse] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
