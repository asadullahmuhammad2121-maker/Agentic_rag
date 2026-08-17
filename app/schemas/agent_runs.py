"""Agent run history API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.agent import AgentStepResponse
from app.schemas.query import CitationResponse


class AgentRunSummaryResponse(BaseModel):
    """Summary item for agent run listing."""

    run_id: str
    query: str
    status: Literal["success", "failure"]
    started_at: datetime
    completed_at: datetime
    duration_ms: int | None = None
    tool_used: str | None = None
    step_count: int = 0
    citation_count: int = 0
    error_message: str | None = None
    error_code: str | None = None


class AgentRunListResponse(BaseModel):
    """Paginated agent run history."""

    runs: list[AgentRunSummaryResponse]
    total: int
    limit: int
    offset: int


class AgentRunDetailResponse(AgentRunSummaryResponse):
    """Full stored agent run payload."""

    answer: str | None = None
    citations: list[CitationResponse] = Field(default_factory=list)
    steps: list[AgentStepResponse] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
