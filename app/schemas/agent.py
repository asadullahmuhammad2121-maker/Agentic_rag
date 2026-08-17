"""Agent API schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.query import CitationResponse, QueryRequest


class AgentQueryRequest(QueryRequest):
    """Request body for ``POST /agent/query``."""


class AgentActionResponse(BaseModel):
    """Safe action trace returned to API clients."""

    type: Literal["call_tool", "call_tools", "execute_plan", "finish"]
    tool_name: str | None = None
    tool_names: list[str] = Field(default_factory=list)
    reasoning: str | None = None


class AgentObservationResponse(BaseModel):
    """Safe observation trace — no full document text."""

    tool_name: str
    success: bool
    citation_count: int = 0


class AgentStepResponse(BaseModel):
    """One recorded decide/act step."""

    action: AgentActionResponse
    observation: AgentObservationResponse | None = None


class AgentQueryResponse(BaseModel):
    """Agent answer with citations and a compact execution trace."""

    answer: str
    citations: list[CitationResponse] = Field(default_factory=list)
    tool_used: str | None = None
    steps: list[AgentStepResponse] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
