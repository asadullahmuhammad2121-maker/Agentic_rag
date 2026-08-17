"""Pydantic models for agent requests, actions, observations, and tool I/O."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AgentActionType(StrEnum):
    """Next step the agent can take."""

    CALL_TOOL = "call_tool"
    CALL_TOOLS = "call_tools"
    EXECUTE_PLAN = "execute_plan"
    FINISH = "finish"


class AgentRequest(BaseModel):
    """Normalized user request handled by the agent orchestrator."""

    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, gt=0, le=50)
    document_ids: list[str] | None = None
    filenames: list[str] | None = None
    file_types: list[str] | None = None
    sections: list[str] | None = None
    filters: dict[str, str | int] | None = None

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            msg = "Query must not be empty"
            raise ValueError(msg)
        return stripped

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

    def tool_arguments(self) -> dict[str, Any]:
        """Serialize request fields into JSON-compatible tool arguments."""
        payload = self.model_dump(exclude_none=True)
        payload.pop("query", None)
        arguments: dict[str, Any] = {"query": self.query}
        arguments.update(payload)
        return arguments


class ToolError(BaseModel):
    """Structured tool failure returned to the agent orchestrator."""

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Structured result from executing a tool."""

    success: bool
    output: BaseModel | None = None
    error: ToolError | None = None

    @model_validator(mode="after")
    def validate_result_shape(self) -> ToolResult:
        if self.success and self.output is None:
            msg = "Successful tool results must include structured output"
            raise ValueError(msg)
        if not self.success and self.error is None:
            msg = "Failed tool results must include an error"
            raise ValueError(msg)
        return self


class RAGRetrievalInput(BaseModel):
    """Structured input for the internal RAG retrieval tool."""

    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, gt=0, le=50)
    document_ids: list[str] | None = None
    filenames: list[str] | None = None
    file_types: list[str] | None = None
    sections: list[str] | None = None
    filters: dict[str, str | int] | None = None

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            msg = "Query must not be empty"
            raise ValueError(msg)
        return stripped


# Backward-compatible alias used by earlier phases.
RAGRetrievalArguments = RAGRetrievalInput


class RetrievedChunkOutput(BaseModel):
    """Structured retrieval chunk returned by the RAG tool."""

    model_config = ConfigDict(frozen=True)

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


class RAGRetrievalOutput(BaseModel):
    """Structured retrieval results from the internal RAG tool."""

    query: str
    chunks: list[RetrievedChunkOutput] = Field(default_factory=list)

    @property
    def result_count(self) -> int:
        return len(self.chunks)

    @property
    def empty(self) -> bool:
        return not self.chunks


class TavilySearchInput(BaseModel):
    """Structured input for the Tavily web search tool."""

    query: str = Field(min_length=1, max_length=4000)
    max_results: int | None = Field(default=None, gt=0, le=20)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            msg = "Query must not be empty"
            raise ValueError(msg)
        return stripped


class WebSearchResultItem(BaseModel):
    """Single web search hit returned by Tavily."""

    model_config = ConfigDict(frozen=True)

    title: str
    url: str
    content: str
    score: float | None = None


class TavilySearchOutput(BaseModel):
    """Structured web search results from Tavily."""

    query: str
    results: list[WebSearchResultItem] = Field(default_factory=list)

    @property
    def result_count(self) -> int:
        return len(self.results)

    @property
    def empty(self) -> bool:
        return not self.results


class RoutingDecision(BaseModel):
    """Validated tool routing plan for a user query."""

    query: str
    tool_names: list[str] = Field(min_length=1)
    reasoning: str | None = None
    used_fallback: bool = False


class RoutingDecisionOutput(BaseModel):
    """Structured JSON routing payload returned by the LLM."""

    tools: list[str] = Field(min_length=1)
    reasoning: str | None = None


class AgentTask(BaseModel):
    """One decomposed sub-query assigned to a registered tool."""

    query: str = Field(min_length=1, max_length=4000)
    tool_name: str = Field(min_length=1)
    reasoning: str | None = None

    @field_validator("query")
    @classmethod
    def strip_task_query(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            msg = "Task query must not be empty"
            raise ValueError(msg)
        return stripped

    @field_validator("tool_name")
    @classmethod
    def strip_tool_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            msg = "Task tool_name must not be empty"
            raise ValueError(msg)
        return stripped


class AgentTaskOutput(BaseModel):
    """Structured task payload returned by the planning LLM."""

    query: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    reasoning: str | None = None


class AgentPlanOutput(BaseModel):
    """Structured JSON planning payload returned by the LLM."""

    tasks: list[AgentTaskOutput] = Field(min_length=1)
    reasoning: str | None = None


class AgentPlan(BaseModel):
    """Validated decomposition plan for a user query."""

    original_query: str
    tasks: list[AgentTask] = Field(min_length=1)
    reasoning: str | None = None
    used_fallback: bool = False


class AgentCitation(BaseModel):
    """Citation produced after generation (mirrors RAG citations)."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    filename: str
    file_type: str
    source: str
    page_number: int
    section: str | None = None
    chunk_index: int
    chunk_id: str
    score: float
    label: str


class AgentAction(BaseModel):
    """Decision emitted by an agent: execute tool(s) or finish the run."""

    type: AgentActionType
    tool_name: str | None = None
    tool_names: list[str] = Field(default_factory=list)
    tasks: list[AgentTask] = Field(default_factory=list)
    arguments: dict[str, Any] = Field(default_factory=dict)
    reasoning: str | None = None
    answer: str | None = None

    @model_validator(mode="after")
    def validate_action_shape(self) -> AgentAction:
        if self.type in {AgentActionType.CALL_TOOL, AgentActionType.CALL_TOOLS}:
            if self.tool_name and not self.tool_names:
                self.tool_names = [self.tool_name.strip()]
            if self.tool_names and not self.tool_name and len(self.tool_names) == 1:
                self.tool_name = self.tool_names[0]
            cleaned = [name.strip() for name in self.tool_names if name and name.strip()]
            if not cleaned:
                msg = "At least one tool_name is required for tool execution actions"
                raise ValueError(msg)
            self.tool_names = list(dict.fromkeys(cleaned))
            if len(self.tool_names) == 1:
                self.tool_name = self.tool_names[0]
        if self.type is AgentActionType.EXECUTE_PLAN:
            if not self.tasks:
                msg = "At least one task is required when type is execute_plan"
                raise ValueError(msg)
            self.tool_names = list(dict.fromkeys(task.tool_name for task in self.tasks))
            if len(self.tool_names) == 1:
                self.tool_name = self.tool_names[0]
        if self.type is AgentActionType.FINISH and self.answer is None:
            msg = "answer is required when type is finish"
            raise ValueError(msg)
        return self


class AgentObservation(BaseModel):
    """Observation after tool execution and optional answer generation."""

    tool_name: str
    success: bool
    tool_names: list[str] = Field(default_factory=list)
    tool_output: dict[str, Any] | None = None
    answer: str | None = None
    citations: list[AgentCitation] = Field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentStep(BaseModel):
    """One think-act-observe cycle."""

    action: AgentAction
    observation: AgentObservation | None = None


class AgentRunResult(BaseModel):
    """Final orchestrator result after the agent finishes."""

    answer: str
    citations: list[AgentCitation] = Field(default_factory=list)
    tool_used: str | None = None
    steps: list[AgentStep] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
