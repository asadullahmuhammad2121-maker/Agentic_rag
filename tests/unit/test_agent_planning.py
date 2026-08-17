"""Unit tests for Phase 3F query decomposition and task planning."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

from pydantic import BaseModel

from app.core.exceptions import ProviderError
from app.services.agent.foundation import FoundationAgent
from app.services.agent.models import (
    AgentActionType,
    AgentRequest,
    AgentTask,
    RAGRetrievalOutput,
    RetrievedChunkOutput,
    TavilySearchOutput,
    ToolError,
    ToolResult,
    WebSearchResultItem,
)
from app.services.agent.planning.fallback import (
    assign_tools_to_plan_tasks,
    plan_with_fallback,
    should_attempt_decomposition,
)
from app.services.agent.planning.planner import QueryPlanner
from app.services.agent.routing.router import QueryRouter
from app.services.agent.service import AgentService
from app.services.agent.tools.base import Tool
from app.services.agent.tools.rag import RAG_RETRIEVAL_TOOL_NAME
from app.services.agent.tools.registry import ToolRegistry
from app.services.agent.tools.tavily import TAVILY_WEB_SEARCH_TOOL_NAME
from app.services.rag.service import Citation, RAGResult
from tests.conftest import make_settings

HYBRID_QUERY = (
    "According to my uploaded document, what is RAG, and what are the latest "
    "developments in RAG in 2026?"
)
INTERNAL_SUBQUERY = "According to my uploaded document, what is RAG?"
WEB_SUBQUERY = "What are the latest developments in RAG in 2026?"


class _SampleInput(BaseModel):
    query: str


class _SampleOutput(BaseModel):
    value: str


class _NamedTool(Tool):
    def __init__(self, name: str, result: ToolResult | None = None) -> None:
        self._name = name
        self._result = result or ToolResult(success=True, output=_SampleOutput(value="ok"))
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"{self._name} description"

    @property
    def input_model(self) -> type[BaseModel]:
        return _SampleInput

    @property
    def output_model(self) -> type[BaseModel]:
        return _SampleOutput

    def execute(self, validated_input: BaseModel) -> ToolResult:
        self.calls.append(validated_input.model_dump())
        return self._result


def _tools() -> ToolRegistry:
    return ToolRegistry(
        [_NamedTool(RAG_RETRIEVAL_TOOL_NAME), _NamedTool(TAVILY_WEB_SEARCH_TOOL_NAME)]
    )


def _planner(
    llm_response: str | None = None,
    *,
    llm_side_effect: Exception | None = None,
    planning_enabled: bool = True,
) -> QueryPlanner:
    llm = MagicMock()
    if llm_side_effect is not None:
        llm.generate.side_effect = llm_side_effect
    elif llm_response is not None:
        llm.generate.return_value = llm_response
    settings = make_settings(agent_planning_enabled=planning_enabled)
    return QueryPlanner(settings, llm)


def _foundation_agent(
    planner: QueryPlanner,
    *,
    routing_enabled: bool = True,
) -> FoundationAgent:
    settings = make_settings(
        agent_planning_enabled=planner._settings.agent_planning_enabled,
        agent_routing_enabled=routing_enabled,
    )
    llm = MagicMock()
    router = QueryRouter(settings, llm)
    return FoundationAgent(router, planner, settings)


def _rag_output() -> RAGRetrievalOutput:
    return RAGRetrievalOutput(
        query=INTERNAL_SUBQUERY,
        chunks=[
            RetrievedChunkOutput(
                chunk_id="c1",
                text="Doc text",
                document_id="doc-1",
                filename="a.pdf",
                file_type="pdf",
                source="a.pdf",
                page_number=1,
                section=None,
                chunk_index=0,
                chunking_strategy="fixed",
                score=0.9,
            )
        ],
        result_count=1,
        empty=False,
    )


def _web_output() -> TavilySearchOutput:
    return TavilySearchOutput(
        query=WEB_SUBQUERY,
        results=[
            WebSearchResultItem(
                title="AI News",
                url="https://example.com/ai",
                content="Latest AI update.",
                score=0.8,
            )
        ],
        result_count=1,
        empty=False,
    )


def test_should_attempt_decomposition_only_for_hybrid_queries() -> None:
    tools = _tools()
    settings = make_settings(agent_planning_enabled=True)
    assert should_attempt_decomposition(
        AgentRequest(query=HYBRID_QUERY),
        tools,
        planning_enabled=settings.agent_planning_enabled,
    )
    assert not should_attempt_decomposition(
        AgentRequest(query="What is RAG?"),
        tools,
        planning_enabled=settings.agent_planning_enabled,
    )
    assert not should_attempt_decomposition(
        AgentRequest(query="What is the latest AI news today?"),
        tools,
        planning_enabled=settings.agent_planning_enabled,
    )


def test_fallback_plan_decomposes_manual_hybrid_query() -> None:
    plan = plan_with_fallback(AgentRequest(query=HYBRID_QUERY), _tools())
    assert len(plan.tasks) == 2
    assert plan.tasks[0].tool_name == RAG_RETRIEVAL_TOOL_NAME
    assert plan.tasks[1].tool_name == TAVILY_WEB_SEARCH_TOOL_NAME
    assert plan.tasks[0].query == INTERNAL_SUBQUERY
    assert plan.tasks[1].query == WEB_SUBQUERY


def test_exact_hybrid_query_assigns_rag_then_tavily() -> None:
    """Regression: hybrid query must not assign both sub-tasks to rag_retrieval."""
    planner = _planner(
        json.dumps(
            {
                "tasks": [
                    {"query": INTERNAL_SUBQUERY, "tool": RAG_RETRIEVAL_TOOL_NAME},
                    {"query": WEB_SUBQUERY, "tool": RAG_RETRIEVAL_TOOL_NAME},
                ],
                "reasoning": "incorrect duplicate rag assignment",
            }
        )
    )
    plan = planner.plan(AgentRequest(query=HYBRID_QUERY), _tools())
    assert [task.tool_name for task in plan.tasks] == [
        RAG_RETRIEVAL_TOOL_NAME,
        TAVILY_WEB_SEARCH_TOOL_NAME,
    ]

    action = _foundation_agent(planner).decide(
        AgentRequest(query=HYBRID_QUERY),
        tools=_tools(),
        history=[],
    )
    assert action.type is AgentActionType.EXECUTE_PLAN
    assert action.tool_names == [RAG_RETRIEVAL_TOOL_NAME, TAVILY_WEB_SEARCH_TOOL_NAME]


def test_assign_tools_to_plan_tasks_uses_phase_3e_heuristics() -> None:
    tasks = assign_tools_to_plan_tasks(
        [
            AgentTask(
                query=INTERNAL_SUBQUERY,
                tool_name=TAVILY_WEB_SEARCH_TOOL_NAME,
            ),
            AgentTask(
                query=WEB_SUBQUERY,
                tool_name=RAG_RETRIEVAL_TOOL_NAME,
            ),
        ],
        _tools(),
    )
    assert [task.tool_name for task in tasks] == [
        RAG_RETRIEVAL_TOOL_NAME,
        TAVILY_WEB_SEARCH_TOOL_NAME,
    ]


def test_llm_plan_assigns_tools_to_subqueries() -> None:
    planner = _planner(
        json.dumps(
            {
                "tasks": [
                    {"query": INTERNAL_SUBQUERY, "tool": RAG_RETRIEVAL_TOOL_NAME},
                    {"query": WEB_SUBQUERY, "tool": TAVILY_WEB_SEARCH_TOOL_NAME},
                ],
                "reasoning": "split hybrid query",
            }
        )
    )
    plan = planner.plan(AgentRequest(query=HYBRID_QUERY), _tools())
    assert [task.tool_name for task in plan.tasks] == [
        RAG_RETRIEVAL_TOOL_NAME,
        TAVILY_WEB_SEARCH_TOOL_NAME,
    ]
    assert plan.tasks[0].query == INTERNAL_SUBQUERY
    assert plan.tasks[1].query == WEB_SUBQUERY


def test_foundation_agent_uses_execute_plan_for_hybrid_query() -> None:
    planner = _planner(
        json.dumps(
            {
                "tasks": [
                    {"query": INTERNAL_SUBQUERY, "tool": RAG_RETRIEVAL_TOOL_NAME},
                    {"query": WEB_SUBQUERY, "tool": TAVILY_WEB_SEARCH_TOOL_NAME},
                ]
            }
        )
    )
    action = _foundation_agent(planner).decide(
        AgentRequest(query=HYBRID_QUERY),
        tools=_tools(),
        history=[],
    )
    assert action.type is AgentActionType.EXECUTE_PLAN
    assert action.tool_names == [RAG_RETRIEVAL_TOOL_NAME, TAVILY_WEB_SEARCH_TOOL_NAME]
    assert len(action.tasks) == 2


def test_simple_query_skips_planning_and_uses_routing() -> None:
    planner = _planner(
        json.dumps(
            {
                "tasks": [
                    {"query": "What is RAG?", "tool": RAG_RETRIEVAL_TOOL_NAME},
                ]
            }
        )
    )
    action = _foundation_agent(planner, routing_enabled=False).decide(
        AgentRequest(query="What is RAG?"),
        tools=_tools(),
        history=[],
    )
    assert action.type is AgentActionType.CALL_TOOL
    assert action.tool_name == RAG_RETRIEVAL_TOOL_NAME
    planner._llm.generate.assert_not_called()


def test_planning_failure_falls_back_to_phase_3e_routing() -> None:
    planner = _planner(llm_side_effect=ProviderError("down", provider="groq"))
    action = _foundation_agent(planner, routing_enabled=False).decide(
        AgentRequest(query=HYBRID_QUERY),
        tools=_tools(),
        history=[],
    )
    assert action.type is AgentActionType.CALL_TOOLS
    assert action.tool_names == [RAG_RETRIEVAL_TOOL_NAME, TAVILY_WEB_SEARCH_TOOL_NAME]


def test_llm_single_task_hybrid_falls_back_to_two_task_plan() -> None:
    planner = _planner(
        json.dumps({"tasks": [{"query": HYBRID_QUERY, "tool": RAG_RETRIEVAL_TOOL_NAME}]})
    )
    plan = planner.plan(AgentRequest(query=HYBRID_QUERY), _tools())
    assert len(plan.tasks) == 2


def test_execute_plan_runs_each_task_with_its_subquery() -> None:
    rag_tool = _NamedTool(RAG_RETRIEVAL_TOOL_NAME, ToolResult(success=True, output=_rag_output()))
    tavily_tool = _NamedTool(
        TAVILY_WEB_SEARCH_TOOL_NAME,
        ToolResult(success=True, output=_web_output()),
    )
    generation = MagicMock()
    generation.generate_from_chunks.return_value = RAGResult(
        answer="Planned hybrid answer",
        citations=[
            Citation(
                document_id="doc-1",
                filename="a.pdf",
                file_type="pdf",
                source="a.pdf",
                page_number=1,
                section=None,
                chunk_index=0,
                chunk_id="c1",
                score=0.9,
                label="S1",
            ),
            Citation(
                document_id="https://example.com/ai",
                filename="AI News",
                file_type="web",
                source="https://example.com/ai",
                page_number=0,
                section=None,
                chunk_index=1,
                chunk_id="https://example.com/ai",
                score=0.8,
                label="S2",
            ),
        ],
    )
    planner = _planner(
        json.dumps(
            {
                "tasks": [
                    {"query": INTERNAL_SUBQUERY, "tool": RAG_RETRIEVAL_TOOL_NAME},
                    {"query": WEB_SUBQUERY, "tool": TAVILY_WEB_SEARCH_TOOL_NAME},
                ]
            }
        )
    )
    service = AgentService(
        agent=_foundation_agent(planner),
        tools=ToolRegistry([rag_tool, tavily_tool]),
        rag_service=generation,
        web_answer_generator=MagicMock(),
        max_steps=2,
    )

    result = service.run(HYBRID_QUERY)

    assert result.steps[0].action.type is AgentActionType.EXECUTE_PLAN
    assert rag_tool.calls[0]["query"] == INTERNAL_SUBQUERY
    assert tavily_tool.calls[0]["query"] == WEB_SUBQUERY
    assert result.tool_used == f"{RAG_RETRIEVAL_TOOL_NAME}+{TAVILY_WEB_SEARCH_TOOL_NAME}"
    assert result.metadata["tool_names"] == [
        RAG_RETRIEVAL_TOOL_NAME,
        TAVILY_WEB_SEARCH_TOOL_NAME,
    ]
    assert {citation.file_type for citation in result.citations} == {"pdf", "web"}


def test_partial_plan_task_failure_preserves_successful_results() -> None:
    rag_tool = _NamedTool(
        RAG_RETRIEVAL_TOOL_NAME,
        ToolResult(
            success=False,
            error=ToolError(code="tool_execution_error", message="rag failed"),
        ),
    )
    tavily_tool = _NamedTool(
        TAVILY_WEB_SEARCH_TOOL_NAME,
        ToolResult(success=True, output=_web_output()),
    )
    web_generator = MagicMock()
    web_generator.generate.return_value = ("Web only answer", [])
    planner = _planner(
        json.dumps(
            {
                "tasks": [
                    {"query": INTERNAL_SUBQUERY, "tool": RAG_RETRIEVAL_TOOL_NAME},
                    {"query": WEB_SUBQUERY, "tool": TAVILY_WEB_SEARCH_TOOL_NAME},
                ]
            }
        )
    )
    service = AgentService(
        agent=_foundation_agent(planner),
        tools=ToolRegistry([rag_tool, tavily_tool]),
        rag_service=MagicMock(),
        web_answer_generator=web_generator,
        max_steps=2,
    )

    result = service.run(HYBRID_QUERY)

    assert result.answer == "Web only answer"
    observation = result.steps[0].observation
    assert observation is not None
    assert observation.metadata.get("partial_success") is True
    assert observation.metadata.get("decomposed") is True


def test_unknown_planned_tool_falls_back_to_routing() -> None:
    planner = _planner(json.dumps({"tasks": [{"query": "q", "tool": "unknown_tool"}]}))
    action = _foundation_agent(planner, routing_enabled=False).decide(
        AgentRequest(query=HYBRID_QUERY),
        tools=_tools(),
        history=[],
    )
    assert action.type is AgentActionType.CALL_TOOLS
    assert action.tool_names == [RAG_RETRIEVAL_TOOL_NAME, TAVILY_WEB_SEARCH_TOOL_NAME]
