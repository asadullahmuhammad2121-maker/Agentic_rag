"""Unit tests for FoundationAgent decisions."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from app.core.exceptions import AgentError
from app.services.agent.foundation import FoundationAgent
from app.services.agent.models import (
    AgentAction,
    AgentActionType,
    AgentObservation,
    AgentRequest,
    AgentStep,
    ToolResult,
)
from app.services.agent.planning.planner import QueryPlanner
from app.services.agent.routing.router import QueryRouter
from app.services.agent.tools.base import Tool
from app.services.agent.tools.rag import RAG_RETRIEVAL_TOOL_NAME
from app.services.agent.tools.registry import ToolRegistry
from app.services.agent.tools.tavily import TAVILY_WEB_SEARCH_TOOL_NAME
from tests.conftest import make_settings

HYBRID_QUERY = (
    "According to my uploaded document, what is RAG, and what are the latest "
    "developments in RAG in 2026?"
)


class _SampleInput(BaseModel):
    query: str


class _SampleOutput(BaseModel):
    value: str


class _NamedTool(Tool):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "test tool"

    @property
    def input_model(self) -> type[BaseModel]:
        return _SampleInput

    @property
    def output_model(self) -> type[BaseModel]:
        return _SampleOutput

    def execute(self, validated_input: BaseModel) -> ToolResult:
        return ToolResult(success=True, output=_SampleOutput(value="ok"))


@pytest.fixture
def agent() -> FoundationAgent:
    settings = make_settings(agent_planning_enabled=False, agent_routing_enabled=False)
    llm = MagicMock()
    return FoundationAgent(QueryRouter(settings, llm), QueryPlanner(settings, llm), settings)


@pytest.fixture
def rag_tools() -> ToolRegistry:
    return ToolRegistry([_NamedTool(RAG_RETRIEVAL_TOOL_NAME)])


@pytest.fixture
def rag_and_tavily_tools() -> ToolRegistry:
    return ToolRegistry(
        [_NamedTool(RAG_RETRIEVAL_TOOL_NAME), _NamedTool(TAVILY_WEB_SEARCH_TOOL_NAME)]
    )


def test_selects_rag_retrieval_for_knowledge_base_query(
    agent: FoundationAgent,
    rag_and_tavily_tools: ToolRegistry,
) -> None:
    request = AgentRequest(query="What is RAG?", top_k=5)
    action = agent.decide(request, tools=rag_and_tavily_tools, history=[])

    assert action.type is AgentActionType.CALL_TOOL
    assert action.tool_name == RAG_RETRIEVAL_TOOL_NAME
    assert action.arguments["query"] == "What is RAG?"
    assert action.arguments["top_k"] == 5


def test_selects_tavily_for_current_web_query(
    agent: FoundationAgent,
    rag_and_tavily_tools: ToolRegistry,
) -> None:
    action = agent.decide(
        AgentRequest(query="What is the latest AI news today?"),
        tools=rag_and_tavily_tools,
        history=[],
    )
    assert action.tool_name == TAVILY_WEB_SEARCH_TOOL_NAME


def test_selects_rag_when_filters_present_even_for_web_query(
    agent: FoundationAgent,
    rag_and_tavily_tools: ToolRegistry,
) -> None:
    action = agent.decide(
        AgentRequest(query="latest updates", document_ids=["doc-1"]),
        tools=rag_and_tavily_tools,
        history=[],
    )
    assert action.tool_name == RAG_RETRIEVAL_TOOL_NAME


def test_selects_rag_when_only_rag_registered(
    agent: FoundationAgent,
    rag_tools: ToolRegistry,
) -> None:
    action = agent.decide(AgentRequest(query="What is RAG?"), tools=rag_tools, history=[])
    assert action.tool_name == RAG_RETRIEVAL_TOOL_NAME


def test_uses_only_registered_tool_when_rag_missing(agent: FoundationAgent) -> None:
    tools = ToolRegistry([_NamedTool("only_tool")])
    action = agent.decide(AgentRequest(query="q"), tools=tools, history=[])
    assert action.tool_name == "only_tool"


def test_raises_when_no_tools_registered(agent: FoundationAgent) -> None:
    with pytest.raises(AgentError) as exc_info:
        agent.decide(AgentRequest(query="q"), tools=ToolRegistry(), history=[])
    assert exc_info.value.details.get("reason") == "missing_tools"


def test_uses_first_tool_when_rag_missing_among_multiple(agent: FoundationAgent) -> None:
    tools = ToolRegistry([_NamedTool("alpha"), _NamedTool("beta")])
    action = agent.decide(AgentRequest(query="q"), tools=tools, history=[])
    assert action.tool_name == "alpha"


def test_selects_both_tools_for_hybrid_query(
    agent: FoundationAgent,
    rag_and_tavily_tools: ToolRegistry,
) -> None:
    action = agent.decide(
        AgentRequest(query=HYBRID_QUERY),
        tools=rag_and_tavily_tools,
        history=[],
    )
    assert action.type is AgentActionType.CALL_TOOLS
    assert action.tool_names == [RAG_RETRIEVAL_TOOL_NAME, TAVILY_WEB_SEARCH_TOOL_NAME]


def test_selects_both_tools_for_legacy_hybrid_query(
    agent: FoundationAgent,
    rag_and_tavily_tools: ToolRegistry,
) -> None:
    action = agent.decide(
        AgentRequest(
            query=(
                "According to my document, what is RAG and how is it being used "
                "in the latest AI systems?"
            )
        ),
        tools=rag_and_tavily_tools,
        history=[],
    )
    assert action.type is AgentActionType.CALL_TOOLS
    assert action.tool_names == [RAG_RETRIEVAL_TOOL_NAME, TAVILY_WEB_SEARCH_TOOL_NAME]


def test_finishes_after_successful_observation(
    agent: FoundationAgent,
    rag_tools: ToolRegistry,
) -> None:
    history = [
        AgentStep(
            action=AgentAction(
                type=AgentActionType.CALL_TOOL,
                tool_name=RAG_RETRIEVAL_TOOL_NAME,
                arguments={"query": "q"},
            ),
            observation=AgentObservation(
                tool_name=RAG_RETRIEVAL_TOOL_NAME,
                success=True,
                answer="Grounded answer",
            ),
        )
    ]
    action = agent.decide(AgentRequest(query="q"), tools=rag_tools, history=history)
    assert action.type is AgentActionType.FINISH
    assert action.answer == "Grounded answer"


def test_failed_observation_raises(
    agent: FoundationAgent,
    rag_tools: ToolRegistry,
) -> None:
    history = [
        AgentStep(
            action=AgentAction(
                type=AgentActionType.CALL_TOOL,
                tool_name=RAG_RETRIEVAL_TOOL_NAME,
                arguments={"query": "q"},
            ),
            observation=AgentObservation(
                tool_name=RAG_RETRIEVAL_TOOL_NAME,
                success=False,
                error="tool failed",
            ),
        )
    ]
    with pytest.raises(AgentError) as exc_info:
        agent.decide(AgentRequest(query="q"), tools=rag_tools, history=history)
    assert exc_info.value.details.get("reason") == "tool_failed"
