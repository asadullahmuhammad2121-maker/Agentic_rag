"""Unit tests for AgentService orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from app.core.exceptions import AgentError, ProviderError, QueryError
from app.services.agent.base import Agent
from app.services.agent.foundation import FoundationAgent
from app.services.agent.models import (
    AgentAction,
    AgentActionType,
    AgentCitation,
    AgentRequest,
    AgentStep,
    RAGRetrievalOutput,
    ToolError,
    ToolResult,
)
from app.services.agent.planning.planner import QueryPlanner
from app.services.agent.routing.router import QueryRouter
from app.services.agent.service import AgentService
from app.services.agent.tools.base import Tool
from app.services.agent.tools.calculator import CALCULATOR_TOOL_NAME, CalculatorTool
from app.services.agent.tools.rag import RAG_RETRIEVAL_TOOL_NAME, RAGRetrievalTool
from app.services.agent.tools.registry import ToolRegistry
from app.services.agent.tools.tavily import TAVILY_WEB_SEARCH_TOOL_NAME, TavilyWebSearchTool
from app.services.rag.service import (
    EMPTY_RETRIEVAL_ANSWER,
    Citation,
    RAGResult,
    RetrievalContext,
)
from app.services.retrieval.filters import RetrievalFilters
from app.services.retrieval.service import RetrievedChunk
from tests.conftest import make_settings


class _ScriptedAgent(Agent):
    def __init__(self, actions: list[AgentAction]) -> None:
        self._actions = list(actions)

    def decide(
        self,
        request: AgentRequest,
        *,
        tools: ToolRegistry,
        history: Sequence[AgentStep],
    ) -> AgentAction:
        if not self._actions:
            raise AssertionError("No scripted actions remaining")
        return self._actions.pop(0)


class _SampleInput(BaseModel):
    query: str


class _SampleOutput(BaseModel):
    answer: str


class _NamedTool(Tool):
    def __init__(self, name: str, result: ToolResult) -> None:
        self._name = name
        self._result = result
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(validated_input.model_dump())
        return self._result


class _FakeTavilyClient:
    def search(
        self,
        query: str,
        *,
        max_results: int,
        search_depth: str,
    ) -> dict[str, Any]:
        return {
            "query": query,
            "results": [
                {
                    "title": "AI News",
                    "url": "https://example.com/ai",
                    "content": "Breaking AI update.",
                    "score": 0.9,
                }
            ],
        }


def _sample_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        text="Grounded context",
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


def _rag_services() -> tuple[MagicMock, MagicMock]:
    retrieval = MagicMock()
    retrieval.retrieve_context.return_value = RetrievalContext(
        query="What is this?",
        chunks=[_sample_chunk()],
    )
    generation = MagicMock()
    generation.generate_from_chunks.return_value = RAGResult(
        answer="Grounded answer [S1]",
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
            )
        ],
    )
    return retrieval, generation


def _web_generator() -> MagicMock:
    generator = MagicMock()
    generator.generate.return_value = (
        "Web summary [S1]",
        [
            AgentCitation(
                document_id="https://example.com/ai",
                filename="AI News",
                file_type="web",
                source="https://example.com/ai",
                page_number=0,
                section=None,
                chunk_index=0,
                chunk_id="https://example.com/ai",
                score=0.9,
                label="S1",
            )
        ],
    )
    return generator


def _foundation_agent() -> FoundationAgent:
    settings = make_settings(agent_planning_enabled=False, agent_routing_enabled=False)
    llm = MagicMock()
    return FoundationAgent(QueryRouter(settings, llm), QueryPlanner(settings, llm), settings)


def test_run_selects_rag_executes_tool_and_generates_answer() -> None:
    rag, generation = _rag_services()
    service = AgentService(
        agent=_foundation_agent(),
        tools=ToolRegistry([RAGRetrievalTool(rag)]),
        rag_service=generation,
        web_answer_generator=_web_generator(),
        max_steps=2,
    )

    result = service.run("What is this?")

    assert result.answer == "Grounded answer [S1]"
    assert result.tool_used == RAG_RETRIEVAL_TOOL_NAME
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "c1"
    assert len(result.steps) == 2
    assert result.steps[0].action.type is AgentActionType.CALL_TOOL
    assert result.steps[0].observation is not None
    assert result.steps[0].observation.tool_output is not None
    assert result.steps[1].action.type is AgentActionType.FINISH
    assert result.metadata["finished"] is True
    rag.retrieve_context.assert_called_once()
    generation.generate_from_chunks.assert_called_once()


def test_run_executes_tavily_and_generates_web_answer() -> None:
    settings = make_settings(tavily_enabled=True, tavily_api_key="test-tavily-key")
    tavily_tool = TavilyWebSearchTool(settings, client=_FakeTavilyClient())
    web_generator = _web_generator()
    service = AgentService(
        agent=_foundation_agent(),
        tools=ToolRegistry([RAGRetrievalTool(MagicMock()), tavily_tool]),
        rag_service=MagicMock(),
        web_answer_generator=web_generator,
        max_steps=2,
    )

    result = service.run("What is the latest AI news today?")

    assert result.tool_used == TAVILY_WEB_SEARCH_TOOL_NAME
    assert result.answer == "Web summary [S1]"
    assert result.citations[0].source == "https://example.com/ai"
    assert result.citations[0].file_type == "web"
    web_generator.generate.assert_called_once()


def test_run_passes_filters_to_rag_tool() -> None:
    rag, generation = _rag_services()
    generation.generate_from_chunks.return_value = RAGResult(answer="ok", citations=[])
    service = AgentService(
        agent=_foundation_agent(),
        tools=ToolRegistry([RAGRetrievalTool(rag)]),
        rag_service=generation,
        web_answer_generator=_web_generator(),
        max_steps=2,
    )
    filters = RetrievalFilters(document_ids=("doc-1",), filenames=("a.pdf",))

    service.run("filtered question", top_k=7, filters=filters)

    assert rag.retrieve_context.call_args.kwargs["top_k"] == 7
    passed = rag.retrieve_context.call_args.kwargs["filters"]
    assert passed.document_ids == ("doc-1",)
    assert passed.filenames == ("a.pdf",)


def test_empty_query_raises_query_error() -> None:
    service = AgentService(
        agent=_foundation_agent(),
        tools=ToolRegistry([RAGRetrievalTool(MagicMock())]),
        rag_service=MagicMock(),
        web_answer_generator=_web_generator(),
        max_steps=2,
    )
    with pytest.raises(QueryError) as exc_info:
        service.run("   ")
    assert exc_info.value.details.get("reason") == "empty_query"


def test_unknown_tool_raises_agent_error() -> None:
    agent = _ScriptedAgent(
        [
            AgentAction(
                type=AgentActionType.CALL_TOOL,
                tool_name="web_search",
                arguments={"query": "q"},
            )
        ]
    )
    service = AgentService(
        agent=agent,
        tools=ToolRegistry(
            [
                _NamedTool(
                    RAG_RETRIEVAL_TOOL_NAME,
                    ToolResult(success=True, output=_SampleOutput(answer="x")),
                )
            ]
        ),
        rag_service=MagicMock(),
        web_answer_generator=_web_generator(),
        max_steps=2,
    )
    with pytest.raises(AgentError) as exc_info:
        service.run("q")
    assert exc_info.value.details.get("reason") == "unknown_tool"


def test_generation_failure_propagates() -> None:
    rag, generation = _rag_services()
    generation.generate_from_chunks.side_effect = ProviderError("fail", provider="groq")
    service = AgentService(
        agent=_foundation_agent(),
        tools=ToolRegistry([RAGRetrievalTool(rag)]),
        rag_service=generation,
        web_answer_generator=_web_generator(),
        max_steps=2,
    )
    with pytest.raises(ProviderError):
        service.run("What is RAG?")


def test_no_retrieval_results_still_generates_empty_answer() -> None:
    rag = MagicMock()
    rag.retrieve_context.return_value = RetrievalContext(query="unknown", chunks=[])
    generation = MagicMock()
    generation.generate_from_chunks.return_value = RAGResult(
        answer=EMPTY_RETRIEVAL_ANSWER,
        citations=[],
    )
    service = AgentService(
        agent=_foundation_agent(),
        tools=ToolRegistry([RAGRetrievalTool(rag)]),
        rag_service=generation,
        web_answer_generator=_web_generator(),
        max_steps=2,
    )

    result = service.run("unknown topic")

    assert result.answer == EMPTY_RETRIEVAL_ANSWER
    assert result.citations == []
    generation.generate_from_chunks.assert_called_once_with("unknown topic", [])


def test_max_steps_one_returns_generated_answer_without_explicit_finish() -> None:
    rag, generation = _rag_services()
    service = AgentService(
        agent=_foundation_agent(),
        tools=ToolRegistry([RAGRetrievalTool(rag)]),
        rag_service=generation,
        web_answer_generator=_web_generator(),
        max_steps=1,
    )

    result = service.run("q")

    assert result.answer == "Grounded answer [S1]"
    assert result.metadata.get("max_steps_reached") is True
    assert result.metadata["finished"] is False


def test_max_steps_exceeded_without_successful_observation_raises() -> None:
    failing_tool = _NamedTool(
        RAG_RETRIEVAL_TOOL_NAME,
        ToolResult(
            success=False,
            error=ToolError(
                code="tool_execution_error",
                message="retrieval failed",
            ),
        ),
    )
    service = AgentService(
        agent=_foundation_agent(),
        tools=ToolRegistry([failing_tool]),
        rag_service=MagicMock(),
        web_answer_generator=_web_generator(),
        max_steps=1,
    )
    with pytest.raises(AgentError) as exc_info:
        service.run("q")
    assert exc_info.value.details.get("reason") == "max_steps_exceeded"


def test_observation_contains_retrieval_metadata_before_finish() -> None:
    rag, generation = _rag_services()
    service = AgentService(
        agent=_foundation_agent(),
        tools=ToolRegistry([RAGRetrievalTool(rag)]),
        rag_service=generation,
        web_answer_generator=_web_generator(),
        max_steps=2,
    )

    result = service.run("What is this?")
    observation = result.steps[0].observation
    assert observation is not None
    assert observation.metadata["generated"] is True
    assert observation.metadata["result_count"] == 1
    assert RAGRetrievalOutput.model_validate(observation.tool_output).result_count == 1


def test_run_executes_calculator_only() -> None:
    settings = make_settings(calculator_enabled=True)
    calc_tool = CalculatorTool(settings)
    service = AgentService(
        agent=_foundation_agent(),
        tools=ToolRegistry([RAGRetrievalTool(MagicMock()), calc_tool]),
        rag_service=MagicMock(),
        web_answer_generator=_web_generator(),
        max_steps=2,
    )

    result = service.run("What is 17.5% of 84000?")

    assert result.tool_used == CALCULATOR_TOOL_NAME
    assert "14700" in result.answer.replace(",", "")
    assert result.steps[0].observation is not None
    assert result.steps[0].observation.metadata.get("result") == 14700


def test_run_executes_rag_and_calculator() -> None:
    rag, generation = _rag_services()
    revenue_chunk = RetrievedChunk(
        chunk_id="c-rev",
        text="Total revenue for the quarter was $2,400,000.",
        document_id="doc-fin",
        filename="finance.txt",
        file_type="txt",
        source="finance.txt",
        page_number=1,
        section=None,
        chunk_index=0,
        chunking_strategy="fixed",
        score=0.99,
    )
    rag.retrieve_context.return_value = RetrievalContext(
        query="15% increase on revenue",
        chunks=[revenue_chunk],
    )
    generation.generate_from_chunks.return_value = RAGResult(
        answer="A 15% increase on $2,400,000 revenue is $2,760,000.",
        citations=[
            Citation(
                document_id="doc-fin",
                filename="finance.txt",
                file_type="txt",
                source="finance.txt",
                page_number=1,
                section=None,
                chunk_index=0,
                chunk_id="c-rev",
                score=0.99,
                label="S1",
            )
        ],
    )
    settings = make_settings(calculator_enabled=True)
    service = AgentService(
        agent=_foundation_agent(),
        tools=ToolRegistry(
            [
                RAGRetrievalTool(rag),
                CalculatorTool(settings),
            ]
        ),
        rag_service=generation,
        web_answer_generator=_web_generator(),
        max_steps=2,
    )

    result = service.run(
        "According to my uploaded document, what is 20% of 2400000?"
    )

    tool_names = result.metadata.get("tool_names") or []
    assert RAG_RETRIEVAL_TOOL_NAME in tool_names
    assert CALCULATOR_TOOL_NAME in tool_names
    assert result.answer
    generation.generate_from_chunks.assert_called_once()
