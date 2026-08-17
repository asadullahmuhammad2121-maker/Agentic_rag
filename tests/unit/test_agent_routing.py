"""Unit tests for intelligent query routing (Phase 3E)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from app.core.exceptions import AgentError, ProviderError
from app.services.agent.foundation import FoundationAgent
from app.services.agent.generation.combined import merge_tool_outputs_to_chunks
from app.services.agent.models import (
    AgentAction,
    AgentActionType,
    AgentObservation,
    AgentRequest,
    RAGRetrievalOutput,
    RetrievedChunkOutput,
    TavilySearchOutput,
    ToolError,
    ToolResult,
    WebSearchResultItem,
)
from app.services.agent.planning.planner import QueryPlanner
from app.services.agent.routing.fallback import route_with_fallback
from app.services.agent.routing.router import QueryRouter
from app.services.agent.service import AgentService
from app.services.agent.tools.base import Tool
from app.services.agent.tools.rag import RAG_RETRIEVAL_TOOL_NAME, RAGRetrievalTool
from app.services.agent.tools.registry import ToolRegistry
from app.services.agent.tools.tavily import TAVILY_WEB_SEARCH_TOOL_NAME, TavilyWebSearchTool
from app.services.rag.service import Citation, RAGResult, RetrievalContext
from app.services.retrieval.service import RetrievedChunk
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


class _ScriptedAgent:
    def __init__(self, actions: list[AgentAction]) -> None:
        self._actions = list(actions)

    def decide(self, request: AgentRequest, *, tools: ToolRegistry, history: list[Any]) -> AgentAction:
        del request, tools, history
        if not self._actions:
            raise AssertionError("No scripted actions remaining")
        return self._actions.pop(0)


def _foundation_agent(
    router: QueryRouter,
    *,
    planning_enabled: bool = False,
) -> FoundationAgent:
    settings = make_settings(
        agent_planning_enabled=planning_enabled,
        agent_routing_enabled=True,
    )
    llm = MagicMock()
    return FoundationAgent(router, QueryPlanner(settings, llm), settings)


def _tools() -> ToolRegistry:
    return ToolRegistry(
        [_NamedTool(RAG_RETRIEVAL_TOOL_NAME), _NamedTool(TAVILY_WEB_SEARCH_TOOL_NAME)]
    )


def _router(
    llm_response: str | None = None,
    *,
    llm_side_effect: Exception | None = None,
    routing_enabled: bool = True,
) -> QueryRouter:
    llm = MagicMock()
    if llm_side_effect is not None:
        llm.generate.side_effect = llm_side_effect
    elif llm_response is not None:
        llm.generate.return_value = llm_response
    settings = make_settings(agent_routing_enabled=routing_enabled)
    return QueryRouter(settings, llm)


def _rag_output() -> RAGRetrievalOutput:
    return RAGRetrievalOutput(
        query="q",
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
        query="q",
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


def test_llm_routes_to_rag_only() -> None:
    router = _router(json.dumps({"tools": [RAG_RETRIEVAL_TOOL_NAME], "reasoning": "internal doc"}))
    decision = router.route(AgentRequest(query="What does my uploaded document say about RAG?"), _tools())
    assert decision.tool_names == [RAG_RETRIEVAL_TOOL_NAME]
    assert decision.used_fallback is False


def test_llm_routes_to_tavily_only() -> None:
    router = _router(
        json.dumps({"tools": [TAVILY_WEB_SEARCH_TOOL_NAME], "reasoning": "current events"})
    )
    decision = router.route(
        AgentRequest(query="What are the latest AI developments in 2026?"),
        _tools(),
    )
    assert decision.tool_names == [TAVILY_WEB_SEARCH_TOOL_NAME]


def test_llm_routes_to_rag_and_tavily() -> None:
    router = _router(
        json.dumps(
            {
                "tools": [RAG_RETRIEVAL_TOOL_NAME, TAVILY_WEB_SEARCH_TOOL_NAME],
                "reasoning": "needs both",
            }
        )
    )
    decision = router.route(
        AgentRequest(
            query=(
                "According to my document, what is RAG and how is it used "
                "in the latest AI systems?"
            )
        ),
        _tools(),
    )
    assert decision.tool_names == [RAG_RETRIEVAL_TOOL_NAME, TAVILY_WEB_SEARCH_TOOL_NAME]


def test_llm_enriches_rag_only_selection_for_manual_hybrid_query() -> None:
    router = _router(
        json.dumps({"tools": [RAG_RETRIEVAL_TOOL_NAME], "reasoning": "document question"})
    )
    decision = router.route(AgentRequest(query=HYBRID_QUERY), _tools())
    assert decision.tool_names == [RAG_RETRIEVAL_TOOL_NAME, TAVILY_WEB_SEARCH_TOOL_NAME]
    assert decision.used_fallback is False


def test_fallback_selects_both_tools_for_manual_hybrid_query() -> None:
    decision = route_with_fallback(AgentRequest(query=HYBRID_QUERY), _tools())
    assert decision.tool_names == [RAG_RETRIEVAL_TOOL_NAME, TAVILY_WEB_SEARCH_TOOL_NAME]


def test_foundation_agent_selects_both_tools_for_manual_hybrid_query() -> None:
    router = _router(
        json.dumps({"tools": [RAG_RETRIEVAL_TOOL_NAME], "reasoning": "document question"})
    )
    action = _foundation_agent(router).decide(
        AgentRequest(query=HYBRID_QUERY),
        tools=_tools(),
        history=[],
    )
    assert action.type is AgentActionType.CALL_TOOLS
    assert action.tool_names == [RAG_RETRIEVAL_TOOL_NAME, TAVILY_WEB_SEARCH_TOOL_NAME]


def test_parse_routing_json_extracts_embedded_object() -> None:
    from app.services.agent.routing.router import _parse_routing_json

    payload = _parse_routing_json(
        'Here is the routing decision: {"tools": ["rag_retrieval"], "reasoning": "doc"}'
    )
    assert payload == {"tools": ["rag_retrieval"], "reasoning": "doc"}


def test_invalid_routing_output_uses_fallback() -> None:
    router = _router("not-json")
    decision = router.route(AgentRequest(query="What is RAG?"), _tools())
    assert decision.used_fallback is True
    assert RAG_RETRIEVAL_TOOL_NAME in decision.tool_names


def test_unknown_tool_selection_uses_fallback() -> None:
    router = _router(json.dumps({"tools": ["unknown_tool"]}))
    decision = router.route(AgentRequest(query="What is RAG?"), _tools())
    assert decision.used_fallback is True
    assert all(
        name in {RAG_RETRIEVAL_TOOL_NAME, TAVILY_WEB_SEARCH_TOOL_NAME}
        for name in decision.tool_names
    )


def test_tool_registry_integration_via_foundation_agent() -> None:
    router = _router(json.dumps({"tools": [TAVILY_WEB_SEARCH_TOOL_NAME]}))
    agent = _foundation_agent(router)
    action = agent.decide(AgentRequest(query="latest news"), tools=_tools(), history=[])
    assert action.tool_name == TAVILY_WEB_SEARCH_TOOL_NAME


def test_rag_execution_through_selected_route() -> None:
    rag = MagicMock()
    rag.retrieve_context.return_value = RetrievalContext(query="q", chunks=[])
    generation = MagicMock()
    generation.generate_from_chunks.return_value = RAGResult(answer="ok", citations=[])
    router = _router(json.dumps({"tools": [RAG_RETRIEVAL_TOOL_NAME]}))
    service = AgentService(
        agent=_foundation_agent(router),
        tools=ToolRegistry([RAGRetrievalTool(rag)]),
        rag_service=generation,
        web_answer_generator=MagicMock(),
        max_steps=2,
    )
    result = service.run("What does my document say?")
    assert result.tool_used == RAG_RETRIEVAL_TOOL_NAME
    rag.retrieve_context.assert_called_once()


def test_tavily_execution_through_selected_route() -> None:
    settings = make_settings(tavily_enabled=True, tavily_api_key="test-key")
    tavily_client = MagicMock()
    tavily_client.search.return_value = {
        "query": "latest",
        "results": [
            {
                "title": "AI News",
                "url": "https://example.com/ai",
                "content": "Update",
                "score": 0.9,
            }
        ],
    }
    web_generator = MagicMock()
    web_generator.generate.return_value = ("Web answer", [])
    router = _router(json.dumps({"tools": [TAVILY_WEB_SEARCH_TOOL_NAME]}))
    service = AgentService(
        agent=_foundation_agent(router),
        tools=ToolRegistry(
            [
                RAGRetrievalTool(MagicMock()),
                TavilyWebSearchTool(settings, client=tavily_client),
            ]
        ),
        rag_service=MagicMock(),
        web_answer_generator=web_generator,
        max_steps=2,
    )
    result = service.run("What are the latest AI developments in 2026?")
    assert result.tool_used == TAVILY_WEB_SEARCH_TOOL_NAME
    tavily_client.search.assert_called_once()


def test_both_tools_execute_successfully() -> None:
    rag_tool = _NamedTool(
        RAG_RETRIEVAL_TOOL_NAME,
        ToolResult(success=True, output=_rag_output()),
    )
    tavily_tool = _NamedTool(
        TAVILY_WEB_SEARCH_TOOL_NAME,
        ToolResult(success=True, output=_web_output()),
    )
    generation = MagicMock()
    generation.generate_from_chunks.return_value = RAGResult(
        answer="Combined answer",
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
    agent = _ScriptedAgent(
        [
            AgentAction(
                type=AgentActionType.CALL_TOOLS,
                tool_names=[RAG_RETRIEVAL_TOOL_NAME, TAVILY_WEB_SEARCH_TOOL_NAME],
                arguments={"query": "hybrid"},
            ),
            AgentAction(type=AgentActionType.FINISH, answer="Combined answer"),
        ]
    )
    service = AgentService(
        agent=agent,
        tools=ToolRegistry([rag_tool, tavily_tool]),
        rag_service=generation,
        web_answer_generator=MagicMock(),
        max_steps=2,
    )
    result = service.run("hybrid")
    assert rag_tool.calls
    assert tavily_tool.calls
    assert result.tool_used == f"{RAG_RETRIEVAL_TOOL_NAME}+{TAVILY_WEB_SEARCH_TOOL_NAME}"
    generation.generate_from_chunks.assert_called_once()


def test_one_tool_fails_while_other_succeeds() -> None:
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
    agent = _ScriptedAgent(
        [
            AgentAction(
                type=AgentActionType.CALL_TOOLS,
                tool_names=[RAG_RETRIEVAL_TOOL_NAME, TAVILY_WEB_SEARCH_TOOL_NAME],
                arguments={"query": "hybrid"},
            ),
            AgentAction(type=AgentActionType.FINISH, answer="Web only answer"),
        ]
    )
    service = AgentService(
        agent=agent,
        tools=ToolRegistry([rag_tool, tavily_tool]),
        rag_service=MagicMock(),
        web_answer_generator=web_generator,
        max_steps=2,
    )
    result = service.run("hybrid")
    assert result.answer == "Web only answer"
    observation = result.steps[0].observation
    assert observation is not None
    assert observation.metadata.get("partial_success") is True
    assert observation.metadata.get("failed_tools") == [RAG_RETRIEVAL_TOOL_NAME]


def test_both_tools_return_no_results() -> None:
    rag_tool = _NamedTool(
        RAG_RETRIEVAL_TOOL_NAME,
        ToolResult(
            success=True,
            output=RAGRetrievalOutput(query="q", chunks=[], result_count=0, empty=True),
        ),
    )
    tavily_tool = _NamedTool(
        TAVILY_WEB_SEARCH_TOOL_NAME,
        ToolResult(
            success=True,
            output=TavilySearchOutput(query="q", results=[], result_count=0, empty=True),
        ),
    )
    generation = MagicMock()
    generation.generate_from_chunks.return_value = RAGResult(answer="No info found.", citations=[])
    agent = _ScriptedAgent(
        [
            AgentAction(
                type=AgentActionType.CALL_TOOLS,
                tool_names=[RAG_RETRIEVAL_TOOL_NAME, TAVILY_WEB_SEARCH_TOOL_NAME],
                arguments={"query": "empty"},
            )
        ]
    )
    service = AgentService(
        agent=agent,
        tools=ToolRegistry([rag_tool, tavily_tool]),
        rag_service=generation,
        web_answer_generator=MagicMock(),
        max_steps=1,
    )
    result = service.run("empty")
    assert result.answer == "No info found."
    generation.generate_from_chunks.assert_called_once_with("empty", [])


def test_routing_llm_failure_uses_fallback() -> None:
    router = _router(llm_side_effect=ProviderError("down", provider="groq"))
    decision = router.route(AgentRequest(query="What is RAG?"), _tools())
    assert decision.used_fallback is True


def test_final_answer_generation_from_one_tool() -> None:
    rag = MagicMock()
    rag.retrieve_context.return_value = RetrievalContext(
        query="q",
        chunks=[
            RetrievedChunk(
                chunk_id="c1",
                text="ctx",
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
    )
    generation = MagicMock()
    generation.generate_from_chunks.return_value = RAGResult(
        answer="Single tool answer",
        citations=[],
    )
    router = _router(json.dumps({"tools": [RAG_RETRIEVAL_TOOL_NAME]}))
    service = AgentService(
        agent=_foundation_agent(router),
        tools=ToolRegistry([RAGRetrievalTool(rag)]),
        rag_service=generation,
        web_answer_generator=MagicMock(),
        max_steps=2,
    )
    result = service.run("doc question")
    assert result.answer == "Single tool answer"


def test_final_answer_generation_from_multiple_tools() -> None:
    chunks = merge_tool_outputs_to_chunks(rag_output=_rag_output(), web_output=_web_output())
    assert len(chunks) == 2
    generation = MagicMock()
    generation.generate_from_chunks.return_value = RAGResult(answer="Merged answer", citations=[])
    observation = AgentObservation(
        tool_name=f"{RAG_RETRIEVAL_TOOL_NAME}+{TAVILY_WEB_SEARCH_TOOL_NAME}",
        tool_names=[RAG_RETRIEVAL_TOOL_NAME, TAVILY_WEB_SEARCH_TOOL_NAME],
        success=True,
        metadata={
            "multi_tool": True,
            "tool_outputs": {
                RAG_RETRIEVAL_TOOL_NAME: _rag_output().model_dump(),
                TAVILY_WEB_SEARCH_TOOL_NAME: _web_output().model_dump(),
            },
        },
    )
    service = AgentService(
        agent=MagicMock(),
        tools=ToolRegistry([]),
        rag_service=generation,
        web_answer_generator=MagicMock(),
        max_steps=1,
    )
    generated = service._generate_from_combined(AgentRequest(query="hybrid"), observation)
    assert generated.answer == "Merged answer"
    generation.generate_from_chunks.assert_called_once()


def test_advanced_rag_path_unchanged_when_routing_disabled() -> None:
    decision = route_with_fallback(AgentRequest(query="What is RAG?"), _tools())
    assert decision.tool_names == [RAG_RETRIEVAL_TOOL_NAME]


def test_hybrid_run_merges_document_and_web_citations() -> None:
    rag_tool = _NamedTool(
        RAG_RETRIEVAL_TOOL_NAME,
        ToolResult(success=True, output=_rag_output()),
    )
    tavily_tool = _NamedTool(
        TAVILY_WEB_SEARCH_TOOL_NAME,
        ToolResult(success=True, output=_web_output()),
    )
    generation = MagicMock()
    generation.generate_from_chunks.return_value = RAGResult(
        answer="Hybrid answer [S1][S2]",
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
    router = _router(
        json.dumps({"tools": [RAG_RETRIEVAL_TOOL_NAME], "reasoning": "document question"})
    )
    service = AgentService(
        agent=_foundation_agent(router),
        tools=ToolRegistry([rag_tool, tavily_tool]),
        rag_service=generation,
        web_answer_generator=MagicMock(),
        max_steps=2,
    )

    result = service.run(HYBRID_QUERY)

    assert result.tool_used == f"{RAG_RETRIEVAL_TOOL_NAME}+{TAVILY_WEB_SEARCH_TOOL_NAME}"
    assert result.metadata["tool_names"] == [
        RAG_RETRIEVAL_TOOL_NAME,
        TAVILY_WEB_SEARCH_TOOL_NAME,
    ]
    assert rag_tool.calls
    assert tavily_tool.calls
    assert len(result.citations) == 2
    assert {citation.file_type for citation in result.citations} == {"pdf", "web"}
    assert result.citations[0].source == "a.pdf"
    assert result.citations[1].source == "https://example.com/ai"
    generation.generate_from_chunks.assert_called_once()
    merged_chunks = generation.generate_from_chunks.call_args.args[1]
    assert len(merged_chunks) == 2
    assert {chunk.file_type for chunk in merged_chunks} == {"pdf", "web"}


def test_no_tools_registered_raises_routing_error() -> None:
    router = _router(json.dumps({"tools": [RAG_RETRIEVAL_TOOL_NAME]}))
    with pytest.raises(AgentError) as exc_info:
        router.route(AgentRequest(query="q"), ToolRegistry())
    assert exc_info.value.details.get("reason") == "missing_tools"
