"""Unit tests for calculator-aware query routing."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.services.agent.models import AgentRequest
from app.services.agent.routing.calculation import (
    enrich_calculation_tool_selection,
    looks_like_calculation_query,
    select_tools_for_calculation_fallback,
)
from app.services.agent.routing.fallback import route_with_fallback
from app.services.agent.tools.calculator import CALCULATOR_TOOL_NAME, CalculatorTool
from app.services.agent.tools.rag import RAG_RETRIEVAL_TOOL_NAME, RAGRetrievalTool
from app.services.agent.tools.registry import ToolRegistry
from app.services.agent.tools.tavily import TAVILY_WEB_SEARCH_TOOL_NAME, TavilyWebSearchTool
from tests.conftest import make_settings


def _all_tools() -> ToolRegistry:
    settings = make_settings(tavily_enabled=True, tavily_api_key="test-key", calculator_enabled=True)
    return ToolRegistry(
        [
            RAGRetrievalTool(MagicMock()),
            TavilyWebSearchTool(settings, client=MagicMock()),
            CalculatorTool(settings),
        ]
    )


def test_looks_like_calculation_query() -> None:
    assert looks_like_calculation_query("What is 25 * 48?") is True
    assert looks_like_calculation_query("What is RAG?") is False


def test_fallback_routes_calculation_only() -> None:
    decision = route_with_fallback(AgentRequest(query="What is 25 * 48?"), _all_tools())
    assert decision.tool_names == [CALCULATOR_TOOL_NAME]


def test_fallback_routes_rag_only_informational_query() -> None:
    decision = route_with_fallback(
        AgentRequest(query="What is retrieval augmented generation in my documents?"),
        _all_tools(),
    )
    assert decision.tool_names == [RAG_RETRIEVAL_TOOL_NAME]


def test_fallback_routes_web_only_query() -> None:
    decision = route_with_fallback(
        AgentRequest(query="What happened in AI in 2026?"),
        _all_tools(),
    )
    assert TAVILY_WEB_SEARCH_TOOL_NAME in decision.tool_names
    assert CALCULATOR_TOOL_NAME not in decision.tool_names


def test_fallback_routes_document_plus_calculator() -> None:
    decision = route_with_fallback(
        AgentRequest(query="What is 20% of the amount in my document?"),
        _all_tools(),
    )
    assert decision.tool_names == [RAG_RETRIEVAL_TOOL_NAME, CALCULATOR_TOOL_NAME]


def test_enrich_adds_calculator_for_doc_calculation() -> None:
    tools = _all_tools()
    enriched = enrich_calculation_tool_selection(
        [RAG_RETRIEVAL_TOOL_NAME],
        "According to my uploaded document, what is a 15% increase on revenue?",
        tools,
    )
    assert enriched == [RAG_RETRIEVAL_TOOL_NAME, CALCULATOR_TOOL_NAME]


def test_select_tools_web_plus_calculator() -> None:
    selected = select_tools_for_calculation_fallback(
        query="Search the web for a value and calculate a 15% increase",
        tools=_all_tools(),
        has_rag=True,
        has_tavily=True,
        has_calculator=True,
        looks_internal=False,
        looks_web=True,
    )
    assert TAVILY_WEB_SEARCH_TOOL_NAME in selected
    assert CALCULATOR_TOOL_NAME in selected
