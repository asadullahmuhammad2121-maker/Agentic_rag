"""Deterministic fallback routing when LLM routing is unavailable."""

from __future__ import annotations

import re

from app.services.agent.models import AgentRequest, RoutingDecision
from app.services.agent.tools.rag import RAG_RETRIEVAL_TOOL_NAME
from app.services.agent.tools.registry import ToolRegistry
from app.services.agent.tools.tavily import TAVILY_WEB_SEARCH_TOOL_NAME

_WEB_QUERY_KEYWORDS: frozenset[str] = frozenset(
    {
        "latest",
        "current",
        "today",
        "recent",
        "news",
        "weather",
        "stock",
        "live",
        "breaking",
        "now",
        "happening",
    }
)
_RECENT_YEAR_PATTERN = re.compile(r"\b20(?:2[4-9]|3[0-9])\b")


def is_hybrid_query(query: str) -> bool:
    """Return True when the query needs both internal documents and web search."""
    return _looks_like_internal_query(query) and _looks_like_web_query(query)


def enrich_hybrid_tool_selection(
    selected: list[str],
    query: str,
    tools: ToolRegistry,
) -> list[str]:
    """Ensure hybrid queries include both RAG and Tavily when available."""
    if not is_hybrid_query(query):
        return list(dict.fromkeys(selected))

    enriched = list(selected)
    if RAG_RETRIEVAL_TOOL_NAME in tools and RAG_RETRIEVAL_TOOL_NAME not in enriched:
        enriched.insert(0, RAG_RETRIEVAL_TOOL_NAME)
    if TAVILY_WEB_SEARCH_TOOL_NAME in tools and TAVILY_WEB_SEARCH_TOOL_NAME not in enriched:
        enriched.append(TAVILY_WEB_SEARCH_TOOL_NAME)
    return list(dict.fromkeys(enriched))


def select_tool_for_subquery(query: str, tools: ToolRegistry) -> str:
    """Select one tool for a decomposed sub-query using Phase 3E heuristics."""
    decision = route_with_fallback(AgentRequest(query=query), tools)
    tool_names = decision.tool_names
    if len(tool_names) == 1:
        return tool_names[0]

    internal = _looks_like_internal_query(query)
    web = _looks_like_web_query(query)
    if internal and not web and RAG_RETRIEVAL_TOOL_NAME in tools:
        return RAG_RETRIEVAL_TOOL_NAME
    if web and not internal and TAVILY_WEB_SEARCH_TOOL_NAME in tools:
        return TAVILY_WEB_SEARCH_TOOL_NAME
    return tool_names[0]


def route_with_fallback(request: AgentRequest, tools: ToolRegistry) -> RoutingDecision:
    """Select tools using simple deterministic rules."""
    names = tools.names()
    has_rag = RAG_RETRIEVAL_TOOL_NAME in tools
    has_tavily = TAVILY_WEB_SEARCH_TOOL_NAME in tools

    if _has_retrieval_filters(request):
        if has_rag:
            return RoutingDecision(
                query=request.query,
                tool_names=[RAG_RETRIEVAL_TOOL_NAME],
                reasoning="Document filters require internal knowledge-base retrieval.",
                used_fallback=True,
            )
        return RoutingDecision(
            query=request.query,
            tool_names=[names[0]],
            reasoning="Fallback routing with the only registered tool.",
            used_fallback=True,
        )

    selected: list[str] = []
    if has_rag and _looks_like_internal_query(request.query):
        selected.append(RAG_RETRIEVAL_TOOL_NAME)
    if (
        has_tavily
        and _looks_like_web_query(request.query)
        and TAVILY_WEB_SEARCH_TOOL_NAME not in selected
    ):
        selected.append(TAVILY_WEB_SEARCH_TOOL_NAME)

    if not selected:
        if has_rag:
            selected = [RAG_RETRIEVAL_TOOL_NAME]
        elif has_tavily:
            selected = [TAVILY_WEB_SEARCH_TOOL_NAME]
        else:
            selected = [names[0]]

    selected = enrich_hybrid_tool_selection(selected, request.query, tools)

    reasoning = "Fallback routing selected the most appropriate registered tool(s)."
    if len(selected) > 1:
        reasoning = (
            "Fallback routing selected both internal documents and web search "
            "because the query needs both sources."
        )
    return RoutingDecision(
        query=request.query,
        tool_names=selected,
        reasoning=reasoning,
        used_fallback=True,
    )


def _has_retrieval_filters(request: AgentRequest) -> bool:
    return bool(
        request.document_ids
        or request.filenames
        or request.file_types
        or request.sections
        or request.filters
    )


def _looks_like_web_query(query: str) -> bool:
    lowered = query.casefold()
    if any(keyword in lowered for keyword in _WEB_QUERY_KEYWORDS):
        return True
    return _RECENT_YEAR_PATTERN.search(query) is not None


def _looks_like_internal_query(query: str) -> bool:
    lowered = query.casefold()
    internal_markers = (
        "uploaded document",
        "my document",
        "knowledge base",
        "ingested",
        "according to my document",
        "in my document",
    )
    return any(marker in lowered for marker in internal_markers) or not _looks_like_web_query(query)
