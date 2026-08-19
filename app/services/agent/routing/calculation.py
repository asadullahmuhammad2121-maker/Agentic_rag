"""Deterministic routing helpers for calculator tool selection."""

from __future__ import annotations

import re

from app.services.agent.tools.calculator import CALCULATOR_TOOL_NAME
from app.services.agent.tools.rag import RAG_RETRIEVAL_TOOL_NAME
from app.services.agent.tools.registry import ToolRegistry
from app.services.agent.tools.tavily import TAVILY_WEB_SEARCH_TOOL_NAME

# Imported lazily in enrich to avoid circular imports at module load.

_CALCULATION_PATTERN = re.compile(
    r"(\d+\s*[\*x×/÷+\-]\s*\d+)"
    r"|(\d+(?:\.\d+)?\s*%\s*of\s*\d+(?:\.\d+)?)"
    r"|\baverage\s+of\b"
    r"|\b(calculate|compute)\b",
    flags=re.IGNORECASE,
)
_CALCULATION_KEYWORDS: frozenset[str] = frozenset(
    {
        "calculate",
        "compute",
        "sum",
        "average",
        "mean",
        "percentage",
        "percent",
        "increase",
        "decrease",
    }
)
_DOCUMENT_MARKERS: tuple[str, ...] = (
    "uploaded document",
    "my document",
    "knowledge base",
    "ingested",
    "according to my document",
    "in my document",
    "according to the uploaded",
    "in the uploaded",
)


def looks_like_calculation_query(query: str) -> bool:
    """Return True when the query appears to require arithmetic."""
    lowered = query.casefold()
    if _CALCULATION_PATTERN.search(query):
        return True
    if any(keyword in lowered for keyword in _CALCULATION_KEYWORDS) and re.search(
        r"\d", query
    ):
        return True
    return "%" in query and re.search(r"\d", query) is not None


def has_document_markers(query: str) -> bool:
    """Return True when the query explicitly references uploaded documents."""
    lowered = query.casefold()
    return any(marker in lowered for marker in _DOCUMENT_MARKERS)


def enrich_calculation_tool_selection(
    selected: list[str],
    query: str,
    tools: ToolRegistry,
) -> list[str]:
    """Ensure calculation queries include calculator and document+calc include RAG."""
    from app.services.agent.routing.fallback import _looks_like_web_query

    if CALCULATOR_TOOL_NAME not in tools or not looks_like_calculation_query(query):
        return list(dict.fromkeys(selected))

    calc_only = (
        looks_like_calculation_query(query)
        and not has_document_markers(query)
        and not _looks_like_web_query(query)
    )
    if calc_only:
        return [CALCULATOR_TOOL_NAME]

    enriched = list(selected)
    if CALCULATOR_TOOL_NAME not in enriched:
        enriched.append(CALCULATOR_TOOL_NAME)
    if (
        has_document_markers(query)
        and RAG_RETRIEVAL_TOOL_NAME in tools
        and RAG_RETRIEVAL_TOOL_NAME not in enriched
    ):
        enriched.insert(0, RAG_RETRIEVAL_TOOL_NAME)
    return list(dict.fromkeys(enriched))


def select_tools_for_calculation_fallback(
    *,
    query: str,
    tools: ToolRegistry,
    has_rag: bool,
    has_tavily: bool,
    has_calculator: bool,
    looks_internal: bool,
    looks_web: bool,
) -> list[str]:
    """Deterministic tool list for calculation-aware fallback routing."""
    calc = looks_like_calculation_query(query)
    doc = has_document_markers(query)

    if calc and doc:
        selected: list[str] = []
        if has_rag:
            selected.append(RAG_RETRIEVAL_TOOL_NAME)
        if has_calculator:
            selected.append(CALCULATOR_TOOL_NAME)
        return selected

    if calc and not doc and not looks_web and has_calculator:
        return [CALCULATOR_TOOL_NAME]

    if calc and looks_web and not doc:
        selected = []
        if has_tavily:
            selected.append(TAVILY_WEB_SEARCH_TOOL_NAME)
        if has_calculator:
            selected.append(CALCULATOR_TOOL_NAME)
        if selected:
            return selected

    selected = []
    if has_rag and looks_internal:
        selected.append(RAG_RETRIEVAL_TOOL_NAME)
    if has_tavily and looks_web and TAVILY_WEB_SEARCH_TOOL_NAME not in selected:
        selected.append(TAVILY_WEB_SEARCH_TOOL_NAME)
    return selected


def _selection_includes_web_signal(selected: list[str]) -> bool:
    return TAVILY_WEB_SEARCH_TOOL_NAME in selected
