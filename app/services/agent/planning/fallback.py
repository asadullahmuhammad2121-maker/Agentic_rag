"""Deterministic planning fallback when LLM planning is unavailable."""

from __future__ import annotations

import re

from app.services.agent.models import AgentPlan, AgentRequest, AgentTask
from app.services.agent.routing.fallback import (
    is_hybrid_query,
    route_with_fallback,
    select_tool_for_subquery,
)
from app.services.agent.tools.rag import RAG_RETRIEVAL_TOOL_NAME
from app.services.agent.tools.registry import ToolRegistry
from app.services.agent.tools.tavily import TAVILY_WEB_SEARCH_TOOL_NAME

_HYBRID_SPLIT_PATTERN = re.compile(
    r"^(?P<internal>.+?)(?:,\s*|\s+)and\s+(?P<web>what are the .+)$",
    flags=re.IGNORECASE,
)


def assign_tools_to_plan_tasks(
    tasks: list[AgentTask],
    tools: ToolRegistry,
) -> list[AgentTask]:
    """Reassign each planned task using Phase 3E routing heuristics."""
    return [
        task.model_copy(update={"tool_name": select_tool_for_subquery(task.query, tools)})
        for task in tasks
    ]


def hybrid_plan_needs_fallback(tasks: list[AgentTask], tools: ToolRegistry) -> bool:
    """Return True when a hybrid plan still lacks distinct tool coverage."""
    if len(tasks) < 2:
        return True
    if not _can_plan_hybrid(tools):
        return False
    return len({task.tool_name for task in tasks}) < 2


def plan_with_fallback(request: AgentRequest, tools: ToolRegistry) -> AgentPlan:
    """Build a deterministic plan using hybrid heuristics or Phase 3E routing."""
    if is_hybrid_query(request.query) and _can_plan_hybrid(tools):
        tasks = _fallback_hybrid_tasks(request.query, tools)
        return AgentPlan(
            original_query=request.query,
            tasks=tasks,
            reasoning=(
                "Fallback planning split the hybrid query into document and web sub-queries."
            ),
            used_fallback=True,
        )

    routing = route_with_fallback(request, tools)
    return AgentPlan(
        original_query=request.query,
        tasks=[
            AgentTask(
                query=request.query,
                tool_name=tool_name,
                reasoning=routing.reasoning,
            )
            for tool_name in routing.tool_names
        ],
        reasoning=routing.reasoning,
        used_fallback=True,
    )


def should_attempt_decomposition(
    request: AgentRequest,
    tools: ToolRegistry,
    *,
    planning_enabled: bool,
) -> bool:
    """Return True when query decomposition should be attempted."""
    if not planning_enabled:
        return False
    if len(tools.names()) < 2:
        return False
    if request.document_ids or request.filenames or request.file_types or request.sections:
        return False
    return is_hybrid_query(request.query)


def _can_plan_hybrid(tools: ToolRegistry) -> bool:
    return RAG_RETRIEVAL_TOOL_NAME in tools and TAVILY_WEB_SEARCH_TOOL_NAME in tools


def _fallback_hybrid_tasks(query: str, tools: ToolRegistry) -> list[AgentTask]:
    match = _HYBRID_SPLIT_PATTERN.match(query.strip())
    if match:
        internal_query = match.group("internal").strip().rstrip(",")
        if not internal_query.endswith("?"):
            internal_query = f"{internal_query}?"
        web_query = match.group("web").strip()
        if web_query and web_query[0].islower():
            web_query = web_query[0].upper() + web_query[1:]
        if not web_query.endswith("?"):
            web_query = f"{web_query}?"
        tasks = [
            AgentTask(
                query=internal_query,
                tool_name=RAG_RETRIEVAL_TOOL_NAME,
                reasoning="Document-focused sub-query for internal retrieval.",
            ),
            AgentTask(
                query=web_query,
                tool_name=TAVILY_WEB_SEARCH_TOOL_NAME,
                reasoning="Web-focused sub-query for current external information.",
            ),
        ]
        return [task for task in tasks if task.tool_name in tools]

    return [
        AgentTask(
            query=_document_focused_query(query),
            tool_name=RAG_RETRIEVAL_TOOL_NAME,
            reasoning="Document-focused sub-query for internal retrieval.",
        ),
        AgentTask(
            query=_web_focused_query(query),
            tool_name=TAVILY_WEB_SEARCH_TOOL_NAME,
            reasoning="Web-focused sub-query for current external information.",
        ),
    ]


def _document_focused_query(query: str) -> str:
    lowered = query.casefold()
    if "uploaded document" in lowered:
        topic = _extract_topic(query)
        return f"What is {topic} according to my uploaded document?"
    return query


def _web_focused_query(query: str) -> str:
    year_match = re.search(r"\b20(?:2[4-9]|3[0-9])\b", query)
    topic = _extract_topic(query)
    if year_match:
        return f"What are the latest developments in {topic} in {year_match.group(0)}?"
    return f"What are the latest developments in {topic}?"


def _extract_topic(query: str) -> str:
    match = re.search(r"what is (.+?)(?:,|\?|$)", query, flags=re.IGNORECASE)
    if match:
        topic = match.group(1).strip()
        for marker in ("according to my uploaded document", "according to my document"):
            topic = topic.replace(marker, "").strip()
        if topic:
            return topic
    return "the topic"
