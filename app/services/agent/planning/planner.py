"""LLM-powered query decomposition and task planning."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from app.core.config import Settings
from app.core.exceptions import AgentError, AppError
from app.core.logging import get_logger
from app.services.agent.models import (
    AgentPlan,
    AgentPlanOutput,
    AgentRequest,
    AgentTask,
    AgentTaskOutput,
)
from app.services.agent.planning.fallback import (
    assign_tools_to_plan_tasks,
    hybrid_plan_needs_fallback,
    plan_with_fallback,
    should_attempt_decomposition,
)
from app.services.agent.routing.fallback import is_hybrid_query
from app.services.agent.routing.router import _parse_routing_json
from app.services.agent.tools.rag import RAG_RETRIEVAL_TOOL_NAME
from app.services.agent.tools.registry import ToolRegistry
from app.services.agent.tools.tavily import TAVILY_WEB_SEARCH_TOOL_NAME
from app.services.llm.base import LLMService

logger = get_logger(__name__)

PLANNING_SYSTEM_PROMPT = (
    "You decompose complex user questions into focused sub-queries for retrieval tools. "
    "Choose only from the available tools listed below. "
    "Return valid JSON with keys 'tasks' (array of objects with 'query' and 'tool') "
    "and optional 'reasoning'. "
    "Each task must contain one self-contained sub-query and exactly one tool name. "
    "Use rag_retrieval for uploaded/internal document questions. "
    "Use tavily_web_search for current or external web information. "
    "Use calculator for arithmetic, percentages, averages, or numeric computation. "
    "Use rag_retrieval and calculator together when a document value must be calculated. "
    "Do not invent tool names."
)


class QueryPlanner:
    """Use Groq to decompose hybrid queries into tool-specific sub-queries."""

    def __init__(self, settings: Settings, llm_service: LLMService) -> None:
        self._settings = settings
        self._llm = llm_service

    def plan(self, request: AgentRequest, tools: ToolRegistry) -> AgentPlan:
        """Return a validated plan or fall back to deterministic planning."""
        if not should_attempt_decomposition(
            request,
            tools,
            planning_enabled=self._settings.agent_planning_enabled,
        ):
            return plan_with_fallback(request, tools)

        try:
            return self._plan_with_llm(request, tools)
        except AppError:
            logger.warning(
                "agent_planning_failed",
                extra={
                    "operation": "plan_query",
                    "query_length": len(request.query),
                },
            )
            raise

    def _plan_with_llm(self, request: AgentRequest, tools: ToolRegistry) -> AgentPlan:
        tool_descriptions = "\n".join(
            f"- {tool.name}: {tool.description}" for tool in tools.list_tools()
        )
        user_prompt = (
            f"Available tools:\n{tool_descriptions}\n\n"
            f"User query:\n{request.query}\n\n"
            'Respond with JSON like {"tasks": [{"query": "...", "tool": "rag_retrieval"}], '
            '"reasoning": "..."}'
        )
        logger.info(
            "agent_planning_started",
            extra={
                "operation": "plan_query",
                "available_tools": tools.names(),
                "query_length": len(request.query),
            },
        )
        try:
            raw = self._llm.generate(
                user_prompt,
                system_prompt=PLANNING_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=self._settings.agent_planning_max_tokens,
            )
        except AppError:
            raise

        payload = _parse_planning_json(raw)
        try:
            parsed = AgentPlanOutput.model_validate(payload)
        except ValidationError as exc:
            raise AgentError(
                "Planner returned an invalid planning payload",
                details={"reason": "invalid_planning_output"},
            ) from exc

        tasks = _validate_planned_tasks(parsed.tasks, tools)
        if (
            is_hybrid_query(request.query)
            and RAG_RETRIEVAL_TOOL_NAME in tools
            and TAVILY_WEB_SEARCH_TOOL_NAME in tools
        ):
            tasks = assign_tools_to_plan_tasks(tasks, tools)
            if hybrid_plan_needs_fallback(tasks, tools):
                return plan_with_fallback(request, tools)

        plan = AgentPlan(
            original_query=request.query,
            tasks=tasks,
            reasoning=parsed.reasoning,
            used_fallback=False,
        )
        logger.info(
            "agent_planning_completed",
            extra={
                "operation": "plan_query",
                "task_count": len(plan.tasks),
                "tool_names": [task.tool_name for task in plan.tasks],
                "used_fallback": plan.used_fallback,
                "query_length": len(request.query),
            },
        )
        return plan


def _parse_planning_json(raw: str) -> dict[str, object]:
    try:
        return _parse_routing_json(raw)
    except AgentError:
        text = raw.strip()
        match = re.search(r'"tasks"\s*:\s*\[', text)
        if match is None:
            raise
        start = text.rfind("{", 0, match.start())
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
        if not isinstance(payload, dict):
            raise AgentError(
                "Planner JSON must be an object",
                details={"reason": "invalid_planning_output"},
            ) from None
        return payload


def _validate_planned_tasks(
    planned: list[AgentTaskOutput],
    tools: ToolRegistry,
) -> list[AgentTask]:
    if not planned:
        raise AgentError(
            "Planner did not return any tasks",
            details={"reason": "empty_task_list"},
        )

    validated: list[AgentTask] = []
    for planned_task in planned:
        tool_name = planned_task.tool.strip()
        if tool_name not in tools:
            raise AgentError(
                "Planner selected an unknown tool",
                details={"reason": "unknown_tool", "tool_name": tool_name},
            )
        validated.append(
            AgentTask(
                query=planned_task.query.strip(),
                tool_name=tool_name,
                reasoning=planned_task.reasoning,
            )
        )

    deduped: list[AgentTask] = []
    seen: set[tuple[str, str]] = set()
    for validated_task in validated:
        key = (validated_task.tool_name, validated_task.query.casefold())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(validated_task)
    if not deduped:
        raise AgentError(
            "Planner returned no usable tasks",
            details={"reason": "empty_task_list"},
        )
    return deduped
