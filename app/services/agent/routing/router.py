"""LLM-powered query routing to registered agent tools."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from app.core.config import Settings
from app.core.exceptions import AgentError, AppError
from app.core.logging import get_logger
from app.services.agent.models import AgentRequest, RoutingDecision, RoutingDecisionOutput
from app.services.agent.routing.fallback import enrich_hybrid_tool_selection, route_with_fallback
from app.services.agent.tools.rag import RAG_RETRIEVAL_TOOL_NAME
from app.services.agent.tools.registry import ToolRegistry
from app.services.llm.base import LLMService

logger = get_logger(__name__)

ROUTING_SYSTEM_PROMPT = (
    "You route user questions to the correct retrieval tools. "
    "Choose only from the available tools listed below. "
    "Return valid JSON with keys 'tools' (array of tool names) and optional 'reasoning'. "
    "Select rag_retrieval when the answer should come from uploaded/internal documents. "
    "Select tavily_web_search when the answer needs current or external web information. "
    "Select both when the question needs internal document context and current web information. "
    "Do not invent tool names."
)


class QueryRouter:
    """Use Groq to choose registered tools for a user query."""

    def __init__(self, settings: Settings, llm_service: LLMService) -> None:
        self._settings = settings
        self._llm = llm_service

    def route(self, request: AgentRequest, tools: ToolRegistry) -> RoutingDecision:
        """Return a validated routing decision for the request."""
        if not tools.names():
            raise AgentError(
                "No agent tools are registered",
                details={"reason": "missing_tools"},
            )

        if not self._settings.agent_routing_enabled or len(tools.names()) == 1:
            return route_with_fallback(request, tools)

        try:
            decision = self._route_with_llm(request, tools)
            logger.info(
                "agent_routing_completed",
                extra={
                    "operation": "route_query",
                    "tool_names": decision.tool_names,
                    "tool_count": len(decision.tool_names),
                    "used_fallback": decision.used_fallback,
                    "query_length": len(request.query),
                },
            )
            return decision
        except AppError as exc:
            logger.warning(
                "agent_routing_failed_using_fallback",
                extra={
                    "operation": "route_query",
                    "error_type": type(exc).__name__,
                    "reason": exc.details.get("reason"),
                },
            )
            return route_with_fallback(request, tools)

    def _route_with_llm(self, request: AgentRequest, tools: ToolRegistry) -> RoutingDecision:
        tool_descriptions = "\n".join(
            f"- {tool.name}: {tool.description}" for tool in tools.list_tools()
        )
        user_prompt = (
            f"Available tools:\n{tool_descriptions}\n\n"
            f"User query:\n{request.query}\n\n"
            'Respond with JSON like {"tools": ["rag_retrieval"], "reasoning": "..."}'
        )
        logger.info(
            "agent_routing_started",
            extra={
                "operation": "route_query",
                "available_tools": tools.names(),
                "query_length": len(request.query),
            },
        )
        try:
            raw = self._llm.generate(
                user_prompt,
                system_prompt=ROUTING_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=self._settings.agent_routing_max_tokens,
            )
        except AppError:
            raise

        payload = _parse_routing_json(raw)
        try:
            parsed = RoutingDecisionOutput.model_validate(payload)
        except ValidationError as exc:
            raise AgentError(
                "Router returned an invalid routing payload",
                details={"reason": "invalid_routing_output"},
            ) from exc

        validated_tools = _validate_selected_tools(parsed.tools, tools)
        if (
            (request.document_ids or request.filenames or request.file_types or request.sections)
            and RAG_RETRIEVAL_TOOL_NAME in tools
            and RAG_RETRIEVAL_TOOL_NAME not in validated_tools
        ):
            validated_tools.insert(0, RAG_RETRIEVAL_TOOL_NAME)
        validated_tools = enrich_hybrid_tool_selection(
            validated_tools,
            request.query,
            tools,
        )

        return RoutingDecision(
            query=request.query,
            tool_names=validated_tools,
            reasoning=parsed.reasoning,
            used_fallback=False,
        )


def _parse_routing_json(raw: str) -> dict[str, object]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(payload, dict):
            return payload
    raise AgentError(
        "Router returned malformed JSON",
        details={"reason": "invalid_routing_output"},
    ) from last_error


def _validate_selected_tools(selected: list[str], tools: ToolRegistry) -> list[str]:
    cleaned = [name.strip() for name in selected if name and name.strip()]
    if not cleaned:
        raise AgentError(
            "Router did not select any tools",
            details={"reason": "empty_tool_selection"},
        )

    unknown = [name for name in cleaned if name not in tools]
    if unknown:
        raise AgentError(
            "Router selected unknown tools",
            details={"reason": "unknown_tool", "tool_names": unknown},
        )

    available = set(tools.names())
    validated = [name for name in cleaned if name in available]
    if not validated:
        raise AgentError(
            "Router selected unavailable tools",
            details={"reason": "tool_unavailable", "tool_names": cleaned},
        )
    return list(dict.fromkeys(validated))
