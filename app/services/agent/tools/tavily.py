"""Tavily web search tool — wraps the Tavily API via the generic Tool interface."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.agent.models import (
    TavilySearchInput,
    TavilySearchOutput,
    ToolError,
    ToolResult,
    WebSearchResultItem,
)
from app.services.agent.tools.base import Tool
from app.services.agent.tools.tavily_client import TavilyClientProtocol, TavilySearchClient

logger = get_logger(__name__)

TAVILY_WEB_SEARCH_TOOL_NAME = "tavily_web_search"


class TavilyWebSearchTool(Tool):
    """Search the public web using Tavily for current or external information."""

    def __init__(
        self,
        settings: Settings,
        client: TavilyClientProtocol | None = None,
    ) -> None:
        self._settings = settings
        self._client = client or TavilySearchClient(settings)

    @property
    def name(self) -> str:
        return TAVILY_WEB_SEARCH_TOOL_NAME

    @property
    def description(self) -> str:
        return (
            "Search the public web for current or external information. "
            "Use this when the answer requires up-to-date web content."
        )

    @property
    def input_model(self) -> type[BaseModel]:
        return TavilySearchInput

    @property
    def output_model(self) -> type[BaseModel]:
        return TavilySearchOutput

    def execute(self, validated_input: BaseModel) -> ToolResult:
        if not self._settings.tavily_configured:
            return ToolResult(
                success=False,
                error=ToolError(
                    code="tavily_disabled",
                    message="Tavily web search is disabled or not configured",
                    details={"reason": "tavily_disabled"},
                ),
            )

        payload = TavilySearchInput.model_validate(validated_input.model_dump())
        max_results = payload.max_results or self._settings.tavily_max_results

        logger.info(
            "tavily_web_search_tool_started",
            extra={
                "operation": "tavily_web_search_tool",
                "query_length": len(payload.query),
                "max_results": max_results,
            },
        )

        response = self._client.search(
            payload.query,
            max_results=max_results,
            search_depth=self._settings.tavily_search_depth,
        )
        results = _parse_results(response)
        output = TavilySearchOutput(query=payload.query, results=results)

        logger.info(
            "tavily_web_search_tool_completed",
            extra={
                "operation": "tavily_web_search_tool",
                "result_count": output.result_count,
                "empty": output.empty,
            },
        )
        return ToolResult(success=True, output=output)


def _parse_results(response: dict[str, Any]) -> list[WebSearchResultItem]:
    raw_results = response.get("results", [])
    if not isinstance(raw_results, list):
        return []

    parsed: list[WebSearchResultItem] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        url = str(item.get("url", "")).strip()
        content = str(item.get("content", "")).strip()
        if not title and not url:
            continue
        score_raw = item.get("score")
        score = float(score_raw) if score_raw is not None else None
        parsed.append(
            WebSearchResultItem(
                title=title or url,
                url=url or title,
                content=content,
                score=score,
            )
        )
    return parsed
