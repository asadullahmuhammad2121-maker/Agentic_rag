"""Unit tests for Tavily web search tool and client."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from app.core.exceptions import ProviderError, QueryError
from app.services.agent.models import TavilySearchOutput, ToolError
from app.services.agent.tools.converters import tool_result_to_observation
from app.services.agent.tools.registry import ToolRegistry
from app.services.agent.tools.tavily import TAVILY_WEB_SEARCH_TOOL_NAME, TavilyWebSearchTool
from app.services.agent.tools.tavily_client import TavilySearchClient
from tests.conftest import make_settings


class _FakeTavilyClient:
    def __init__(self, response: dict[str, Any] | None = None, *, error: Exception | None = None):
        self._response = response or {}
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def search(
        self,
        query: str,
        *,
        max_results: int,
        search_depth: str,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
            }
        )
        if self._error is not None:
            raise self._error
        return self._response


def _sample_response() -> dict[str, Any]:
    return {
        "query": "latest AI news",
        "results": [
            {
                "title": "AI Breakthrough",
                "url": "https://example.com/ai",
                "content": "Major AI announcement today.",
                "score": 0.91,
            }
        ],
    }


@pytest.fixture
def tavily_settings() -> Any:
    return make_settings(
        tavily_enabled=True,
        tavily_api_key="test-tavily-key",
        tavily_max_results=5,
    )


def test_tavily_tool_registration_and_lookup(tavily_settings: Any) -> None:
    tool = TavilyWebSearchTool(tavily_settings, client=_FakeTavilyClient(_sample_response()))
    registry = ToolRegistry([tool])

    assert TAVILY_WEB_SEARCH_TOOL_NAME in registry
    assert registry.get(TAVILY_WEB_SEARCH_TOOL_NAME) is tool
    assert registry.list_tools()[0].name == TAVILY_WEB_SEARCH_TOOL_NAME


def test_tavily_tool_valid_search(tavily_settings: Any) -> None:
    client = _FakeTavilyClient(_sample_response())
    tool = TavilyWebSearchTool(tavily_settings, client=client)

    result = tool.run({"query": "latest AI news"})

    assert result.success is True
    assert isinstance(result.output, TavilySearchOutput)
    assert result.output.result_count == 1
    assert result.output.results[0].url == "https://example.com/ai"
    assert result.output.results[0].title == "AI Breakthrough"
    assert client.calls[0]["max_results"] == 5


def test_tavily_tool_input_validation(tavily_settings: Any) -> None:
    tool = TavilyWebSearchTool(tavily_settings, client=_FakeTavilyClient())
    with pytest.raises(QueryError) as exc_info:
        tool.run({"query": "   "})
    assert exc_info.value.details.get("reason") == "invalid_tool_input"


def test_tavily_tool_api_failure_returns_structured_error(tavily_settings: Any) -> None:
    tool = TavilyWebSearchTool(
        tavily_settings,
        client=_FakeTavilyClient(error=RuntimeError("api down")),
    )
    result = tool.run({"query": "latest news"})
    assert result.success is False
    assert result.error is not None
    assert result.error.code == "tool_execution_error"


def test_tavily_client_timeout_failure(tavily_settings: Any) -> None:
    mock_client = MagicMock()
    mock_client.search.side_effect = TimeoutError("timed out")
    client = TavilySearchClient(tavily_settings, client=mock_client)
    with pytest.raises(ProviderError) as exc_info:
        client.search("latest news", max_results=5, search_depth="basic")
    assert exc_info.value.details.get("reason") == "timeout"


def test_tavily_client_network_failure(tavily_settings: Any) -> None:
    mock_client = MagicMock()
    mock_client.search.side_effect = ConnectionError("network unreachable")
    client = TavilySearchClient(tavily_settings, client=mock_client)
    with pytest.raises(ProviderError) as exc_info:
        client.search("latest news", max_results=5, search_depth="basic")
    assert exc_info.value.details.get("reason") == "connection_error"


def test_tavily_client_api_failure(tavily_settings: Any) -> None:
    mock_client = MagicMock()
    mock_client.search.side_effect = RuntimeError("api down")
    client = TavilySearchClient(tavily_settings, client=mock_client)
    with pytest.raises(ProviderError) as exc_info:
        client.search("latest news", max_results=5, search_depth="basic")
    assert exc_info.value.details.get("reason") == "api_error"


def test_tavily_tool_empty_results(tavily_settings: Any) -> None:
    tool = TavilyWebSearchTool(tavily_settings, client=_FakeTavilyClient({"results": []}))
    result = tool.run({"query": "obscure topic xyz"})
    assert result.success is True
    assert isinstance(result.output, TavilySearchOutput)
    assert result.output.empty is True


def test_tavily_tool_disabled_returns_structured_error() -> None:
    settings = make_settings(tavily_enabled=False, tavily_api_key="")
    tool = TavilyWebSearchTool(settings, client=_FakeTavilyClient(_sample_response()))
    result = tool.execute(tool.input_model.model_validate({"query": "latest news"}))
    assert result.success is False
    assert isinstance(result.error, ToolError)
    assert result.error.code == "tavily_disabled"


def test_tavily_client_missing_api_key_raises_provider_error() -> None:
    settings = make_settings(tavily_enabled=True, tavily_api_key="")
    with pytest.raises(ProviderError) as exc_info:
        TavilySearchClient(settings, client=MagicMock())
    assert exc_info.value.details.get("reason") == "missing_api_key"


def test_tavily_observation_creation(tavily_settings: Any) -> None:
    tool = TavilyWebSearchTool(tavily_settings, client=_FakeTavilyClient(_sample_response()))
    result = tool.run({"query": "latest AI news"})
    observation = tool_result_to_observation(tool.name, result)
    assert observation.success is True
    assert observation.metadata["result_count"] == 1
    assert observation.tool_output is not None
