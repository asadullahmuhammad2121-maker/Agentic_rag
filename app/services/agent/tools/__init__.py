"""Agent tools."""

from app.services.agent.tools.base import Tool
from app.services.agent.tools.converters import (
    chunk_to_output,
    citations_from_rag,
    output_to_chunk,
    tool_result_to_observation,
)
from app.services.agent.tools.rag import RAG_RETRIEVAL_TOOL_NAME, RAGRetrievalTool
from app.services.agent.tools.registry import ToolInfo, ToolRegistry

__all__ = [
    "RAG_RETRIEVAL_TOOL_NAME",
    "RAGRetrievalTool",
    "TAVILY_WEB_SEARCH_TOOL_NAME",
    "TavilySearchClient",
    "TavilyWebSearchTool",
    "Tool",
    "ToolInfo",
    "ToolRegistry",
    "chunk_to_output",
    "citations_from_rag",
    "output_to_chunk",
    "tool_result_to_observation",
]


def __getattr__(name: str) -> object:
    """Lazy-load Tavily modules so the app starts without tavily-python installed."""
    if name == "TAVILY_WEB_SEARCH_TOOL_NAME":
        from app.services.agent.tools.tavily import TAVILY_WEB_SEARCH_TOOL_NAME

        return TAVILY_WEB_SEARCH_TOOL_NAME
    if name == "TavilyWebSearchTool":
        from app.services.agent.tools.tavily import TavilyWebSearchTool

        return TavilyWebSearchTool
    if name == "TavilySearchClient":
        from app.services.agent.tools.tavily_client import TavilySearchClient

        return TavilySearchClient
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
