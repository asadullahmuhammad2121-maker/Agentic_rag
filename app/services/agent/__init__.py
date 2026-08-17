"""Agent foundation: decide → tool → observe → generate → finish."""

from app.services.agent.base import Agent
from app.services.agent.foundation import FoundationAgent
from app.services.agent.generation import WebAnswerGenerator, web_results_to_citations
from app.services.agent.models import (
    AgentAction,
    AgentActionType,
    AgentCitation,
    AgentObservation,
    AgentRequest,
    AgentRunResult,
    AgentStep,
    RAGRetrievalArguments,
    RAGRetrievalInput,
    RAGRetrievalOutput,
    RetrievedChunkOutput,
    TavilySearchInput,
    TavilySearchOutput,
    ToolError,
    ToolResult,
    WebSearchResultItem,
)
from app.services.agent.service import AgentService
from app.services.agent.tools import (
    RAG_RETRIEVAL_TOOL_NAME,
    TAVILY_WEB_SEARCH_TOOL_NAME,
    RAGRetrievalTool,
    TavilyWebSearchTool,
    Tool,
    ToolInfo,
    ToolRegistry,
)

__all__ = [
    "RAG_RETRIEVAL_TOOL_NAME",
    "TAVILY_WEB_SEARCH_TOOL_NAME",
    "Agent",
    "AgentAction",
    "AgentActionType",
    "AgentCitation",
    "AgentObservation",
    "AgentRequest",
    "AgentRunResult",
    "AgentService",
    "AgentStep",
    "FoundationAgent",
    "RAGRetrievalArguments",
    "RAGRetrievalInput",
    "RAGRetrievalOutput",
    "RAGRetrievalTool",
    "RetrievedChunkOutput",
    "TavilySearchInput",
    "TavilySearchOutput",
    "TavilyWebSearchTool",
    "Tool",
    "ToolError",
    "ToolInfo",
    "ToolRegistry",
    "ToolResult",
    "WebAnswerGenerator",
    "WebSearchResultItem",
    "web_results_to_citations",
]
