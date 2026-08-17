"""Build user-safe settings payloads from internal configuration."""

from __future__ import annotations

from app.core.config import Settings
from app.schemas.settings import (
    AgentSettingsResponse,
    GeneralSettingsResponse,
    PublicSettingsResponse,
    RAGSettingsResponse,
    SearchSettingsResponse,
    ToolStatusResponse,
)
from app.services.agent.tools.rag import RAG_RETRIEVAL_TOOL_NAME
from app.services.agent.tools.registry import ToolRegistry
from app.services.agent.tools.tavily import TAVILY_WEB_SEARCH_TOOL_NAME

_TOOL_LABELS = {
    RAG_RETRIEVAL_TOOL_NAME: "RAG Retrieval",
    TAVILY_WEB_SEARCH_TOOL_NAME: "Tavily Web Search",
}


def build_public_settings(settings: Settings, tools: ToolRegistry) -> PublicSettingsResponse:
    """Return a sanitized snapshot of runtime configuration."""
    registered = set(tools.names())
    tool_statuses = [
        ToolStatusResponse(
            name=RAG_RETRIEVAL_TOOL_NAME,
            label=_TOOL_LABELS[RAG_RETRIEVAL_TOOL_NAME],
            enabled=True,
            configured=True,
            available=RAG_RETRIEVAL_TOOL_NAME in registered,
        ),
        ToolStatusResponse(
            name=TAVILY_WEB_SEARCH_TOOL_NAME,
            label=_TOOL_LABELS[TAVILY_WEB_SEARCH_TOOL_NAME],
            enabled=settings.tavily_enabled,
            configured=settings.tavily_configured,
            available=TAVILY_WEB_SEARCH_TOOL_NAME in registered,
        ),
    ]

    return PublicSettingsResponse(
        general=GeneralSettingsResponse(
            app_name=settings.app_name,
            app_version=settings.app_version,
            environment=settings.app_env,
            log_level=settings.log_level,
            request_timeout_seconds=settings.request_timeout_seconds,
        ),
        rag=RAGSettingsResponse(
            chunking_strategy=settings.chunking_strategy,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            chunk_min_size=settings.chunk_min_size,
            chunk_max_size=settings.chunk_max_size,
            semantic_similarity_threshold=settings.semantic_similarity_threshold,
            retrieval_top_k=settings.retrieval_top_k,
            retrieval_score_threshold=settings.retrieval_score_threshold,
            hybrid_search_enabled=settings.hybrid_search_enabled,
            hybrid_top_k=settings.hybrid_top_k,
            vector_search_weight=settings.vector_search_weight,
            keyword_search_weight=settings.keyword_search_weight,
            query_transformation_enabled=settings.query_transformation_enabled,
            multi_query_enabled=settings.multi_query_enabled,
            multi_query_count=settings.multi_query_count,
            context_optimization_enabled=settings.context_optimization_enabled,
            context_max_chunks=settings.context_max_chunks,
            context_max_tokens=settings.context_max_tokens,
            context_min_score=settings.context_min_score,
            reranking_enabled=False,
        ),
        agent=AgentSettingsResponse(
            agent_max_steps=settings.agent_max_steps,
            agent_routing_enabled=settings.agent_routing_enabled,
            agent_planning_enabled=settings.agent_planning_enabled,
            groq_model=settings.groq_model,
            groq_configured=bool(settings.groq_api_key.get_secret_value().strip()),
            llm_temperature=settings.llm_temperature,
            llm_max_tokens=settings.llm_max_tokens,
            tools=tool_statuses,
        ),
        search=SearchSettingsResponse(
            bm25_enabled=settings.hybrid_search_enabled,
            web_search_enabled=settings.tavily_enabled,
            web_search_configured=settings.tavily_configured,
            tavily_max_results=settings.tavily_max_results,
            tavily_search_depth=settings.tavily_search_depth,
            embedding_model=settings.huggingface_embedding_model,
            embedding_dimension=settings.embedding_dimension,
            qdrant_collection_name=settings.qdrant_collection_name,
        ),
    )
