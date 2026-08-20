"""FastAPI dependency injection wiring."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.services.agent.base import Agent
from app.services.agent.foundation import FoundationAgent
from app.services.agent.generation.web import WebAnswerGenerator
from app.services.agent.planning.planner import QueryPlanner
from app.services.agent.routing.router import QueryRouter
from app.services.agent.runs.store import AgentRunStore
from app.services.agent.service import AgentService
from app.services.agent.tools.base import Tool
from app.services.agent.tools.calculator import CalculatorTool
from app.services.agent.tools.document_navigation import DocumentNavigationTool
from app.services.agent.tools.rag import RAGRetrievalTool
from app.services.agent.tools.registry import ToolRegistry
from app.services.context_optimization.service import ContextOptimizationService
from app.services.embeddings.base import EmbeddingService
from app.services.embeddings.huggingface import HuggingFaceEmbeddingService
from app.services.ingestion.service import DocumentIngestionService
from app.services.llm.base import LLMService
from app.services.llm.groq import GroqLLMService
from app.services.query_transformation.service import QueryTransformationService
from app.services.rag.service import RAGService
from app.services.retrieval.document_navigation import DocumentNavigationService
from app.services.retrieval.explorer import RetrievalExplorerService
from app.services.retrieval.hybrid import HybridRetrievalService
from app.services.retrieval.keyword.bm25 import BM25KeywordSearch
from app.services.retrieval.multi_query import MultiQueryGenerator, MultiQueryRetrievalService
from app.services.retrieval.service import RetrievalService
from app.vector_store.base import VectorStore
from app.vector_store.qdrant import QdrantVectorStore


@lru_cache
def get_llm_service() -> LLMService:
    """Provide the configured LLM service (Groq)."""
    return GroqLLMService(get_settings())


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Provide the configured embedding service (Hugging Face)."""
    return HuggingFaceEmbeddingService(get_settings())


@lru_cache
def get_vector_store() -> VectorStore:
    """Provide the configured vector store (Qdrant)."""
    return QdrantVectorStore(get_settings())


@lru_cache
def get_keyword_search() -> BM25KeywordSearch:
    """Provide the BM25 keyword index."""
    settings = get_settings()
    return BM25KeywordSearch(
        settings.keyword_index_path,
        lock_timeout_seconds=settings.keyword_index_lock_timeout_seconds,
    )


def get_ingestion_service(
    settings: Annotated[Settings, Depends(get_settings)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
    keyword_search: Annotated[BM25KeywordSearch, Depends(get_keyword_search)],
) -> DocumentIngestionService:
    """Provide the document ingestion service."""
    return DocumentIngestionService(
        settings=settings,
        vector_store=vector_store,
        embedding_service=embedding_service,
        keyword_search=keyword_search,
    )


def get_retrieval_service(
    settings: Annotated[Settings, Depends(get_settings)],
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
) -> RetrievalService:
    """Provide the vector retrieval service."""
    return RetrievalService(
        settings=settings,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )


def get_hybrid_retrieval_service(
    settings: Annotated[Settings, Depends(get_settings)],
    retrieval_service: Annotated[RetrievalService, Depends(get_retrieval_service)],
    keyword_search: Annotated[BM25KeywordSearch, Depends(get_keyword_search)],
) -> HybridRetrievalService:
    """Provide hybrid vector + keyword retrieval."""
    return HybridRetrievalService(
        settings=settings,
        vector_retrieval=retrieval_service,
        keyword_search=keyword_search,
    )


def get_query_transformation_service(
    settings: Annotated[Settings, Depends(get_settings)],
    llm_service: Annotated[LLMService, Depends(get_llm_service)],
) -> QueryTransformationService:
    """Provide the query transformation service."""
    return QueryTransformationService(settings, llm_service)


def get_multi_query_retrieval_service(
    settings: Annotated[Settings, Depends(get_settings)],
    retrieval_service: Annotated[RetrievalService, Depends(get_retrieval_service)],
    hybrid_retrieval: Annotated[HybridRetrievalService, Depends(get_hybrid_retrieval_service)],
    llm_service: Annotated[LLMService, Depends(get_llm_service)],
) -> MultiQueryRetrievalService:
    """Provide multi-query retrieval wrapper."""
    backend = hybrid_retrieval if settings.hybrid_search_enabled else retrieval_service
    return MultiQueryRetrievalService(settings, backend, llm_service)


def get_context_optimizer(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ContextOptimizationService:
    """Provide context optimization for retrieved chunks."""
    return ContextOptimizationService(settings)


def get_rag_service(
    settings: Annotated[Settings, Depends(get_settings)],
    llm_service: Annotated[LLMService, Depends(get_llm_service)],
    query_transformer: Annotated[
        QueryTransformationService,
        Depends(get_query_transformation_service),
    ],
    multi_query_retrieval: Annotated[
        MultiQueryRetrievalService,
        Depends(get_multi_query_retrieval_service),
    ],
    context_optimizer: Annotated[
        ContextOptimizationService,
        Depends(get_context_optimizer),
    ],
) -> RAGService:
    """Provide the RAG orchestration service."""
    transformer = query_transformer if settings.query_transformation_enabled else None
    return RAGService(
        retrieval_service=multi_query_retrieval,
        llm_service=llm_service,
        query_transformer=transformer,
        context_optimizer=context_optimizer,
    )


def get_retrieval_explorer_service(
    settings: Annotated[Settings, Depends(get_settings)],
    retrieval_service: Annotated[RetrievalService, Depends(get_retrieval_service)],
    hybrid_retrieval: Annotated[HybridRetrievalService, Depends(get_hybrid_retrieval_service)],
    keyword_search: Annotated[BM25KeywordSearch, Depends(get_keyword_search)],
    multi_query_retrieval: Annotated[
        MultiQueryRetrievalService,
        Depends(get_multi_query_retrieval_service),
    ],
    query_transformer: Annotated[
        QueryTransformationService,
        Depends(get_query_transformation_service),
    ],
    llm_service: Annotated[LLMService, Depends(get_llm_service)],
    context_optimizer: Annotated[
        ContextOptimizationService,
        Depends(get_context_optimizer),
    ],
) -> RetrievalExplorerService:
    """Provide the retrieval explorer service."""
    transformer = query_transformer if settings.query_transformation_enabled else None
    generator = (
        MultiQueryGenerator(settings, llm_service)
        if settings.multi_query_enabled
        else None
    )
    optimizer = context_optimizer if settings.context_optimization_enabled else None
    return RetrievalExplorerService(
        settings=settings,
        vector_retrieval=retrieval_service,
        hybrid_retrieval=hybrid_retrieval,
        keyword_search=keyword_search,
        multi_query_retrieval=multi_query_retrieval,
        query_transformer=transformer,
        multi_query_generator=generator,
        context_optimizer=optimizer,
    )


def get_rag_retrieval_tool(
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
) -> RAGRetrievalTool:
    """Provide the internal RAG retrieval tool."""
    return RAGRetrievalTool(rag_service)


def get_tavily_web_search_tool(
    settings: Annotated[Settings, Depends(get_settings)],
) -> Tool | None:
    """Provide the Tavily web search tool when enabled and configured."""
    if not settings.tavily_configured:
        return None
    from app.services.agent.tools.tavily import TavilyWebSearchTool

    return TavilyWebSearchTool(settings)


def get_calculator_tool(
    settings: Annotated[Settings, Depends(get_settings)],
) -> Tool | None:
    """Provide the calculator tool when enabled."""
    if not settings.calculator_enabled:
        return None
    return CalculatorTool(settings)


def get_document_navigation_service(
    settings: Annotated[Settings, Depends(get_settings)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
) -> DocumentNavigationService:
    """Provide metadata-based document navigation."""
    return DocumentNavigationService(settings, vector_store)


def get_document_navigation_tool(
    navigation_service: Annotated[
        DocumentNavigationService,
        Depends(get_document_navigation_service),
    ],
) -> DocumentNavigationTool:
    """Provide the document navigation tool."""
    return DocumentNavigationTool(navigation_service)


def get_tool_registry(
    rag_tool: Annotated[RAGRetrievalTool, Depends(get_rag_retrieval_tool)],
    navigation_tool: Annotated[DocumentNavigationTool, Depends(get_document_navigation_tool)],
    tavily_tool: Annotated[Tool | None, Depends(get_tavily_web_search_tool)],
    calculator_tool: Annotated[Tool | None, Depends(get_calculator_tool)],
) -> ToolRegistry:
    """Provide the agent tool registry."""
    tools: list[Tool] = [rag_tool, navigation_tool]
    if tavily_tool is not None:
        tools.append(tavily_tool)
    if calculator_tool is not None:
        tools.append(calculator_tool)
    return ToolRegistry(tools)


def get_web_answer_generator(
    settings: Annotated[Settings, Depends(get_settings)],
    llm_service: Annotated[LLMService, Depends(get_llm_service)],
) -> WebAnswerGenerator:
    """Provide web answer generation for Tavily search results."""
    return WebAnswerGenerator(llm_service, settings)


def get_query_router(
    settings: Annotated[Settings, Depends(get_settings)],
    llm_service: Annotated[LLMService, Depends(get_llm_service)],
) -> QueryRouter:
    """Provide the LLM query router for agent tool selection."""
    return QueryRouter(settings, llm_service)


def get_query_planner(
    settings: Annotated[Settings, Depends(get_settings)],
    llm_service: Annotated[LLMService, Depends(get_llm_service)],
) -> QueryPlanner:
    """Provide the LLM query planner for hybrid decomposition."""
    return QueryPlanner(settings, llm_service)


def get_agent(
    settings: Annotated[Settings, Depends(get_settings)],
    router: Annotated[QueryRouter, Depends(get_query_router)],
    planner: Annotated[QueryPlanner, Depends(get_query_planner)],
) -> Agent:
    """Provide the foundation agent."""
    return FoundationAgent(router, planner, settings)


def get_agent_run_store(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AgentRunStore:
    """Provide SQLite-backed agent run history storage."""
    return AgentRunStore(settings.agent_runs_db_path)


def get_agent_service(
    settings: Annotated[Settings, Depends(get_settings)],
    agent: Annotated[Agent, Depends(get_agent)],
    tools: Annotated[ToolRegistry, Depends(get_tool_registry)],
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
    web_answer_generator: Annotated[WebAnswerGenerator, Depends(get_web_answer_generator)],
) -> AgentService:
    """Provide the agent orchestrator."""
    return AgentService(
        agent=agent,
        tools=tools,
        rag_service=rag_service,
        web_answer_generator=web_answer_generator,
        max_steps=settings.agent_max_steps,
    )


def clear_dependency_caches() -> None:
    """Clear cached dependencies (for tests)."""
    get_llm_service.cache_clear()
    get_embedding_service.cache_clear()
    get_vector_store.cache_clear()
    get_keyword_search.cache_clear()


SettingsDep = Annotated[Settings, Depends(get_settings)]
LLMServiceDep = Annotated[LLMService, Depends(get_llm_service)]
EmbeddingServiceDep = Annotated[EmbeddingService, Depends(get_embedding_service)]
VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]
KeywordSearchDep = Annotated[BM25KeywordSearch, Depends(get_keyword_search)]
IngestionServiceDep = Annotated[DocumentIngestionService, Depends(get_ingestion_service)]
RetrievalServiceDep = Annotated[RetrievalService, Depends(get_retrieval_service)]
RetrievalExplorerServiceDep = Annotated[
    RetrievalExplorerService,
    Depends(get_retrieval_explorer_service),
]
RAGServiceDep = Annotated[RAGService, Depends(get_rag_service)]
AgentServiceDep = Annotated[AgentService, Depends(get_agent_service)]
AgentRunStoreDep = Annotated[AgentRunStore, Depends(get_agent_run_store)]
ToolRegistryDep = Annotated[ToolRegistry, Depends(get_tool_registry)]
