"""FastAPI dependency injection wiring."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.services.context_optimization.service import ContextOptimizationService
from app.services.embeddings.base import EmbeddingService
from app.services.embeddings.huggingface import HuggingFaceEmbeddingService
from app.services.ingestion.service import DocumentIngestionService
from app.services.llm.base import LLMService
from app.services.llm.groq import GroqLLMService
from app.services.query_transformation.service import QueryTransformationService
from app.services.rag.service import RAGService
from app.services.retrieval.hybrid import HybridRetrievalService
from app.services.retrieval.keyword.bm25 import BM25KeywordSearch
from app.services.retrieval.multi_query import MultiQueryRetrievalService
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
    return BM25KeywordSearch(get_settings().keyword_index_path)


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
IngestionServiceDep = Annotated[DocumentIngestionService, Depends(get_ingestion_service)]
RetrievalServiceDep = Annotated[RetrievalService, Depends(get_retrieval_service)]
RAGServiceDep = Annotated[RAGService, Depends(get_rag_service)]
