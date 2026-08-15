"""FastAPI dependency injection wiring."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.services.embeddings.base import EmbeddingService
from app.services.embeddings.huggingface import HuggingFaceEmbeddingService
from app.services.ingestion.service import DocumentIngestionService
from app.services.llm.base import LLMService
from app.services.llm.groq import GroqLLMService
from app.services.rag.service import RAGService
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


def get_ingestion_service(
    settings: Annotated[Settings, Depends(get_settings)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
) -> DocumentIngestionService:
    """Provide the document ingestion service."""
    return DocumentIngestionService(
        settings=settings,
        vector_store=vector_store,
        embedding_service=embedding_service,
    )


def get_retrieval_service(
    settings: Annotated[Settings, Depends(get_settings)],
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
) -> RetrievalService:
    """Provide the retrieval service."""
    return RetrievalService(
        settings=settings,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )


def get_rag_service(
    retrieval_service: Annotated[RetrievalService, Depends(get_retrieval_service)],
    llm_service: Annotated[LLMService, Depends(get_llm_service)],
) -> RAGService:
    """Provide the RAG orchestration service."""
    return RAGService(
        retrieval_service=retrieval_service,
        llm_service=llm_service,
    )


def clear_dependency_caches() -> None:
    """Clear cached dependencies (for tests)."""
    get_llm_service.cache_clear()
    get_embedding_service.cache_clear()
    get_vector_store.cache_clear()


SettingsDep = Annotated[Settings, Depends(get_settings)]
LLMServiceDep = Annotated[LLMService, Depends(get_llm_service)]
EmbeddingServiceDep = Annotated[EmbeddingService, Depends(get_embedding_service)]
VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]
IngestionServiceDep = Annotated[DocumentIngestionService, Depends(get_ingestion_service)]
RetrievalServiceDep = Annotated[RetrievalService, Depends(get_retrieval_service)]
RAGServiceDep = Annotated[RAGService, Depends(get_rag_service)]
