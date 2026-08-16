"""Regression tests for Basic RAG with advanced features disabled."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core.exceptions import QueryError
from app.services.context_optimization.service import ContextOptimizationService
from app.services.rag.prompt_builder import PromptBuilder
from app.services.rag.service import EMPTY_RETRIEVAL_ANSWER, RAGService
from app.services.retrieval.hybrid import HybridRetrievalService
from app.services.retrieval.multi_query import MultiQueryRetrievalService
from app.services.retrieval.service import RetrievalService, RetrievedChunk
from tests.conftest import make_settings


def _chunk(**overrides: object) -> RetrievedChunk:
    base = {
        "chunk_id": "doc-1:00000",
        "text": "Basic RAG context text.",
        "document_id": "doc-1",
        "filename": "guide.pdf",
        "file_type": "pdf",
        "source": "guide.pdf",
        "page_number": 1,
        "section": None,
        "chunk_index": 0,
        "chunking_strategy": "fixed",
        "score": 0.91,
    }
    base.update(overrides)
    return RetrievedChunk(**base)  # type: ignore[arg-type]


@pytest.fixture
def basic_settings():
    return make_settings(
        chunking_strategy="fixed",
        query_transformation_enabled=False,
        multi_query_enabled=False,
        hybrid_search_enabled=False,
        context_optimization_enabled=False,
        retrieval_top_k=5,
    )


@pytest.fixture
def vector_stack(basic_settings, mock_vector_store: MagicMock) -> RetrievalService:
    embedding = MagicMock()
    embedding.provider_name = "huggingface"
    embedding.embed_query.return_value = [0.1] * basic_settings.embedding_dimension
    mock_vector_store.search.return_value = []
    return RetrievalService(basic_settings, embedding, mock_vector_store)


def test_basic_rag_defaults_in_settings(basic_settings) -> None:
    assert basic_settings.chunking_strategy == "fixed"
    assert basic_settings.query_transformation_enabled is False
    assert basic_settings.multi_query_enabled is False
    assert basic_settings.hybrid_search_enabled is False
    assert basic_settings.context_optimization_enabled is False


def test_basic_rag_end_to_end_with_defaults(basic_settings) -> None:
    retrieval = MagicMock()
    llm = MagicMock()
    llm.generate.return_value = "Answer grounded in context. [S1]"
    retrieval.retrieve.return_value = [_chunk()]

    optimizer = ContextOptimizationService(basic_settings)
    rag = RAGService(
        retrieval_service=retrieval,
        llm_service=llm,
        context_optimizer=optimizer,
    )

    result = rag.answer("What is Basic RAG?")

    retrieval.retrieve.assert_called_once_with(
        "What is Basic RAG?",
        top_k=None,
        filters=None,
    )
    assert "Answer grounded" in result.answer
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "doc-1:00000"
    assert result.citations[0].label == "S1"


def test_basic_rag_empty_retrieval_skips_generation(basic_settings) -> None:
    retrieval = MagicMock()
    retrieval.retrieve.return_value = []
    llm = MagicMock()
    rag = RAGService(
        retrieval_service=retrieval,
        llm_service=llm,
        context_optimizer=ContextOptimizationService(basic_settings),
    )

    result = rag.answer("missing topic")

    assert result.answer == EMPTY_RETRIEVAL_ANSWER
    assert result.citations == []
    llm.generate.assert_not_called()


def test_basic_rag_rejects_empty_query(basic_settings) -> None:
    rag = RAGService(
        retrieval_service=MagicMock(),
        llm_service=MagicMock(),
        context_optimizer=ContextOptimizationService(basic_settings),
    )
    with pytest.raises(QueryError):
        rag.answer("   ")


def test_disabled_multi_query_delegates_to_inner_retriever(basic_settings) -> None:
    inner = MagicMock()
    inner.retrieve.return_value = [_chunk()]
    llm = MagicMock()
    llm.generate.return_value = "ok"
    service = MultiQueryRetrievalService(basic_settings, inner, llm)

    service.retrieve("question")

    inner.retrieve.assert_called_once()
    llm.generate.assert_not_called()


def test_disabled_hybrid_delegates_to_vector_retriever(
    basic_settings,
    vector_stack: RetrievalService,
) -> None:
    keyword = MagicMock()
    hybrid = HybridRetrievalService(basic_settings, vector_stack, keyword)

    with pytest.raises(QueryError):
        hybrid.retrieve("   ")

    keyword.search.assert_not_called()


def test_disabled_context_optimizer_passthrough(basic_settings) -> None:
    chunks = [_chunk(), _chunk(chunk_id="doc-1:00001", chunk_index=1)]
    optimizer = ContextOptimizationService(basic_settings)

    result = optimizer.optimize(chunks)

    assert result.chunks is chunks
    assert result.removed_count == 0


def test_prompt_builder_unchanged_for_basic_flow() -> None:
    chunks = [_chunk()]
    built = PromptBuilder().build("Question?", chunks)

    assert built.context_chunk_count == 1
    assert "Basic RAG context text." in built.user_prompt
    assert "Question?" in built.user_prompt
    assert "[S1]" in built.user_prompt
