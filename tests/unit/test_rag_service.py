"""Unit tests for RAG orchestration."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core.exceptions import ProviderError, QdrantConnectionError
from app.services.rag.service import EMPTY_RETRIEVAL_ANSWER, RAGService
from app.services.retrieval.service import RetrievedChunk


@pytest.fixture
def retrieval() -> MagicMock:
    return MagicMock()


@pytest.fixture
def llm() -> MagicMock:
    service = MagicMock()
    service.generate.return_value = "Cats are mammals. [S1]"
    return service


@pytest.fixture
def rag(retrieval: MagicMock, llm: MagicMock) -> RAGService:
    return RAGService(retrieval_service=retrieval, llm_service=llm)


def test_successful_rag_flow(rag: RAGService, retrieval: MagicMock, llm: MagicMock) -> None:
    retrieval.retrieve.return_value = [
        RetrievedChunk(
            chunk_id="c1",
            text="Cats are mammals.",
            document_id="doc-1",
            filename="animals.pdf",
            file_type="pdf",
            source="animals.pdf",
            page_number=1,
            section=None,
            chunk_index=0,
            chunking_strategy="fixed",
            score=0.95,
        )
    ]

    result = rag.answer("What are cats?")
    assert "Cats are mammals" in result.answer
    assert len(result.citations) == 1
    citation = result.citations[0]
    assert citation.filename == "animals.pdf"
    assert citation.document_id == "doc-1"
    assert citation.page_number == 1
    assert citation.chunk_id == "c1"
    assert citation.label == "S1"
    llm.generate.assert_called_once()
    assert "system_prompt" in llm.generate.call_args.kwargs


def test_empty_retrieval_skips_llm(rag: RAGService, retrieval: MagicMock, llm: MagicMock) -> None:
    retrieval.retrieve.return_value = []
    result = rag.answer("unknown topic")
    assert result.answer == EMPTY_RETRIEVAL_ANSWER
    assert result.citations == []
    llm.generate.assert_not_called()


def test_groq_failure_propagates(rag: RAGService, retrieval: MagicMock, llm: MagicMock) -> None:
    retrieval.retrieve.return_value = [
        RetrievedChunk(
            chunk_id="c1",
            text="context",
            document_id="d",
            filename="f.pdf",
            file_type="pdf",
            source="f.pdf",
            page_number=1,
            section=None,
            chunk_index=0,
            chunking_strategy="fixed",
            score=0.5,
        )
    ]
    llm.generate.side_effect = ProviderError(
        "boom", provider="groq", details={"reason": "api_error"}
    )
    with pytest.raises(ProviderError) as exc_info:
        rag.answer("q")
    assert exc_info.value.details.get("provider") == "groq"


def test_qdrant_failure_propagates(rag: RAGService, retrieval: MagicMock) -> None:
    retrieval.retrieve.side_effect = QdrantConnectionError()
    with pytest.raises(QdrantConnectionError):
        rag.answer("q")


def test_retrieve_context_returns_chunks_without_generation(
    rag: RAGService,
    retrieval: MagicMock,
    llm: MagicMock,
) -> None:
    retrieval.retrieve.return_value = [
        RetrievedChunk(
            chunk_id="c1",
            text="context",
            document_id="d",
            filename="f.pdf",
            file_type="pdf",
            source="f.pdf",
            page_number=1,
            section=None,
            chunk_index=0,
            chunking_strategy="fixed",
            score=0.5,
        )
    ]

    context = rag.retrieve_context("What is this?")

    assert context.query == "What is this?"
    assert len(context.chunks) == 1
    llm.generate.assert_not_called()


def test_generate_from_chunks_builds_answer_and_citations(
    rag: RAGService,
    llm: MagicMock,
) -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            text="Cats are mammals.",
            document_id="doc-1",
            filename="animals.pdf",
            file_type="pdf",
            source="animals.pdf",
            page_number=1,
            section=None,
            chunk_index=0,
            chunking_strategy="fixed",
            score=0.95,
        )
    ]

    result = rag.generate_from_chunks("What are cats?", chunks)

    assert "Cats are mammals" in result.answer
    assert len(result.citations) == 1
    assert result.citations[0].label == "S1"
    llm.generate.assert_called_once()


def test_generate_from_chunks_empty_context_skips_llm(rag: RAGService, llm: MagicMock) -> None:
    result = rag.generate_from_chunks("unknown", [])
    assert result.answer == EMPTY_RETRIEVAL_ANSWER
    assert result.citations == []
    llm.generate.assert_not_called()
