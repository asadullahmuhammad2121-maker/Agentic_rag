"""Unit tests for retrieval service."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core.exceptions import ProviderError, QdrantConnectionError, QueryError
from app.services.retrieval.service import RetrievalService
from app.vector_store.base import SearchResult
from tests.conftest import make_settings


@pytest.fixture
def embedding_service() -> MagicMock:
    service = MagicMock()
    service.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
    return service


@pytest.fixture
def vector_store() -> MagicMock:
    return MagicMock()


@pytest.fixture
def retrieval(embedding_service: MagicMock, vector_store: MagicMock) -> RetrievalService:
    return RetrievalService(
        make_settings(retrieval_top_k=5, embedding_dimension=4),
        embedding_service,
        vector_store,
    )


def test_retrieve_maps_hits(
    retrieval: RetrievalService,
    embedding_service: MagicMock,
    vector_store: MagicMock,
) -> None:
    vector_store.search.return_value = [
        SearchResult(
            id="p1",
            score=0.92,
            payload={
                "text": "chunk text",
                "document_id": "doc-1",
                "filename": "a.pdf",
                "page_number": 2,
                "chunk_index": 1,
            },
        )
    ]

    chunks = retrieval.retrieve("what is this?")
    assert len(chunks) == 1
    assert chunks[0].text == "chunk text"
    assert chunks[0].document_id == "doc-1"
    assert chunks[0].filename == "a.pdf"
    assert chunks[0].page_number == 2
    assert chunks[0].chunk_index == 1
    assert chunks[0].score == pytest.approx(0.92)
    embedding_service.embed_query.assert_called_once_with("what is this?")
    vector_store.search.assert_called_once()


def test_retrieve_top_k_override(
    retrieval: RetrievalService,
    vector_store: MagicMock,
) -> None:
    vector_store.search.return_value = []
    retrieval.retrieve("q", top_k=3)
    assert vector_store.search.call_args.kwargs["limit"] == 3


def test_retrieve_passes_metadata_filters(
    retrieval: RetrievalService,
    vector_store: MagicMock,
) -> None:
    vector_store.search.return_value = []
    retrieval.retrieve("q", filters={"document_id": "doc-1"})
    assert vector_store.search.call_args.kwargs["filters"] == {"document_id": "doc-1"}


def test_retrieve_empty_results(
    retrieval: RetrievalService,
    vector_store: MagicMock,
) -> None:
    vector_store.search.return_value = []
    assert retrieval.retrieve("missing topic") == []


def test_retrieve_rejects_empty_query(retrieval: RetrievalService) -> None:
    with pytest.raises(QueryError):
        retrieval.retrieve("   ")


def test_retrieve_qdrant_failure(
    retrieval: RetrievalService,
    vector_store: MagicMock,
) -> None:
    vector_store.search.side_effect = QdrantConnectionError()
    with pytest.raises(QdrantConnectionError):
        retrieval.retrieve("q")


def test_retrieve_embedding_failure(
    retrieval: RetrievalService,
    embedding_service: MagicMock,
) -> None:
    embedding_service.embed_query.side_effect = ProviderError(
        "embed failed",
        provider="huggingface",
    )
    with pytest.raises(ProviderError):
        retrieval.retrieve("q")
