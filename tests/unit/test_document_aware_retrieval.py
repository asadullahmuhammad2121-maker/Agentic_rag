"""Unit tests for document-aware retrieval filtering."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import QueryError
from app.main import create_app
from app.services.retrieval.filters import RetrievalFilters
from app.services.retrieval.service import RetrievalService
from app.vector_store.filters import PayloadFilter
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


def test_retrieve_without_filters_passes_none(
    retrieval: RetrievalService,
    vector_store: MagicMock,
) -> None:
    vector_store.search.return_value = []
    retrieval.retrieve("question")
    assert vector_store.search.call_args.kwargs["filters"] is None


def test_retrieve_document_id_filter(
    retrieval: RetrievalService,
    vector_store: MagicMock,
) -> None:
    vector_store.search.return_value = []
    filters = RetrievalFilters.from_query(document_ids=["doc-1"])
    retrieval.retrieve("question", filters=filters)
    payload_filter = vector_store.search.call_args.kwargs["filters"]
    assert payload_filter == PayloadFilter(exact={"document_id": "doc-1"})


def test_retrieve_multiple_document_ids_filter(
    retrieval: RetrievalService,
    vector_store: MagicMock,
) -> None:
    vector_store.search.return_value = []
    filters = RetrievalFilters.from_query(document_ids=["doc-1", "doc-2"])
    retrieval.retrieve("question", filters=filters)
    payload_filter = vector_store.search.call_args.kwargs["filters"]
    assert payload_filter == PayloadFilter(any_of={"document_id": ("doc-1", "doc-2")})


def test_retrieve_file_type_filter(
    retrieval: RetrievalService,
    vector_store: MagicMock,
) -> None:
    vector_store.search.return_value = []
    filters = RetrievalFilters.from_query(file_types=["pdf"])
    retrieval.retrieve("question", filters=filters)
    payload_filter = vector_store.search.call_args.kwargs["filters"]
    assert payload_filter == PayloadFilter(exact={"file_type": "pdf"})


def test_retrieve_section_filter(
    retrieval: RetrievalService,
    vector_store: MagicMock,
) -> None:
    vector_store.search.return_value = []
    filters = RetrievalFilters.from_query(sections=["Introduction"])
    retrieval.retrieve("question", filters=filters)
    payload_filter = vector_store.search.call_args.kwargs["filters"]
    assert payload_filter == PayloadFilter(exact={"section": "Introduction"})


def test_retrieve_combined_filters(
    retrieval: RetrievalService,
    vector_store: MagicMock,
) -> None:
    vector_store.search.return_value = []
    filters = RetrievalFilters.from_query(
        document_ids=["doc-1"],
        file_types=["pdf"],
        sections=["Introduction"],
    )
    retrieval.retrieve("question", filters=filters)
    payload_filter = vector_store.search.call_args.kwargs["filters"]
    assert payload_filter == PayloadFilter(
        exact={"document_id": "doc-1", "file_type": "pdf", "section": "Introduction"},
    )


def test_retrieve_maps_citation_metadata(
    retrieval: RetrievalService,
    vector_store: MagicMock,
) -> None:
    vector_store.search.return_value = [
        MagicMock(
            id="point-1",
            score=0.88,
            payload={
                "text": "RAG overview",
                "document_id": "doc-a",
                "filename": "guide.pdf",
                "file_type": "pdf",
                "source": "guide.pdf",
                "page_number": 2,
                "section": "Introduction",
                "chunk_index": 1,
                "chunk_id": "doc-a:00001",
                "chunking_strategy": "fixed",
            },
        )
    ]
    chunks = retrieval.retrieve("What is RAG?")
    chunk = chunks[0]
    assert chunk.document_id == "doc-a"
    assert chunk.page_number == 2
    assert chunk.section == "Introduction"
    assert chunk.chunk_id == "doc-a:00001"
    assert chunk.chunking_strategy == "fixed"


def test_retrieve_multi_document_results(
    retrieval: RetrievalService,
    vector_store: MagicMock,
) -> None:
    vector_store.search.return_value = [
        MagicMock(
            id="p1",
            score=0.9,
            payload={
                "text": "From A",
                "document_id": "doc-a",
                "filename": "a.txt",
                "file_type": "txt",
                "source": "a.txt",
                "page_number": 1,
                "section": None,
                "chunk_index": 0,
                "chunk_id": "doc-a:00000",
                "chunking_strategy": "fixed",
            },
        ),
        MagicMock(
            id="p2",
            score=0.85,
            payload={
                "text": "From B",
                "document_id": "doc-b",
                "filename": "b.md",
                "file_type": "markdown",
                "source": "b.md",
                "page_number": 1,
                "section": "Overview",
                "chunk_index": 0,
                "chunk_id": "doc-b:00000",
                "chunking_strategy": "fixed",
            },
        ),
    ]
    chunks = retrieval.retrieve("cross document")
    assert {chunk.document_id for chunk in chunks} == {"doc-a", "doc-b"}


def test_invalid_filter_values_raise_query_error() -> None:
    with pytest.raises(QueryError):
        RetrievalFilters.from_query(document_ids=["  "])


def test_query_api_rejects_empty_filter_list() -> None:
    application = create_app()
    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.post(
            "/query",
            json={"query": "What is RAG?", "document_ids": [""]},
        )
    assert response.status_code == 422
