"""API tests for POST /query."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_rag_service
from app.core.exceptions import ProviderError, QdrantConnectionError
from app.main import create_app
from app.services.rag.service import Citation, RAGResult


@pytest.fixture
def query_client() -> TestClient:
    application = create_app()
    mock_rag = MagicMock()
    mock_rag.answer.return_value = RAGResult(
        answer="Answer text",
        citations=[
            Citation(
                document_id="doc-1",
                filename="a.pdf",
                file_type="pdf",
                source="a.pdf",
                page_number=2,
                section=None,
                chunk_index=0,
                chunk_id="c1",
                score=0.9,
                label="S1",
            )
        ],
    )
    application.dependency_overrides[get_rag_service] = lambda: mock_rag
    client = TestClient(application)
    client.mock_rag = mock_rag  # type: ignore[attr-defined]
    yield client
    application.dependency_overrides.clear()


def test_query_success(query_client: TestClient) -> None:
    response = query_client.post("/query", json={"query": "What is this?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Answer text"
    assert len(body["citations"]) == 1
    assert body["citations"][0]["filename"] == "a.pdf"
    assert body["citations"][0]["document_id"] == "doc-1"
    assert body["citations"][0]["page_number"] == 2
    assert body["citations"][0]["chunk_id"] == "c1"


def test_query_groq_failure_returns_502() -> None:
    application = create_app()
    mock_rag = MagicMock()
    mock_rag.answer.side_effect = ProviderError("fail", provider="groq")
    application.dependency_overrides[get_rag_service] = lambda: mock_rag

    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.post("/query", json={"query": "q"})

    application.dependency_overrides.clear()
    assert response.status_code == 502
    assert response.json()["error"] == "provider_error"


def test_query_qdrant_failure_returns_503() -> None:
    application = create_app()
    mock_rag = MagicMock()
    mock_rag.answer.side_effect = QdrantConnectionError()
    application.dependency_overrides[get_rag_service] = lambda: mock_rag

    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.post("/query", json={"query": "q"})

    application.dependency_overrides.clear()
    assert response.status_code == 503
    assert response.json()["error"] == "qdrant_connection_error"
