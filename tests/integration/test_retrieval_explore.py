"""API tests for POST /retrieval/explore."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_retrieval_explorer_service
from app.main import create_app
from app.services.retrieval.explorer import PipelineStage, RetrievalExploreResult
from app.services.retrieval.service import RetrievedChunk


def _sample_chunk(*, chunk_id: str = "c1", score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text="Sample chunk text",
        document_id="doc-1",
        filename="sample.pdf",
        file_type="pdf",
        source="sample.pdf",
        page_number=1,
        section="Intro",
        chunk_index=0,
        chunking_strategy="fixed",
        score=score,
    )


@pytest.fixture
def explore_client() -> TestClient:
    application = create_app()
    mock_explorer = MagicMock()
    mock_explorer.explore.return_value = RetrievalExploreResult(
        query="What is RAG?",
        retrieval_query="What is RAG?",
        generated_queries=None,
        configuration={
            "query_transformation_enabled": False,
            "multi_query_enabled": False,
            "hybrid_search_enabled": False,
            "context_optimization_enabled": False,
            "reranking_enabled": False,
        },
        pipeline=[
            PipelineStage(
                id="query",
                label="Query",
                enabled=True,
                executed=True,
            ),
            PipelineStage(
                id="vector_search",
                label="Vector Search",
                enabled=True,
                executed=True,
                result_count=1,
            ),
            PipelineStage(
                id="reranking",
                label="Reranking",
                enabled=False,
                executed=False,
            ),
            PipelineStage(
                id="final_results",
                label="Final Results",
                enabled=True,
                executed=True,
                result_count=1,
            ),
        ],
        vector_results=[_sample_chunk()],
        bm25_results=[],
        fused_results=None,
        results=[_sample_chunk()],
        result_methods={"c1": "vector"},
        metadata={},
    )
    application.dependency_overrides[get_retrieval_explorer_service] = lambda: mock_explorer
    client = TestClient(application)
    client.mock_explorer = mock_explorer  # type: ignore[attr-defined]
    yield client
    application.dependency_overrides.clear()


def test_retrieval_explore_success(explore_client: TestClient) -> None:
    response = explore_client.post("/retrieval/explore", json={"query": "What is RAG?"})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "What is RAG?"
    assert body["configuration"]["reranking_enabled"] is False
    assert len(body["results"]) == 1
    assert body["results"][0]["filename"] == "sample.pdf"
    assert body["results"][0]["text"] == "Sample chunk text"
    assert body["results"][0]["retrieval_method"] == "vector"
    assert body["results"][0]["chunk_id"] == "c1"
    explore_client.mock_explorer.explore.assert_called_once()  # type: ignore[attr-defined]


def test_retrieval_explore_empty_query_returns_422(explore_client: TestClient) -> None:
    response = explore_client.post("/retrieval/explore", json={"query": ""})
    assert response.status_code == 422
