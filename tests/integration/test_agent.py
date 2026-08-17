"""API tests for POST /agent/query."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_agent_service, get_rag_service
from app.core.exceptions import ProviderError, QdrantConnectionError, QueryError
from app.main import create_app
from app.services.agent.models import (
    AgentAction,
    AgentActionType,
    AgentCitation,
    AgentObservation,
    AgentRunResult,
    AgentStep,
)
from app.services.rag.service import RAGResult


@pytest.fixture
def agent_client() -> TestClient:
    application = create_app()
    mock_agent = MagicMock()
    mock_agent.run.return_value = AgentRunResult(
        answer="Agent answer",
        citations=[
            AgentCitation(
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
        tool_used="rag_retrieval",
        steps=[
            AgentStep(
                action=AgentAction(
                    type=AgentActionType.CALL_TOOL,
                    tool_name="rag_retrieval",
                    arguments={"query": "What is this?"},
                    reasoning="Using RAG",
                ),
                observation=AgentObservation(
                    tool_name="rag_retrieval",
                    success=True,
                    answer="Agent answer",
                    citations=[],
                    metadata={"citation_count": 1},
                ),
            )
        ],
        metadata={"step_count": 1, "finished": True},
    )
    application.dependency_overrides[get_agent_service] = lambda: mock_agent
    client = TestClient(application)
    client.mock_agent = mock_agent  # type: ignore[attr-defined]
    yield client
    application.dependency_overrides.clear()


def test_agent_query_success(agent_client: TestClient) -> None:
    response = agent_client.post("/agent/query", json={"query": "What is this?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Agent answer"
    assert body["tool_used"] == "rag_retrieval"
    assert len(body["citations"]) == 1
    assert body["citations"][0]["filename"] == "a.pdf"
    assert body["steps"][0]["action"]["tool_name"] == "rag_retrieval"
    assert body["steps"][0]["observation"]["success"] is True
    agent_client.mock_agent.run.assert_called_once()  # type: ignore[attr-defined]


def test_agent_query_empty_body_is_422() -> None:
    application = create_app()
    with TestClient(application) as client:
        response = client.post("/agent/query", json={"query": ""})
    assert response.status_code == 422


def test_agent_query_groq_failure_returns_502() -> None:
    application = create_app()
    mock_agent = MagicMock()
    mock_agent.run.side_effect = ProviderError("fail", provider="groq")
    application.dependency_overrides[get_agent_service] = lambda: mock_agent

    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.post("/agent/query", json={"query": "q"})

    application.dependency_overrides.clear()
    assert response.status_code == 502
    assert response.json()["error"] == "provider_error"


def test_agent_query_qdrant_failure_returns_503() -> None:
    application = create_app()
    mock_agent = MagicMock()
    mock_agent.run.side_effect = QdrantConnectionError()
    application.dependency_overrides[get_agent_service] = lambda: mock_agent

    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.post("/agent/query", json={"query": "q"})

    application.dependency_overrides.clear()
    assert response.status_code == 503
    assert response.json()["error"] == "qdrant_connection_error"


def test_agent_query_empty_query_from_service_returns_400() -> None:
    application = create_app()
    mock_agent = MagicMock()
    mock_agent.run.side_effect = QueryError(
        "Query must not be empty",
        details={"reason": "empty_query"},
    )
    application.dependency_overrides[get_agent_service] = lambda: mock_agent

    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.post("/agent/query", json={"query": "q"})

    application.dependency_overrides.clear()
    assert response.status_code == 400
    assert response.json()["error"] == "query_error"


def test_existing_query_route_is_unchanged() -> None:
    application = create_app()
    mock_rag = MagicMock()
    mock_rag.answer.return_value = RAGResult(answer="Direct RAG", citations=[])
    application.dependency_overrides[get_rag_service] = lambda: mock_rag

    with TestClient(application) as client:
        response = client.post("/query", json={"query": "What is this?"})

    application.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["answer"] == "Direct RAG"
    mock_rag.answer.assert_called_once()
