"""API tests for agent run history endpoints."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_agent_run_store, get_agent_service
from app.core.exceptions import ProviderError
from app.main import create_app
from app.services.agent.models import AgentRunResult
from app.services.agent.runs.store import AgentRunStore


@pytest.fixture
def runs_client(tmp_path: Path) -> TestClient:
    application = create_app()
    store = AgentRunStore(tmp_path / "agent_runs.db")
    mock_agent = MagicMock()
    mock_agent.run.return_value = AgentRunResult(
        answer="Stored answer",
        citations=[],
        tool_used="rag_retrieval",
        steps=[],
        metadata={"step_count": 0, "finished": True},
    )
    application.dependency_overrides[get_agent_service] = lambda: mock_agent
    application.dependency_overrides[get_agent_run_store] = lambda: store
    client = TestClient(application)
    client.run_store = store  # type: ignore[attr-defined]
    yield client
    application.dependency_overrides.clear()


def test_agent_query_persists_success_run(runs_client: TestClient) -> None:
    response = runs_client.post("/agent/query", json={"query": "What is this?"})
    assert response.status_code == 200

    listing = runs_client.get("/agent/runs")
    assert listing.status_code == 200
    body = listing.json()
    assert body["total"] == 1
    assert body["runs"][0]["status"] == "success"
    assert body["runs"][0]["query"] == "What is this?"

    run_id = body["runs"][0]["run_id"]
    detail = runs_client.get(f"/agent/runs/{run_id}")
    assert detail.status_code == 200
    assert detail.json()["answer"] == "Stored answer"


def test_agent_query_persists_failure_run(runs_client: TestClient) -> None:
    application = create_app()
    store = AgentRunStore(runs_client.run_store._db_path)  # type: ignore[attr-defined]
    mock_agent = MagicMock()
    mock_agent.run.side_effect = ProviderError("fail", provider="groq")
    application.dependency_overrides[get_agent_service] = lambda: mock_agent
    application.dependency_overrides[get_agent_run_store] = lambda: store

    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.post("/agent/query", json={"query": "broken"})
        assert response.status_code == 502
        listing = client.get("/agent/runs?status=failure")
        assert listing.status_code == 200
        assert listing.json()["total"] == 1
        assert listing.json()["runs"][0]["error_code"] == "provider_error"

    application.dependency_overrides.clear()


def test_get_missing_run_returns_404(runs_client: TestClient) -> None:
    response = runs_client.get("/agent/runs/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"] == "agent_run_not_found"
