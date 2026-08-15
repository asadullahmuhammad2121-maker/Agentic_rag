"""Integration tests for the FastAPI health endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.api.deps import get_vector_store
from app.main import create_app


def test_health_endpoint_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app"] == "rag-foundation"
    assert body["version"]
    assert body["environment"] == "test"
    assert any(c["name"] == "qdrant" and c["status"] == "ok" for c in body["components"])


def test_health_endpoint_degraded_when_qdrant_down() -> None:
    application = create_app()
    unhealthy = MagicMock()
    unhealthy.health_check.return_value = False
    application.dependency_overrides[get_vector_store] = lambda: unhealthy

    with TestClient(application) as test_client:
        response = test_client.get("/health")

    application.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    qdrant = next(c for c in body["components"] if c["name"] == "qdrant")
    assert qdrant["status"] == "unavailable"
