"""Integration tests for the FastAPI health endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.api.deps import get_keyword_search, get_vector_store
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
    assert any(c["name"] == "keyword_index" and c["status"] == "ok" for c in body["components"])


def test_liveness_probe_ok(client: TestClient) -> None:
    response = client.get("/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_readiness_probe_ok(client: TestClient) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


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


def test_readiness_probe_unavailable_when_qdrant_down() -> None:
    application = create_app()
    unhealthy = MagicMock()
    unhealthy.health_check.return_value = False
    application.dependency_overrides[get_vector_store] = lambda: unhealthy

    with TestClient(application) as test_client:
        response = test_client.get("/ready")

    application.dependency_overrides.clear()

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unavailable"


def test_readiness_probe_unavailable_when_keyword_index_unhealthy() -> None:
    application = create_app()
    unhealthy_keyword = MagicMock()
    unhealthy_keyword.health_check.return_value = False
    application.dependency_overrides[get_keyword_search] = lambda: unhealthy_keyword

    with TestClient(application) as test_client:
        response = test_client.get("/ready")

    application.dependency_overrides.clear()

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unavailable"
