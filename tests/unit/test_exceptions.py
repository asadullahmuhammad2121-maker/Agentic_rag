"""Unit tests for error handling and safe API responses."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.exceptions import (
    AgentError,
    AppError,
    ConfigurationError,
    ProviderError,
    QdrantConnectionError,
    register_exception_handlers,
)


def _build_error_app() -> FastAPI:
    application = FastAPI()
    register_exception_handlers(application)

    @application.get("/raise-app")
    def raise_app() -> None:
        raise ConfigurationError("bad config")

    @application.get("/raise-qdrant")
    def raise_qdrant() -> None:
        raise QdrantConnectionError()

    @application.get("/raise-provider")
    def raise_provider() -> None:
        raise ProviderError("upstream failed", provider="groq")

    @application.get("/raise-unexpected")
    def raise_unexpected() -> None:
        raise RuntimeError("secret stack trace with key=sk-secret")

    @application.get("/validate")
    def validate(value: int) -> dict[str, int]:
        return {"value": value}

    return application


def test_app_error_response_is_safe() -> None:
    client = TestClient(_build_error_app(), raise_server_exceptions=False)
    response = client.get("/raise-app")
    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "configuration_error"
    assert "bad config" in body["message"]
    assert "traceback" not in body


def test_qdrant_error_returns_503() -> None:
    client = TestClient(_build_error_app(), raise_server_exceptions=False)
    response = client.get("/raise-qdrant")
    assert response.status_code == 503
    assert response.json()["error"] == "qdrant_connection_error"


def test_provider_error_includes_provider_not_secrets() -> None:
    client = TestClient(_build_error_app(), raise_server_exceptions=False)
    response = client.get("/raise-provider")
    assert response.status_code == 502
    body = response.json()
    assert body["details"]["provider"] == "groq"
    assert "sk-" not in response.text


def test_unexpected_error_hides_internals() -> None:
    client = TestClient(_build_error_app(), raise_server_exceptions=False)
    response = client.get("/raise-unexpected")
    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "internal_error"
    assert "sk-secret" not in response.text
    assert "RuntimeError" not in response.text


def test_validation_error_safe_shape() -> None:
    client = TestClient(_build_error_app(), raise_server_exceptions=False)
    response = client.get("/validate", params={"value": "not-int"})
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"
    assert "errors" in body["details"]


def test_app_error_hierarchy() -> None:
    err = ProviderError("x", provider="huggingface")
    assert isinstance(err, AppError)
    assert err.status_code == 502


def test_agent_error_is_safe_app_error() -> None:
    err = AgentError("agent failed", details={"reason": "unknown_tool"})
    assert isinstance(err, AppError)
    assert err.code == "agent_error"
    assert err.status_code == 500
    assert err.details["reason"] == "unknown_tool"
