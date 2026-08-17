"""API tests for GET /settings."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app


def test_public_settings_excludes_secrets() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/settings")

    assert response.status_code == 200
    body = response.json()
    assert body["read_only"] is True
    assert "groq_api_key" not in body
    assert "tavily_api_key" not in body
    assert "huggingface_api_key" not in body
    assert body["general"]["app_name"]
    assert body["rag"]["chunking_strategy"]
    assert body["rag"]["reranking_enabled"] is False
    assert isinstance(body["agent"]["tools"], list)
    assert len(body["agent"]["tools"]) >= 1
    rag_tool = next(tool for tool in body["agent"]["tools"] if tool["name"] == "rag_retrieval")
    assert rag_tool["available"] is True
    assert "configured" in rag_tool
