"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# Ensure deterministic env before Settings / app imports.
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("GROQ_MODEL", "llama-3.3-70b-versatile")
os.environ.setdefault("HUGGINGFACE_API_KEY", "test-hf-key")
os.environ.setdefault(
    "HUGGINGFACE_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_COLLECTION_NAME", "documents_test")
os.environ.setdefault("EMBEDDING_DIMENSION", "384")

from app.api.deps import (  # noqa: E402
    clear_dependency_caches,
    get_embedding_service,
    get_llm_service,
    get_vector_store,
)
from app.core.config import Settings, clear_settings_cache, get_settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.vector_store.base import VectorStore  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_caches() -> Generator[None, None, None]:
    clear_settings_cache()
    clear_dependency_caches()
    yield
    clear_settings_cache()
    clear_dependency_caches()


@pytest.fixture
def settings() -> Settings:
    clear_settings_cache()
    return get_settings()


@pytest.fixture
def mock_vector_store() -> MagicMock:
    store = MagicMock(spec=VectorStore)
    store.health_check.return_value = True
    return store


@pytest.fixture
def client(mock_vector_store: MagicMock) -> Generator[TestClient, None, None]:
    application = create_app()
    application.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    application.dependency_overrides[get_llm_service] = lambda: MagicMock()
    application.dependency_overrides[get_embedding_service] = lambda: MagicMock()
    with TestClient(application) as test_client:
        yield test_client
    application.dependency_overrides.clear()


@pytest.fixture
def raw_client() -> Generator[TestClient, None, None]:
    """TestClient without dependency overrides (for integration tests)."""
    application = create_app()
    with TestClient(application) as test_client:
        yield test_client


def make_settings(**overrides: Any) -> Settings:
    """Build Settings from explicit kwargs without reading a polluted process env."""
    clear_settings_cache()
    base: dict[str, Any] = {
        "app_env": "test",
        "log_level": "WARNING",
        "groq_api_key": "test-groq-key",
        "huggingface_api_key": "test-hf-key",
        "qdrant_url": "http://localhost:6333",
        "qdrant_collection_name": "documents_test",
        "embedding_dimension": 384,
    }
    base.update(overrides)
    return Settings(**base)
