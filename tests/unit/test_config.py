"""Unit tests for configuration loading and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tests.conftest import make_settings


def test_settings_loads_defaults() -> None:
    settings = make_settings()
    assert settings.app_name == "rag-foundation"
    assert settings.app_version == "0.1.0"
    assert settings.embedding_dimension == 384
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50
    assert settings.embedding_batch_size == 16
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.max_upload_file_size_bytes == 25 * 1024 * 1024


def test_settings_rejects_invalid_chunk_overlap() -> None:
    with pytest.raises(ValidationError):
        make_settings(chunk_size=100, chunk_overlap=100)


def test_settings_rejects_invalid_qdrant_url() -> None:
    with pytest.raises(ValidationError):
        make_settings(qdrant_url="not-a-url")


def test_settings_normalizes_qdrant_url_trailing_slash() -> None:
    settings = make_settings(qdrant_url="http://localhost:6333/")
    assert settings.qdrant_url == "http://localhost:6333"


def test_settings_rejects_empty_model_name() -> None:
    with pytest.raises(ValidationError):
        make_settings(groq_model="   ")


def test_settings_rejects_empty_collection_name() -> None:
    with pytest.raises(ValidationError):
        make_settings(qdrant_collection_name="")


def test_production_requires_api_keys() -> None:
    with pytest.raises(ValidationError) as exc_info:
        make_settings(
            app_env="production",
            groq_api_key="",
            huggingface_api_key="",
        )
    assert "GROQ_API_KEY" in str(exc_info.value)
    assert "HUGGINGFACE_API_KEY" in str(exc_info.value)


def test_production_accepts_present_api_keys() -> None:
    settings = make_settings(
        app_env="production",
        groq_api_key="real-groq",
        huggingface_api_key="real-hf",
    )
    assert settings.is_production is True


def test_secrets_not_exposed_in_repr() -> None:
    settings = make_settings(groq_api_key="super-secret-key")
    rendered = repr(settings)
    assert "super-secret-key" not in rendered
