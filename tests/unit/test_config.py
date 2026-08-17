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
    assert settings.chunking_strategy == "fixed"
    assert settings.query_transformation_enabled is False
    assert settings.multi_query_enabled is False
    assert settings.multi_query_count == 3
    assert settings.context_optimization_enabled is False
    assert settings.context_max_chunks == 8
    assert settings.context_max_tokens == 6000
    assert settings.context_min_score == 0.0
    assert settings.agent_max_steps == 2
    assert settings.agent_routing_enabled is True
    assert settings.agent_routing_max_tokens == 256
    assert settings.agent_planning_enabled is True
    assert settings.agent_planning_max_tokens == 512
    assert settings.tavily_enabled is False
    assert settings.tavily_max_results == 5
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50
    assert settings.chunk_min_size == 20
    assert settings.chunk_max_size == 2000
    assert settings.embedding_batch_size == 16
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.max_upload_file_size_bytes == 25 * 1024 * 1024


def test_settings_rejects_invalid_chunk_overlap() -> None:
    with pytest.raises(ValidationError):
        make_settings(chunk_size=100, chunk_overlap=100)


def test_settings_rejects_chunk_min_size_above_chunk_size() -> None:
    with pytest.raises(ValidationError):
        make_settings(chunk_size=100, chunk_min_size=150)


def test_settings_rejects_invalid_qdrant_url() -> None:
    with pytest.raises(ValidationError):
        make_settings(qdrant_url="not-a-url")


def test_settings_rejects_agent_max_steps_below_one() -> None:
    with pytest.raises(ValidationError):
        make_settings(agent_max_steps=0)


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
    settings = make_settings(groq_api_key="super-secret-key", tavily_api_key="tavily-secret")
    rendered = repr(settings)
    assert "super-secret-key" not in rendered
    assert "tavily-secret" not in rendered


def test_web_search_enabled_alias_enables_tavily() -> None:
    settings = make_settings(web_search_enabled=True, tavily_api_key="test-key")
    assert settings.tavily_enabled is True
    assert settings.tavily_configured is True


def test_tavily_configured_requires_enabled_and_key() -> None:
    disabled = make_settings(tavily_enabled=False, tavily_api_key="key")
    assert disabled.tavily_configured is False
    enabled = make_settings(tavily_enabled=True, tavily_api_key="key")
    assert enabled.tavily_configured is True
