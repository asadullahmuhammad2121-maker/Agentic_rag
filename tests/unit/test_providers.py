"""Unit tests for provider abstractions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.services.embeddings.huggingface import HuggingFaceEmbeddingService
from app.services.llm.groq import GroqLLMService
from tests.conftest import make_settings


@patch("app.services.llm.groq.Groq")
def test_groq_llm_service_initializes(mock_groq: MagicMock) -> None:
    mock_groq.return_value = MagicMock()
    settings = make_settings()
    service = GroqLLMService(settings)

    assert service.provider_name == "groq"
    assert service.model_name == settings.groq_model
    assert service.health_check() is True
    mock_groq.assert_called_once()


@patch("app.services.embeddings.huggingface.InferenceClient")
def test_huggingface_embedding_service_initializes(mock_client: MagicMock) -> None:
    mock_client.return_value = MagicMock()
    settings = make_settings()
    service = HuggingFaceEmbeddingService(settings)

    assert service.provider_name == "huggingface"
    assert service.model_name == settings.huggingface_embedding_model
    assert service.dimension == settings.embedding_dimension
    assert service.health_check() is True


@patch("app.services.llm.groq.Groq", side_effect=RuntimeError("boom"))
def test_groq_init_failure_raises_provider_error(_mock_groq: MagicMock) -> None:
    from app.core.exceptions import ProviderError

    with pytest.raises(ProviderError) as exc_info:
        GroqLLMService(make_settings())
    assert exc_info.value.details.get("provider") == "groq"
