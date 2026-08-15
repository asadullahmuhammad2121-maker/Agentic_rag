"""Unit tests for Hugging Face embedding generation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.exceptions import ProviderError
from app.services.embeddings.huggingface import HuggingFaceEmbeddingService
from tests.conftest import make_settings


def _settings(**overrides: object) -> object:
    base = {
        "embedding_dimension": 4,
        "embedding_batch_size": 2,
        "huggingface_api_key": "hf-test",
    }
    base.update(overrides)
    return make_settings(**base)


@patch("app.services.embeddings.huggingface.InferenceClient")
def test_embed_documents_batches_and_returns_vectors(mock_client_cls: MagicMock) -> None:
    client = MagicMock()
    client.feature_extraction.side_effect = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 0.8, 0.7, 0.6],
    ]
    mock_client_cls.return_value = client

    service = HuggingFaceEmbeddingService(_settings())  # type: ignore[arg-type]
    vectors = service.embed_documents(["one", "two", "three"])

    assert len(vectors) == 3
    assert vectors[0] == [0.1, 0.2, 0.3, 0.4]
    assert client.feature_extraction.call_count == 3


@patch("app.services.embeddings.huggingface.InferenceClient")
def test_embed_documents_uses_cache_for_duplicates(mock_client_cls: MagicMock) -> None:
    client = MagicMock()
    client.feature_extraction.return_value = [1.0, 0.0, 0.0, 0.0]
    mock_client_cls.return_value = client

    service = HuggingFaceEmbeddingService(_settings())  # type: ignore[arg-type]
    vectors = service.embed_documents(["same", "same", "same"])

    assert len(vectors) == 3
    assert vectors[0] == vectors[1] == vectors[2]
    assert client.feature_extraction.call_count == 1


@patch("app.services.embeddings.huggingface.InferenceClient")
def test_embed_query_delegates(mock_client_cls: MagicMock) -> None:
    client = MagicMock()
    client.feature_extraction.return_value = [0.1, 0.2, 0.3, 0.4]
    mock_client_cls.return_value = client

    service = HuggingFaceEmbeddingService(_settings())  # type: ignore[arg-type]
    vector = service.embed_query("hello")
    assert vector == [0.1, 0.2, 0.3, 0.4]


@patch("app.services.embeddings.huggingface.InferenceClient")
def test_embed_failure_raises_provider_error(mock_client_cls: MagicMock) -> None:
    client = MagicMock()
    client.feature_extraction.side_effect = RuntimeError("upstream down")
    mock_client_cls.return_value = client

    service = HuggingFaceEmbeddingService(_settings())  # type: ignore[arg-type]
    with pytest.raises(ProviderError) as exc_info:
        service.embed_documents(["fail"])
    assert exc_info.value.details.get("provider") == "huggingface"
    assert exc_info.value.details.get("reason") == "api_failure"


@patch("app.services.embeddings.huggingface.InferenceClient")
def test_embed_dimension_mismatch(mock_client_cls: MagicMock) -> None:
    client = MagicMock()
    client.feature_extraction.return_value = [0.1, 0.2]  # expected 4
    mock_client_cls.return_value = client

    service = HuggingFaceEmbeddingService(_settings())  # type: ignore[arg-type]
    with pytest.raises(ProviderError) as exc_info:
        service.embed_documents(["x"])
    assert exc_info.value.details.get("reason") == "dimension_mismatch"


@patch("app.services.embeddings.huggingface.InferenceClient")
def test_mean_pools_token_vectors(mock_client_cls: MagicMock) -> None:
    client = MagicMock()
    client.feature_extraction.return_value = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    mock_client_cls.return_value = client

    service = HuggingFaceEmbeddingService(_settings())  # type: ignore[arg-type]
    vectors = service.embed_documents(["tokens"])
    assert vectors[0] == [0.5, 0.5, 0.0, 0.0]
