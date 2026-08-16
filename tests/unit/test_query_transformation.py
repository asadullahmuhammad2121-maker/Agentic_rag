"""Unit tests for query transformation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core.exceptions import ProviderError
from app.services.query_transformation.service import QueryTransformationService
from tests.conftest import make_settings


@pytest.fixture
def llm() -> MagicMock:
    service = MagicMock()
    service.generate.return_value = "RAG definition and architecture"
    return service


@pytest.fixture
def transformer(llm: MagicMock) -> QueryTransformationService:
    settings = make_settings(query_transformation_enabled=True)
    return QueryTransformationService(settings, llm)


def test_disabled_transformation_returns_original(llm: MagicMock) -> None:
    settings = make_settings(query_transformation_enabled=False)
    service = QueryTransformationService(settings, llm)
    result = service.transform("Can you please explain what RAG is?")

    assert result.original_query == "Can you please explain what RAG is?"
    assert result.transformed_query == result.original_query
    assert result.was_transformed is False
    llm.generate.assert_not_called()


def test_successful_rewrite(transformer: QueryTransformationService, llm: MagicMock) -> None:
    llm.generate.return_value = "RAG architecture overview"
    original = "Can you please tell me what RAG is and how it works in detail?"
    result = transformer.transform(original)

    assert result.original_query == original
    assert result.transformed_query == "RAG architecture overview"
    assert result.was_transformed is True
    llm.generate.assert_called_once()
    assert "Original question:" in llm.generate.call_args.args[0]


def test_groq_failure_falls_back_to_original(transformer: QueryTransformationService, llm: MagicMock) -> None:
    llm.generate.side_effect = ProviderError("fail", provider="groq")
    original = "Can you please explain what RAG is in this project?"
    result = transformer.transform(original)

    assert result.original_query == original
    assert result.transformed_query == original
    assert result.was_transformed is False


def test_empty_output_falls_back_to_original(transformer: QueryTransformationService, llm: MagicMock) -> None:
    llm.generate.return_value = "   "
    original = "Could you tell me about vector databases please?"
    result = transformer.transform(original)

    assert result.transformed_query == original
    assert result.was_transformed is False


def test_invalid_output_falls_back_to_original(transformer: QueryTransformationService, llm: MagicMock) -> None:
    llm.generate.return_value = ""
    original = "Please tell me what chunk overlap means in this system?"
    result = transformer.transform(original)

    assert result.transformed_query == original
    assert result.was_transformed is False


def test_clear_simple_query_skips_transformation(transformer: QueryTransformationService, llm: MagicMock) -> None:
    original = "What is RAG?"
    result = transformer.transform(original)

    assert result.original_query == original
    assert result.transformed_query == original
    assert result.was_transformed is False
    llm.generate.assert_not_called()


def test_unchanged_rewrite_not_marked_transformed(transformer: QueryTransformationService, llm: MagicMock) -> None:
    original = "Can you please explain what chunk overlap means in retrieval systems?"
    llm.generate.return_value = original
    result = transformer.transform(original)

    assert result.transformed_query == original
    assert result.was_transformed is False
