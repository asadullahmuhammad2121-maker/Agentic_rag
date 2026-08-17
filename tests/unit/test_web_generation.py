"""Unit tests for web answer generation from Tavily results."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core.exceptions import ProviderError
from app.services.agent.generation.web import (
    EMPTY_WEB_SEARCH_ANSWER,
    WebAnswerGenerator,
    web_results_to_citations,
)
from app.services.agent.models import TavilySearchOutput, WebSearchResultItem
from tests.conftest import make_settings


@pytest.fixture
def llm() -> MagicMock:
    service = MagicMock()
    service.generate.return_value = "Latest AI news summary. [S1]"
    return service


@pytest.fixture
def generator(llm: MagicMock) -> WebAnswerGenerator:
    return WebAnswerGenerator(llm, make_settings())


def test_web_results_to_citations_preserves_urls() -> None:
    results = [
        WebSearchResultItem(
            title="AI News",
            url="https://example.com/ai",
            content="Snippet",
            score=0.88,
        )
    ]
    citations = web_results_to_citations(results)
    assert len(citations) == 1
    assert citations[0].source == "https://example.com/ai"
    assert citations[0].file_type == "web"
    assert citations[0].label == "S1"


def test_generate_from_web_results(generator: WebAnswerGenerator, llm: MagicMock) -> None:
    output = TavilySearchOutput(
        query="latest AI news",
        results=[
            WebSearchResultItem(
                title="AI News",
                url="https://example.com/ai",
                content="Major announcement.",
                score=0.9,
            )
        ],
    )

    answer, citations = generator.generate("latest AI news", output)

    assert "Latest AI news summary" in answer
    assert len(citations) == 1
    assert citations[0].filename == "AI News"
    llm.generate.assert_called_once()
    assert "system_prompt" in llm.generate.call_args.kwargs


def test_generate_from_empty_web_results(generator: WebAnswerGenerator, llm: MagicMock) -> None:
    answer, citations = generator.generate("unknown", TavilySearchOutput(query="unknown", results=[]))
    assert answer == EMPTY_WEB_SEARCH_ANSWER
    assert citations == []
    llm.generate.assert_not_called()


def test_generate_from_web_propagates_provider_errors(generator: WebAnswerGenerator, llm: MagicMock) -> None:
    llm.generate.side_effect = ProviderError("fail", provider="groq")
    output = TavilySearchOutput(
        query="latest AI news",
        results=[
            WebSearchResultItem(
                title="AI News",
                url="https://example.com/ai",
                content="Major announcement.",
                score=0.9,
            )
        ],
    )
    with pytest.raises(ProviderError):
        generator.generate("latest AI news", output)
