"""Unit tests for web answer generation from Tavily results."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core.exceptions import ProviderError
from app.services.agent.generation.web import (
    EMPTY_WEB_SEARCH_ANSWER,
    WEB_SYSTEM_INSTRUCTIONS,
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


def _schedule_output(*, content: str) -> TavilySearchOutput:
    return TavilySearchOutput(
        query="entity alpha test schedule",
        results=[
            WebSearchResultItem(
                title="Tour schedule",
                url="https://example.com/schedule",
                content=content,
                score=0.91,
            )
        ],
    )


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


def test_web_system_instructions_allow_partial_answers() -> None:
    lowered = WEB_SYSTEM_INSTRUCTIONS.casefold()
    assert "partial" in lowered
    assert "only say you do not have enough information when" in lowered
    assert "do not invent facts" in lowered


def test_build_prompt_encourages_partial_answer_not_full_refusal(
    generator: WebAnswerGenerator,
) -> None:
    results = [
        WebSearchResultItem(
            title="Policy overview",
            url="https://example.com/policy",
            content="Refund requests must be submitted within 30 days.",
            score=0.8,
        )
    ]
    prompt = generator._build_prompt("company refund policy", results)
    lowered = prompt.casefold()
    assert "use all relevant facts" in lowered
    assert "only say you do not have enough information if none of the results are relevant" in lowered
    assert "if the results are insufficient, say you do not have enough information" not in lowered


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
    assert llm.generate.call_args.kwargs["system_prompt"] == WEB_SYSTEM_INSTRUCTIONS


def test_generate_from_complete_web_results(generator: WebAnswerGenerator, llm: MagicMock) -> None:
    llm.generate.return_value = (
        "Based on [S1], the series runs from 19-23 August and 27-31 August 2026, "
        "with venues listed for both matches."
    )
    output = _schedule_output(
        content=(
            "The series includes Tests on 19-23 August 2026 at Venue A and "
            "27-31 August 2026 at Venue B."
        ),
    )

    answer, citations = generator.generate("entity alpha test schedule", output)

    assert "19-23 August" in answer
    assert "venues" in answer.casefold()
    assert len(citations) == 1
    assert "[S1]" in llm.generate.call_args.args[0]


def test_generate_from_partial_web_results(generator: WebAnswerGenerator, llm: MagicMock) -> None:
    llm.generate.return_value = (
        "Based on [S1], the matches are scheduled for 19-23 August, 27-31 August, "
        "and 9-13 September 2026. The retrieved sources do not provide complete "
        "venue or start-time details."
    )
    output = _schedule_output(
        content="Tests are scheduled for 19-23 August, 27-31 August, and 9-13 September 2026.",
    )

    answer, citations = generator.generate("entity alpha test schedule", output)

    assert "19-23 August" in answer
    assert "venue" in answer.casefold() or "start-time" in answer.casefold()
    assert "do not have enough information" not in answer.casefold()
    assert len(citations) == 1
    user_prompt = llm.generate.call_args.args[0]
    assert "19-23 August" in user_prompt
    assert "partial" not in user_prompt  # prompt should not leak implementation wording


def test_generate_from_irrelevant_web_results(generator: WebAnswerGenerator, llm: MagicMock) -> None:
    llm.generate.return_value = (
        "I do not have enough information to answer that question from the provided results."
    )
    output = TavilySearchOutput(
        query="product architecture",
        results=[
            WebSearchResultItem(
                title="Unrelated cooking blog",
                url="https://example.com/recipes",
                content="How to bake bread at home.",
                score=0.2,
            )
        ],
    )

    answer, citations = generator.generate("product architecture", output)

    assert "do not have enough information" in answer.casefold()
    assert len(citations) == 1
    llm.generate.assert_called_once()


def test_generate_from_empty_web_results(generator: WebAnswerGenerator, llm: MagicMock) -> None:
    answer, citations = generator.generate("unknown", TavilySearchOutput(query="unknown", results=[]))
    assert answer == EMPTY_WEB_SEARCH_ANSWER
    assert citations == []
    llm.generate.assert_not_called()


def test_prompt_forbids_inventing_facts_beyond_results() -> None:
    lowered = WEB_SYSTEM_INSTRUCTIONS.casefold()
    assert "only" in lowered and "provided web search results" in lowered
    assert "do not invent" in lowered


def test_generated_answer_preserves_citation_labels(
    generator: WebAnswerGenerator,
    llm: MagicMock,
) -> None:
    llm.generate.return_value = "The policy allows 30-day refunds. [S1][S2]"
    output = TavilySearchOutput(
        query="company refund policy",
        results=[
            WebSearchResultItem(
                title="Policy A",
                url="https://example.com/a",
                content="30-day refund window.",
                score=0.9,
            ),
            WebSearchResultItem(
                title="Policy B",
                url="https://example.com/b",
                content="Receipt required.",
                score=0.8,
            ),
        ],
    )

    answer, citations = generator.generate("company refund policy", output)

    assert "[S1]" in answer
    assert "[S2]" in answer
    assert [citation.label for citation in citations] == ["S1", "S2"]


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
