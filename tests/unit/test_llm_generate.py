"""Unit tests for Groq LLM generate."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from groq import APITimeoutError

from app.core.exceptions import ProviderError
from app.services.llm.groq import GroqLLMService
from tests.conftest import make_settings


@patch("app.services.llm.groq.Groq")
def test_groq_generate_returns_content(mock_groq: MagicMock) -> None:
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = "Hello from Groq"
    choice.finish_reason = "stop"
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    mock_groq.return_value = client

    service = GroqLLMService(make_settings())
    answer = service.generate("user question", system_prompt="system rules")
    assert answer == "Hello from Groq"
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"]
    assert kwargs["max_completion_tokens"]
    assert "max_tokens" not in kwargs
    assert kwargs["messages"][0]["role"] == "system"
    assert kwargs["messages"][1]["content"] == "user question"


@patch("app.services.llm.groq.Groq")
def test_groq_generate_uses_low_reasoning_effort_for_gpt_oss(mock_groq: MagicMock) -> None:
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = "Answer"
    choice.finish_reason = "stop"
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    mock_groq.return_value = client

    service = GroqLLMService(make_settings(groq_model="openai/gpt-oss-20b"))
    service.generate("question")

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["max_completion_tokens"]


@patch("app.services.llm.groq.Groq")
def test_groq_generate_retries_when_completion_budget_exhausted(mock_groq: MagicMock) -> None:
    client = MagicMock()
    empty_choice = MagicMock()
    empty_choice.message.content = ""
    empty_choice.finish_reason = "length"
    full_choice = MagicMock()
    full_choice.message.content = "Retried answer"
    full_choice.finish_reason = "stop"
    client.chat.completions.create.side_effect = [
        MagicMock(choices=[empty_choice]),
        MagicMock(choices=[full_choice]),
    ]
    mock_groq.return_value = client

    service = GroqLLMService(make_settings(groq_model="openai/gpt-oss-20b", llm_max_tokens=1024))
    answer = service.generate("question")

    assert answer == "Retried answer"
    assert client.chat.completions.create.call_count == 2
    first_tokens = client.chat.completions.create.call_args_list[0].kwargs["max_completion_tokens"]
    second_tokens = client.chat.completions.create.call_args_list[1].kwargs["max_completion_tokens"]
    assert first_tokens == 1024
    assert second_tokens == 2048


@patch("app.services.llm.groq.Groq")
def test_groq_generate_timeout(mock_groq: MagicMock) -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())
    mock_groq.return_value = client

    service = GroqLLMService(make_settings())
    with pytest.raises(ProviderError) as exc_info:
        service.generate("q")
    assert exc_info.value.details.get("reason") == "timeout"


@patch("app.services.llm.groq.Groq")
def test_groq_generate_empty_response(mock_groq: MagicMock) -> None:
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = "   "
    choice.finish_reason = "stop"
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    mock_groq.return_value = client

    service = GroqLLMService(make_settings())
    with pytest.raises(ProviderError) as exc_info:
        service.generate("q")
    assert exc_info.value.details.get("reason") == "empty_response"
