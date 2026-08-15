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
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    mock_groq.return_value = client

    service = GroqLLMService(make_settings())
    answer = service.generate("user question", system_prompt="system rules")
    assert answer == "Hello from Groq"
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"]
    assert kwargs["messages"][0]["role"] == "system"
    assert kwargs["messages"][1]["content"] == "user question"


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
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    mock_groq.return_value = client

    service = GroqLLMService(make_settings())
    with pytest.raises(ProviderError) as exc_info:
        service.generate("q")
    assert exc_info.value.details.get("reason") == "empty_response"
