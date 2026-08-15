"""Provider-independent LLM service interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMService(ABC):
    """Abstract LLM provider used by future generation / RAG phases."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return a stable provider identifier (e.g. 'groq')."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the configured model name."""

    @abstractmethod
    def health_check(self) -> bool:
        """
        Verify the provider client is configured and reachable enough to use later.

        Phase 1A: validate client construction without performing generation.
        """

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a completion for ``prompt``.

        Implemented by providers in Phase 1E.
        """
