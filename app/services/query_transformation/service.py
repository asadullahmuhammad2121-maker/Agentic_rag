"""Rewrite user queries into search-oriented retrieval queries."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.config import Settings
from app.core.exceptions import AppError
from app.core.logging import get_logger
from app.services.llm.base import LLMService

logger = get_logger(__name__)

REWRITE_SYSTEM_PROMPT = (
    "You rewrite user questions into concise search queries for a document retrieval system. "
    "Remove conversational filler and unnecessary wording. "
    "Preserve important technical terms, names, numbers, and entities. "
    "If the question is already clear and search-ready, return it unchanged. "
    "Return only the rewritten query with no explanation, preamble, or quotes."
)

_CONVERSATIONAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(can you|could you|please|tell me|i want to know)\b", re.IGNORECASE),
    re.compile(r"\b(can you|could you|please tell me|i would like to know)\b", re.IGNORECASE),
)


@dataclass(frozen=True, slots=True)
class TransformedQuery:
    """Original and retrieval-ready query pair."""

    original_query: str
    transformed_query: str
    was_transformed: bool


class QueryTransformationService:
    """Transform user queries for retrieval while preserving the original for generation."""

    def __init__(self, settings: Settings, llm_service: LLMService) -> None:
        self._settings = settings
        self._llm = llm_service

    def transform(self, query: str) -> TransformedQuery:
        """Return a retrieval query, falling back to the original when disabled or on failure."""
        original = query.strip()
        if not original:
            return TransformedQuery(
                original_query=original,
                transformed_query=original,
                was_transformed=False,
            )

        if not self._settings.query_transformation_enabled:
            logger.debug(
                "query_transformation_disabled",
                extra={"operation": "transform_query", "query_length": len(original)},
            )
            return TransformedQuery(
                original_query=original,
                transformed_query=original,
                was_transformed=False,
            )

        if not self._should_attempt_transformation(original):
            logger.info(
                "query_transformation_skipped_clear_query",
                extra={"operation": "transform_query", "query_length": len(original)},
            )
            return TransformedQuery(
                original_query=original,
                transformed_query=original,
                was_transformed=False,
            )

        try:
            rewritten = self._rewrite_with_groq(original)
            normalized = _normalize_rewritten_query(rewritten)
            if not normalized:
                logger.warning(
                    "query_transformation_empty_output",
                    extra={"operation": "transform_query"},
                )
                return _unchanged(original)

            if normalized.casefold() == original.casefold():
                return _unchanged(original)

            logger.info(
                "query_transformation_completed",
                extra={
                    "operation": "transform_query",
                    "original_length": len(original),
                    "transformed_length": len(normalized),
                },
            )
            return TransformedQuery(
                original_query=original,
                transformed_query=normalized,
                was_transformed=True,
            )
        except AppError as exc:
            logger.warning(
                "query_transformation_failed",
                extra={
                    "operation": "transform_query",
                    "error_code": exc.code,
                    "error_type": type(exc).__name__,
                },
            )
            return _unchanged(original)
        except Exception as exc:
            logger.warning(
                "query_transformation_failed",
                extra={
                    "operation": "transform_query",
                    "error_type": type(exc).__name__,
                },
            )
            return _unchanged(original)

    def _should_attempt_transformation(self, query: str) -> bool:
        if len(query) <= 20:
            return False
        return any(pattern.search(query) for pattern in _CONVERSATIONAL_PATTERNS) or len(query) > 80

    def _rewrite_with_groq(self, query: str) -> str:
        user_prompt = f"Original question:\n{query}\n\nSearch query:"
        return self._llm.generate(
            user_prompt,
            system_prompt=REWRITE_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=self._settings.query_transformation_max_tokens,
        )


def _normalize_rewritten_query(text: str) -> str:
    cleaned = text.strip().strip("\"'`")
    cleaned = cleaned.removeprefix("Search query:").strip()
    return cleaned.strip()


def _unchanged(original: str) -> TransformedQuery:
    return TransformedQuery(
        original_query=original,
        transformed_query=original,
        was_transformed=False,
    )
