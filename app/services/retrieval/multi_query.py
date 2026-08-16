"""Multi-query retrieval using Groq-generated search variants."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.config import Settings
from app.core.exceptions import AppError
from app.core.logging import get_logger
from app.services.llm.base import LLMService
from app.services.retrieval.combiner import combine_retrieved_chunks
from app.services.retrieval.filters import RetrievalFilters
from app.services.retrieval.hybrid import HybridRetrievalService
from app.services.retrieval.service import RetrievalService, RetrievedChunk

logger = get_logger(__name__)

MULTI_QUERY_SYSTEM_PROMPT = (
    "You generate diverse search queries for a document retrieval system. "
    "Each query should explore a different angle, synonym, or subtopic of the same intent. "
    "Preserve important technical terms, names, numbers, and entities. "
    "Do not repeat the same wording. "
    "Return one query per line with no numbering, bullets, or explanations."
)

_LINE_PREFIX_PATTERN = re.compile(r"^\s*(?:\d+[\).\:-]?\s*|[-*•]\s*)")


@dataclass(frozen=True, slots=True)
class GeneratedQueries:
    """Queries produced for multi-query retrieval."""

    basis_query: str
    queries: tuple[str, ...]


class MultiQueryGenerator:
    """Generate diverse retrieval queries from a basis query using Groq."""

    def __init__(self, settings: Settings, llm_service: LLMService) -> None:
        self._settings = settings
        self._llm = llm_service

    def generate(self, basis_query: str) -> GeneratedQueries:
        basis = basis_query.strip()
        if not basis:
            return GeneratedQueries(basis_query=basis, queries=())

        user_prompt = (
            f"Original question:\n{basis}\n\n"
            f"Generate exactly {self._settings.multi_query_count} diverse search queries."
        )
        raw = self._llm.generate(
            user_prompt,
            system_prompt=MULTI_QUERY_SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=self._settings.multi_query_max_tokens,
        )
        queries = _normalize_generated_queries(
            raw,
            basis=basis,
            target_count=self._settings.multi_query_count,
        )
        return GeneratedQueries(basis_query=basis, queries=queries)


class MultiQueryRetrievalService:
    """Retrieve using one or many search queries and combine the results."""

    def __init__(
        self,
        settings: Settings,
        retrieval_service: RetrievalService | HybridRetrievalService,
        llm_service: LLMService,
        *,
        generator: MultiQueryGenerator | None = None,
    ) -> None:
        self._settings = settings
        self._retrieval = retrieval_service
        self._generator = generator or MultiQueryGenerator(settings, llm_service)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks using single- or multi-query search."""
        if not self._settings.multi_query_enabled:
            return self._retrieval.retrieve(
                query,
                top_k=top_k,
                filters=filters,
                score_threshold=score_threshold,
            )

        limit = top_k if top_k is not None else self._settings.retrieval_top_k
        basis = query.strip()

        try:
            generated = self._generator.generate(basis)
            queries = generated.queries
            if len(queries) <= 1:
                logger.info(
                    "multi_query_fallback_single",
                    extra={
                        "operation": "multi_query_retrieve",
                        "reason": "insufficient_generated_queries",
                        "query_count": len(queries),
                    },
                )
                return self._retrieval.retrieve(
                    basis,
                    top_k=top_k,
                    filters=filters,
                    score_threshold=score_threshold,
                )

            logger.info(
                "multi_query_retrieval_started",
                extra={
                    "operation": "multi_query_retrieve",
                    "basis_query_length": len(basis),
                    "query_count": len(queries),
                    "has_filters": filters is not None and not filters.is_empty(),
                },
            )

            chunk_groups: list[list[RetrievedChunk]] = []
            for search_query in queries:
                chunk_groups.append(
                    self._retrieval.retrieve(
                        search_query,
                        top_k=top_k,
                        filters=filters,
                        score_threshold=score_threshold,
                    )
                )

            combined = combine_retrieved_chunks(chunk_groups, limit=limit)
            logger.info(
                "multi_query_retrieval_completed",
                extra={
                    "operation": "multi_query_retrieve",
                    "query_count": len(queries),
                    "result_count": len(combined),
                },
            )
            return combined
        except AppError as exc:
            logger.warning(
                "multi_query_generation_failed",
                extra={
                    "operation": "multi_query_retrieve",
                    "error_code": exc.code,
                    "error_type": type(exc).__name__,
                },
            )
        except Exception as exc:
            logger.warning(
                "multi_query_generation_failed",
                extra={
                    "operation": "multi_query_retrieve",
                    "error_type": type(exc).__name__,
                },
            )

        return self._retrieval.retrieve(
            basis,
            top_k=top_k,
            filters=filters,
            score_threshold=score_threshold,
        )


def _normalize_generated_queries(raw: str, *, basis: str, target_count: int) -> tuple[str, ...]:
    candidates: list[str] = []
    for line in raw.splitlines():
        cleaned = _LINE_PREFIX_PATTERN.sub("", line.strip()).strip("\"'`")
        if cleaned:
            candidates.append(cleaned)

    if not candidates:
        candidates = [basis]

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = _normalize_query_text(candidate)
        if not normalized or _is_meaningless_query(normalized):
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(normalized)

    basis_normalized = _normalize_query_text(basis)
    if basis_normalized and basis_normalized.casefold() not in seen:
        unique.insert(0, basis_normalized)

    if not unique:
        return (basis_normalized,) if basis_normalized else ()

    return tuple(unique[:target_count])


def _normalize_query_text(text: str) -> str:
    return text.strip().strip("\"'`")


def _is_meaningless_query(text: str) -> bool:
    return len(text) < 3
