"""Context optimization for retrieved chunks."""

from __future__ import annotations

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.context_optimization.base import ContextOptimizer
from app.services.context_optimization.models import (
    ContextOptimizationMetadata,
    ContextOptimizationResult,
)
from app.services.context_optimization.tokens import estimate_chunk_prompt_tokens
from app.services.retrieval.service import RetrievedChunk

logger = get_logger(__name__)

_RELIABLE_SCORE_MIN = 0.01
_RELIABLE_SCORE_MAX = 1.0
_REDUNDANCY_OVERLAP_THRESHOLD = 0.85


class ContextOptimizationService(ContextOptimizer):
    """Filter, deduplicate, and budget retrieved context before prompting."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def optimize(self, chunks: list[RetrievedChunk]) -> ContextOptimizationResult:
        if not self._settings.context_optimization_enabled:
            return self._passthrough(chunks)

        if not chunks:
            return ContextOptimizationResult(
                chunks=[],
                removed_count=0,
                estimated_tokens=0,
            )

        metadata = ContextOptimizationMetadata()
        deduped = self._deduplicate(chunks, metadata)
        score_filtered = self._filter_by_score(deduped, metadata)
        non_redundant = self._remove_redundant(score_filtered, metadata)
        selected, estimated_tokens = self._apply_limits(non_redundant, metadata)

        removed_count = len(chunks) - len(selected)
        logger.info(
            "context_optimization_completed",
            extra={
                "operation": "optimize_context",
                "input_count": len(chunks),
                "output_count": len(selected),
                "removed_count": removed_count,
                "estimated_tokens": estimated_tokens,
                "duplicate_removed": metadata.duplicate_removed,
                "score_filtered": metadata.score_filtered,
                "redundant_removed": metadata.redundant_removed,
                "max_chunks_truncated": metadata.max_chunks_truncated,
                "token_budget_truncated": metadata.token_budget_truncated,
            },
        )
        return ContextOptimizationResult(
            chunks=selected,
            removed_count=removed_count,
            estimated_tokens=estimated_tokens,
            metadata=metadata,
        )

    def _passthrough(self, chunks: list[RetrievedChunk]) -> ContextOptimizationResult:
        estimated_tokens = sum(estimate_chunk_prompt_tokens(chunk) for chunk in chunks)
        return ContextOptimizationResult(
            chunks=chunks,
            removed_count=0,
            estimated_tokens=estimated_tokens,
        )

    def _deduplicate(
        self,
        chunks: list[RetrievedChunk],
        metadata: ContextOptimizationMetadata,
    ) -> list[RetrievedChunk]:
        seen: set[str] = set()
        deduped: list[RetrievedChunk] = []
        for chunk in chunks:
            if chunk.chunk_id in seen:
                metadata.duplicate_removed += 1
                continue
            seen.add(chunk.chunk_id)
            deduped.append(chunk)
        return deduped

    def _filter_by_score(
        self,
        chunks: list[RetrievedChunk],
        metadata: ContextOptimizationMetadata,
    ) -> list[RetrievedChunk]:
        min_score = self._settings.context_min_score
        if min_score <= 0.0:
            return chunks

        kept: list[RetrievedChunk] = []
        for chunk in chunks:
            if not _has_reliable_score(chunk.score):
                kept.append(chunk)
                continue
            if chunk.score < min_score:
                metadata.score_filtered += 1
                continue
            kept.append(chunk)
        return kept

    def _remove_redundant(
        self,
        chunks: list[RetrievedChunk],
        metadata: ContextOptimizationMetadata,
    ) -> list[RetrievedChunk]:
        kept: list[RetrievedChunk] = []
        for chunk in chunks:
            if _is_redundant_with_selected(chunk, kept):
                metadata.redundant_removed += 1
                continue
            kept.append(chunk)
        return kept

    def _apply_limits(
        self,
        chunks: list[RetrievedChunk],
        metadata: ContextOptimizationMetadata,
    ) -> tuple[list[RetrievedChunk], int]:
        max_chunks = self._settings.context_max_chunks
        max_tokens = self._settings.context_max_tokens

        selected: list[RetrievedChunk] = []
        token_total = 0

        for chunk in chunks:
            if len(selected) >= max_chunks:
                metadata.max_chunks_truncated += 1
                continue

            chunk_tokens = estimate_chunk_prompt_tokens(chunk)
            if selected and token_total + chunk_tokens > max_tokens:
                metadata.token_budget_truncated += 1
                continue

            selected.append(chunk)
            token_total += chunk_tokens

            if token_total > max_tokens:
                break

        return selected, token_total


def _has_reliable_score(score: float) -> bool:
    """Return whether a score looks like a vector similarity value."""
    return _RELIABLE_SCORE_MIN <= score <= _RELIABLE_SCORE_MAX


def _word_overlap_ratio(left: str, right: str) -> float:
    left_words = set(left.casefold().split())
    right_words = set(right.casefold().split())
    if not left_words or not right_words:
        return 0.0
    overlap = len(left_words & right_words)
    return overlap / min(len(left_words), len(right_words))


def _is_redundant_with_selected(
    candidate: RetrievedChunk,
    selected: list[RetrievedChunk],
) -> bool:
    for existing in selected:
        if _word_overlap_ratio(candidate.text, existing.text) >= _REDUNDANCY_OVERLAP_THRESHOLD:
            return True
    return False
