"""Context optimization result models."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.services.retrieval.service import RetrievedChunk


@dataclass(slots=True)
class ContextOptimizationMetadata:
    """Breakdown of optimization decisions."""

    duplicate_removed: int = 0
    score_filtered: int = 0
    redundant_removed: int = 0
    max_chunks_truncated: int = 0
    token_budget_truncated: int = 0

    @property
    def total_removed(self) -> int:
        return (
            self.duplicate_removed
            + self.score_filtered
            + self.redundant_removed
            + self.max_chunks_truncated
            + self.token_budget_truncated
        )


@dataclass(slots=True, frozen=True)
class ContextOptimizationResult:
    """Optimized context ready for prompt construction."""

    chunks: list[RetrievedChunk]
    removed_count: int
    estimated_tokens: int
    metadata: ContextOptimizationMetadata = field(default_factory=ContextOptimizationMetadata)
