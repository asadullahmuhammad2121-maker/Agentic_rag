"""Lightweight token estimation helpers."""

from __future__ import annotations

from app.services.retrieval.service import RetrievedChunk

_PROMPT_HEADER_OVERHEAD_TOKENS = 80


def estimate_tokens(text: str) -> int:
    """Estimate token count using a simple chars/4 heuristic."""
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, (len(stripped) + 3) // 4)


def estimate_chunk_prompt_tokens(chunk: RetrievedChunk) -> int:
    """Estimate prompt tokens for one formatted context chunk."""
    return estimate_tokens(chunk.text) + _PROMPT_HEADER_OVERHEAD_TOKENS
