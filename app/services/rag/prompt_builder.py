"""Prompt construction for grounded RAG answers."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.logging import get_logger
from app.services.retrieval.service import RetrievedChunk

logger = get_logger(__name__)

SYSTEM_INSTRUCTIONS = (
    "You are a careful assistant that answers questions using only the provided context. "
    "If the context is missing, empty, or insufficient to answer the question, say that you "
    "do not have enough information in the knowledge base. "
    "Do not invent facts, documents, page numbers, or citations. "
    "When you use information from the context, refer to sources using their labels "
    "(for example [S1], [S2])."
)


@dataclass(slots=True, frozen=True)
class BuiltPrompt:
    """System and user messages ready for the LLM."""

    system_prompt: str
    user_prompt: str
    context_chunk_count: int


class PromptBuilder:
    """Build grounded prompts from retrieved chunks and the user query."""

    def build(self, query: str, chunks: list[RetrievedChunk]) -> BuiltPrompt:
        context_block = self._format_context(chunks)
        user_prompt = (
            f"Context:\n{context_block}\n\n"
            f"Question: {query.strip()}\n\n"
            "Answer using only the context above. "
            "If the context is insufficient, say you do not have enough information."
        )
        logger.info(
            "prompt_built",
            extra={
                "operation": "build_prompt",
                "context_chunk_count": len(chunks),
                "query_length": len(query.strip()),
                "user_prompt_length": len(user_prompt),
            },
        )
        return BuiltPrompt(
            system_prompt=SYSTEM_INSTRUCTIONS,
            user_prompt=user_prompt,
            context_chunk_count=len(chunks),
        )

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return (
                "[No relevant context was retrieved from the knowledge base. "
                "You must say you do not have enough information.]"
            )

        sections: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            label = f"S{index}"
            section_part = f" section={chunk.section}" if chunk.section else ""
            header = (
                f"[{label}] document_id={chunk.document_id} "
                f"filename={chunk.filename} "
                f"file_type={chunk.file_type} "
                f"source={chunk.source} "
                f"page={chunk.page_number}{section_part} "
                f"chunk_index={chunk.chunk_index} "
                f"chunk_id={chunk.chunk_id} "
                f"score={chunk.score:.4f}"
            )
            sections.append(f"{header}\n{chunk.text}")
        return "\n\n".join(sections)
