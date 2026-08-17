"""Prompt construction for grounded RAG answers."""

from __future__ import annotations

from collections.abc import Callable
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

HYBRID_SYSTEM_INSTRUCTIONS = (
    "You are a careful assistant that answers questions using uploaded document context "
    "and web search results provided below. "
    "Use document sources for information from uploaded files and web sources for external "
    "or current information. "
    "Answer each part of the question from the most appropriate source type. "
    "If document context covers one part and web results cover another, combine both. "
    "Only say you lack enough information when neither source type contains relevant material. "
    "Do not invent facts, documents, page numbers, URLs, or citations. "
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

    def build_combined(self, query: str, chunks: list[RetrievedChunk]) -> BuiltPrompt:
        """Build a prompt for hybrid agent generation (uploaded documents + web results)."""
        context_block = self._format_combined_context(chunks)
        user_prompt = (
            f"{context_block}\n\n"
            f"Question: {query.strip()}\n\n"
            "Answer using the document context and/or web search results above as appropriate. "
            "Use web results for current or external comparisons when the question asks for them. "
            "Only say you do not have enough information if neither source type is relevant."
        )
        logger.info(
            "combined_prompt_built",
            extra={
                "operation": "build_combined_prompt",
                "context_chunk_count": len(chunks),
                "document_chunk_count": sum(1 for chunk in chunks if chunk.file_type != "web"),
                "web_chunk_count": sum(1 for chunk in chunks if chunk.file_type == "web"),
                "query_length": len(query.strip()),
                "user_prompt_length": len(user_prompt),
            },
        )
        return BuiltPrompt(
            system_prompt=HYBRID_SYSTEM_INSTRUCTIONS,
            user_prompt=user_prompt,
            context_chunk_count=len(chunks),
        )

    def _format_combined_context(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return (
                "[No document context or web search results were retrieved. "
                "You must say you do not have enough information.]"
            )

        blocks: list[str] = []
        document_sections = self._format_chunk_sections(chunks, include=lambda c: c.file_type != "web")
        if document_sections:
            blocks.append(f"Uploaded document context:\n{document_sections}")
        web_sections = self._format_chunk_sections(
            chunks,
            include=lambda c: c.file_type == "web",
            web=True,
        )
        if web_sections:
            blocks.append(f"Web search results:\n{web_sections}")
        return "\n\n".join(blocks)

    def _format_chunk_sections(
        self,
        chunks: list[RetrievedChunk],
        *,
        include: Callable[[RetrievedChunk], bool],
        web: bool = False,
    ) -> str:
        sections: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            if not include(chunk):
                continue
            label = f"S{index}"
            if web:
                header = (
                    f"[{label}] title={chunk.filename} url={chunk.source} "
                    f"file_type=web score={chunk.score:.4f}"
                )
            else:
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
