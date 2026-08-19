"""Generate grounded answers from Tavily web search results."""

from __future__ import annotations

from app.core.config import Settings
from app.core.exceptions import AppError, QueryError
from app.core.logging import get_logger
from app.services.agent.models import AgentCitation, TavilySearchOutput, WebSearchResultItem
from app.services.llm.base import LLMService

logger = get_logger(__name__)

EMPTY_WEB_SEARCH_ANSWER = (
    "I could not find relevant web information to answer that question."
)

WEB_SYSTEM_INSTRUCTIONS = (
    "You are a careful assistant that answers questions using only the provided web search results. "
    "If the results contain relevant information, answer with what is supported by the results, "
    "even when some requested details are missing. "
    "Provide partial answers when the results answer part of the question; briefly note which "
    "details are not present in the results. "
    "Only say you do not have enough information when the results contain no meaningful or "
    "relevant information for the question. "
    "Do not invent facts, URLs, dates, names, or citations beyond what appears in the results. "
    "When you use information from the results, cite sources using their labels "
    "(for example [S1], [S2])."
)


class WebAnswerGenerator:
    """Build prompts from web results and generate cited answers via the LLM."""

    def __init__(self, llm_service: LLMService, settings: Settings) -> None:
        self._llm = llm_service
        self._settings = settings

    def generate(self, query: str, output: TavilySearchOutput) -> tuple[str, list[AgentCitation]]:
        normalized = query.strip()
        if not normalized:
            raise QueryError(
                "Query must not be empty",
                details={"reason": "empty_query"},
            )

        if output.empty:
            logger.info(
                "web_generation_skipped_empty_results",
                extra={"operation": "generate_from_web", "result_count": 0},
            )
            return EMPTY_WEB_SEARCH_ANSWER, []

        user_prompt = self._build_prompt(normalized, output.results)
        logger.info(
            "web_generation_started",
            extra={
                "operation": "generate_from_web",
                "query_length": len(normalized),
                "result_count": output.result_count,
            },
        )
        try:
            answer_text = self._llm.generate(
                user_prompt,
                system_prompt=WEB_SYSTEM_INSTRUCTIONS,
                temperature=self._settings.llm_temperature,
                max_tokens=self._settings.llm_max_tokens,
            )
        except AppError:
            raise

        citations = web_results_to_citations(output.results)
        answer = answer_text.strip()
        logger.info(
            "web_generation_completed",
            extra={
                "operation": "generate_from_web",
                "result_count": output.result_count,
                "citation_count": len(citations),
                "answer_length": len(answer),
            },
        )
        return answer, citations

    def _build_prompt(self, query: str, results: list[WebSearchResultItem]) -> str:
        context_block = self._format_results(results)
        return (
            f"Web search results:\n{context_block}\n\n"
            f"Question: {query}\n\n"
            "Answer using only the web results above. "
            "Use all relevant facts from the results, even if they do not fully answer every part "
            "of the question. "
            "If some requested details are missing from the results, answer with what is available "
            "and briefly state which details were not found in the results. "
            "Only say you do not have enough information if none of the results are relevant."
        )

    def _format_results(self, results: list[WebSearchResultItem]) -> str:
        sections: list[str] = []
        for index, result in enumerate(results, start=1):
            label = f"S{index}"
            score_part = f" score={result.score:.4f}" if result.score is not None else ""
            header = f"[{label}] title={result.title} url={result.url}{score_part}"
            body = result.content or "[No snippet available]"
            sections.append(f"{header}\n{body}")
        return "\n\n".join(sections)


def web_results_to_citations(results: list[WebSearchResultItem]) -> list[AgentCitation]:
    """Convert web search hits into agent citations with source URLs."""
    citations: list[AgentCitation] = []
    for index, result in enumerate(results, start=1):
        citations.append(
            AgentCitation(
                document_id=result.url,
                filename=result.title,
                file_type="web",
                source=result.url,
                page_number=0,
                section=None,
                chunk_index=index - 1,
                chunk_id=result.url,
                score=result.score if result.score is not None else 0.0,
                label=f"S{index}",
            )
        )
    return citations
