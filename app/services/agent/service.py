"""Agent orchestrator: decide → execute tool(s) → generate → observe → finish."""

from __future__ import annotations

from app.core.exceptions import AgentError, AppError, QueryError
from app.core.logging import get_logger
from app.services.agent.base import Agent
from app.services.agent.generation.calculator import format_calculator_answer
from app.services.agent.generation.combined import merge_tool_outputs_to_chunks
from app.services.agent.generation.web import WebAnswerGenerator
from app.services.agent.models import (
    AgentAction,
    AgentActionType,
    AgentObservation,
    AgentRequest,
    AgentRunResult,
    AgentStep,
    AgentTask,
    CalculatorOutput,
    RAGRetrievalOutput,
    TavilySearchOutput,
)
from app.services.agent.tools.calculator import CALCULATOR_TOOL_NAME
from app.services.agent.tools.converters import (
    citations_from_rag,
    output_to_chunk,
    tool_result_to_observation,
)
from app.services.agent.tools.rag import RAG_RETRIEVAL_TOOL_NAME
from app.services.agent.tools.registry import ToolRegistry
from app.services.agent.tools.tavily import TAVILY_WEB_SEARCH_TOOL_NAME
from app.services.rag.service import RAGService
from app.services.retrieval.filters import RetrievalFilters

logger = get_logger(__name__)


class AgentService:
    """Run the agent loop: select a tool, retrieve context, then generate an answer."""

    def __init__(
        self,
        agent: Agent,
        tools: ToolRegistry,
        rag_service: RAGService,
        web_answer_generator: WebAnswerGenerator,
        *,
        max_steps: int,
    ) -> None:
        if max_steps < 1:
            raise AgentError(
                "Agent max_steps must be at least 1",
                details={"reason": "invalid_max_steps", "max_steps": max_steps},
            )
        self._agent = agent
        self._tools = tools
        self._rag = rag_service
        self._web_generator = web_answer_generator
        self._max_steps = max_steps

    def run(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> AgentRunResult:
        request = _build_request(query, top_k=top_k, filters=filters)
        history: list[AgentStep] = []

        logger.info(
            "agent_run_started",
            extra={
                "operation": "agent_run",
                "query_length": len(request.query),
                "top_k": request.top_k,
                "has_filters": filters is not None and not filters.is_empty(),
                "max_steps": self._max_steps,
                "available_tools": self._tools.names(),
            },
        )

        for step_index in range(self._max_steps):
            action = self._agent.decide(request, tools=self._tools, history=history)
            logger.info(
                "agent_action_selected",
                extra={
                    "operation": "agent_run",
                    "step_index": step_index,
                    "action_type": str(action.type),
                    "tool_name": action.tool_name,
                    "tool_names": action.tool_names,
                },
            )
            if action.type is AgentActionType.FINISH:
                history.append(AgentStep(action=action, observation=None))
                result = self._result_from_finish(action, history)
                self._log_completed(result, step_count=len(history), finished=True)
                return result

            observation = self._execute_action(action)
            observation = self._generate_answer(request, observation)
            history.append(AgentStep(action=action, observation=observation))

        result = self._result_from_history(history)
        self._log_completed(result, step_count=len(history), finished=False)
        return result

    def _execute_action(self, action: AgentAction) -> AgentObservation:
        if action.type is AgentActionType.EXECUTE_PLAN:
            return self._execute_plan(action)
        if action.type is AgentActionType.CALL_TOOLS or len(action.tool_names) > 1:
            return self._execute_tools(action)
        return self._execute_tool(action)

    def _execute_plan(self, action: AgentAction) -> AgentObservation:
        observations: list[AgentObservation] = []
        for task in action.tasks:
            single_action = AgentAction(
                type=AgentActionType.CALL_TOOL,
                tool_name=task.tool_name,
                tool_names=[task.tool_name],
                arguments=_task_arguments(action.arguments, task),
                reasoning=task.reasoning or action.reasoning,
            )
            try:
                observations.append(self._execute_tool(single_action))
            except AppError as exc:
                observations.append(
                    AgentObservation(
                        tool_name=task.tool_name,
                        success=False,
                        error=exc.message,
                        metadata={"error_code": exc.code, **exc.details},
                    )
                )
        merged = self._merge_tool_observations(observations)
        metadata = dict(merged.metadata)
        metadata["decomposed"] = True
        metadata["plan_tasks"] = [task.model_dump() for task in action.tasks]
        return merged.model_copy(update={"metadata": metadata})

    def _execute_tools(self, action: AgentAction) -> AgentObservation:
        tool_names = action.tool_names or ([action.tool_name] if action.tool_name else [])
        observations: list[AgentObservation] = []
        for tool_name in tool_names:
            single_action = AgentAction(
                type=AgentActionType.CALL_TOOL,
                tool_name=tool_name,
                tool_names=[tool_name],
                arguments=action.arguments,
                reasoning=action.reasoning,
            )
            try:
                observations.append(self._execute_tool(single_action))
            except AppError as exc:
                observations.append(
                    AgentObservation(
                        tool_name=tool_name,
                        success=False,
                        error=exc.message,
                        metadata={"error_code": exc.code, **exc.details},
                    )
                )
        return self._merge_tool_observations(observations)

    def _merge_tool_observations(
        self,
        observations: list[AgentObservation],
    ) -> AgentObservation:
        if len(observations) == 1:
            return observations[0]

        successful = [observation for observation in observations if observation.success]
        failed = [observation for observation in observations if not observation.success]
        if not successful:
            errors = {
                observation.tool_name: observation.error or "tool failed"
                for observation in failed
            }
            return AgentObservation(
                tool_name="+".join(observation.tool_name for observation in observations),
                tool_names=[observation.tool_name for observation in observations],
                success=False,
                error="All selected tools failed",
                metadata={
                    "failed_tools": list(errors.keys()),
                    "errors": errors,
                },
            )

        tool_names = [observation.tool_name for observation in successful]
        tool_outputs = {
            observation.tool_name: observation.tool_output
            for observation in successful
            if observation.tool_output is not None
        }
        metadata: dict[str, object] = {
            "multi_tool": True,
            "tools_used": tool_names,
            "failed_tools": [observation.tool_name for observation in failed],
            "tool_outputs": tool_outputs,
        }
        if failed:
            metadata["partial_success"] = True
            metadata["errors"] = {
                observation.tool_name: observation.error or "tool failed"
                for observation in failed
            }
        return AgentObservation(
            tool_name="+".join(tool_names),
            tool_names=tool_names,
            success=True,
            metadata=metadata,
        )

    def _execute_tool(self, action: AgentAction) -> AgentObservation:
        tool_name = action.tool_name or ""
        tool = self._tools.get(tool_name)
        if tool is None:
            raise AgentError(
                "The agent selected an unknown tool",
                details={"reason": "unknown_tool", "tool_name": tool_name},
            )

        logger.info(
            "agent_tool_started",
            extra={"operation": "agent_run", "tool_name": tool.name},
        )
        tool_result = tool.run(action.arguments)
        observation = tool_result_to_observation(tool.name, tool_result)
        if not observation.tool_names:
            observation = observation.model_copy(update={"tool_names": [tool.name]})
        logger.info(
            "agent_tool_completed",
            extra={
                "operation": "agent_run",
                "tool_name": tool.name,
                "success": observation.success,
                "result_count": observation.metadata.get("result_count"),
            },
        )
        return observation

    def _generate_answer(
        self,
        request: AgentRequest,
        observation: AgentObservation,
    ) -> AgentObservation:
        if not observation.success:
            return observation
        if observation.metadata.get("multi_tool"):
            return self._generate_from_combined(request, observation)
        if observation.tool_name == RAG_RETRIEVAL_TOOL_NAME:
            return self._generate_from_rag(request, observation)
        if observation.tool_name == TAVILY_WEB_SEARCH_TOOL_NAME:
            return self._generate_from_web(request, observation)
        if observation.tool_name == CALCULATOR_TOOL_NAME:
            return self._generate_from_calculator(request, observation)
        return observation

    def _generate_from_combined(
        self,
        request: AgentRequest,
        observation: AgentObservation,
    ) -> AgentObservation:
        tool_outputs = observation.metadata.get("tool_outputs", {})
        if not isinstance(tool_outputs, dict):
            raise AgentError(
                "Combined tool observation is missing structured outputs",
                details={"reason": "missing_tool_output", "tool_name": observation.tool_name},
            )

        rag_output: RAGRetrievalOutput | None = None
        web_output: TavilySearchOutput | None = None
        calculator_output: CalculatorOutput | None = None
        for tool_name, payload in tool_outputs.items():
            if payload is None:
                continue
            if tool_name == RAG_RETRIEVAL_TOOL_NAME:
                rag_output = RAGRetrievalOutput.model_validate(payload)
            elif tool_name == TAVILY_WEB_SEARCH_TOOL_NAME:
                web_output = TavilySearchOutput.model_validate(payload)
            elif tool_name == CALCULATOR_TOOL_NAME:
                calculator_output = CalculatorOutput.model_validate(payload)

        if rag_output is not None and web_output is None and calculator_output is None:
            single = observation.model_copy(
                update={
                    "tool_name": RAG_RETRIEVAL_TOOL_NAME,
                    "tool_output": rag_output.model_dump(),
                }
            )
            generated = self._generate_from_rag(request, single)
            return self._with_multi_tool_metadata(generated, observation)
        if web_output is not None and rag_output is None and calculator_output is None:
            single = observation.model_copy(
                update={
                    "tool_name": TAVILY_WEB_SEARCH_TOOL_NAME,
                    "tool_output": web_output.model_dump(),
                }
            )
            generated = self._generate_from_web(request, single)
            return self._with_multi_tool_metadata(generated, observation)
        if calculator_output is not None and rag_output is None and web_output is None:
            single = observation.model_copy(
                update={
                    "tool_name": CALCULATOR_TOOL_NAME,
                    "tool_output": calculator_output.model_dump(),
                }
            )
            generated = self._generate_from_calculator(request, single)
            return self._with_multi_tool_metadata(generated, observation)

        chunks = merge_tool_outputs_to_chunks(
            rag_output=rag_output,
            web_output=web_output,
            calculator_output=calculator_output,
        )
        logger.info(
            "agent_generation_started",
            extra={
                "operation": "agent_run",
                "tool_name": observation.tool_name,
                "result_count": len(chunks),
                "empty": not chunks,
            },
        )
        rag_result = self._rag.generate_from_chunks(request.query, chunks)
        citations = citations_from_rag(rag_result.citations)
        logger.info(
            "agent_generation_completed",
            extra={
                "operation": "agent_run",
                "tool_name": observation.tool_name,
                "citation_count": len(citations),
                "answer_length": len(rag_result.answer),
            },
        )
        metadata = dict(observation.metadata)
        metadata.update(
            {
                "generated": True,
                "citation_count": len(citations),
                "empty_retrieval": not chunks,
            }
        )
        tool_names = observation.tool_names or list(observation.metadata.get("tools_used", []))
        return observation.model_copy(
            update={
                "answer": rag_result.answer,
                "citations": citations,
                "metadata": metadata,
                "tool_name": observation.tool_name,
                "tool_names": tool_names,
            }
        )

    def _with_multi_tool_metadata(
        self,
        generated: AgentObservation,
        source: AgentObservation,
    ) -> AgentObservation:
        metadata = dict(generated.metadata)
        for key in ("multi_tool", "tools_used", "failed_tools", "partial_success", "errors"):
            if key in source.metadata:
                metadata[key] = source.metadata[key]
        return generated.model_copy(
            update={
                "metadata": metadata,
                "tool_name": source.tool_name,
                "tool_names": source.tool_names or list(source.metadata.get("tools_used", [])),
            }
        )

    def _generate_from_rag(
        self,
        request: AgentRequest,
        observation: AgentObservation,
    ) -> AgentObservation:
        if observation.tool_output is None:
            raise AgentError(
                "RAG retrieval succeeded without structured output",
                details={"reason": "missing_tool_output", "tool_name": observation.tool_name},
            )

        retrieval = RAGRetrievalOutput.model_validate(observation.tool_output)
        chunks = [output_to_chunk(chunk) for chunk in retrieval.chunks]
        logger.info(
            "agent_generation_started",
            extra={
                "operation": "agent_run",
                "tool_name": observation.tool_name,
                "result_count": retrieval.result_count,
                "empty": retrieval.empty,
            },
        )
        rag_result = self._rag.generate_from_chunks(request.query, chunks)
        citations = citations_from_rag(rag_result.citations)
        logger.info(
            "agent_generation_completed",
            extra={
                "operation": "agent_run",
                "tool_name": observation.tool_name,
                "citation_count": len(citations),
                "answer_length": len(rag_result.answer),
            },
        )
        metadata = dict(observation.metadata)
        metadata.update(
            {
                "generated": True,
                "citation_count": len(citations),
                "empty_retrieval": retrieval.empty,
            }
        )
        return observation.model_copy(
            update={
                "answer": rag_result.answer,
                "citations": citations,
                "metadata": metadata,
            }
        )

    def _generate_from_web(
        self,
        request: AgentRequest,
        observation: AgentObservation,
    ) -> AgentObservation:
        if observation.tool_output is None:
            raise AgentError(
                "Tavily search succeeded without structured output",
                details={"reason": "missing_tool_output", "tool_name": observation.tool_name},
            )

        search_output = TavilySearchOutput.model_validate(observation.tool_output)
        logger.info(
            "agent_generation_started",
            extra={
                "operation": "agent_run",
                "tool_name": observation.tool_name,
                "result_count": search_output.result_count,
                "empty": search_output.empty,
            },
        )
        answer, citations = self._web_generator.generate(request.query, search_output)
        logger.info(
            "agent_generation_completed",
            extra={
                "operation": "agent_run",
                "tool_name": observation.tool_name,
                "citation_count": len(citations),
                "answer_length": len(answer),
            },
        )
        metadata = dict(observation.metadata)
        metadata.update(
            {
                "generated": True,
                "citation_count": len(citations),
                "empty_retrieval": search_output.empty,
            }
        )
        return observation.model_copy(
            update={
                "answer": answer,
                "citations": citations,
                "metadata": metadata,
            }
        )

    def _generate_from_calculator(
        self,
        request: AgentRequest,
        observation: AgentObservation,
    ) -> AgentObservation:
        if observation.tool_output is None:
            raise AgentError(
                "Calculator succeeded without structured output",
                details={"reason": "missing_tool_output", "tool_name": observation.tool_name},
            )

        calc_output = CalculatorOutput.model_validate(observation.tool_output)
        logger.info(
            "agent_generation_started",
            extra={
                "operation": "agent_run",
                "tool_name": observation.tool_name,
                "expression": calc_output.expression,
            },
        )
        answer = format_calculator_answer(calc_output)
        logger.info(
            "agent_generation_completed",
            extra={
                "operation": "agent_run",
                "tool_name": observation.tool_name,
                "answer_length": len(answer),
            },
        )
        metadata = dict(observation.metadata)
        metadata.update(
            {
                "generated": True,
                "citation_count": 0,
                "empty_retrieval": False,
                "expression": calc_output.expression,
                "result": calc_output.result,
            }
        )
        return observation.model_copy(
            update={
                "answer": answer,
                "citations": [],
                "metadata": metadata,
            }
        )

    def _result_from_finish(
        self,
        action: AgentAction,
        history: list[AgentStep],
    ) -> AgentRunResult:
        last_observation = _last_observation(history)
        citations = last_observation.citations if last_observation else []
        tool_names = _tool_names_from_history(history, last_observation)
        return AgentRunResult(
            answer=action.answer or "",
            citations=citations,
            tool_used=_last_tool_name(history),
            steps=history,
            metadata={
                "step_count": len(history),
                "finished": True,
                "tool_names": tool_names,
            },
        )

    def _result_from_history(self, history: list[AgentStep]) -> AgentRunResult:
        observation = _last_observation(history)
        if observation is None or not observation.success:
            raise AgentError(
                "Agent exceeded the maximum number of steps without finishing",
                details={
                    "reason": "max_steps_exceeded",
                    "max_steps": self._max_steps,
                    "step_count": len(history),
                },
            )
        if observation.answer is None:
            raise AgentError(
                "Agent completed tool execution without generating an answer",
                details={
                    "reason": "missing_generated_answer",
                    "tool_name": observation.tool_name,
                },
            )
        return AgentRunResult(
            answer=observation.answer,
            citations=observation.citations,
            tool_used=observation.tool_name,
            steps=history,
            metadata={
                "step_count": len(history),
                "finished": False,
                "max_steps_reached": True,
                "tool_names": observation.tool_names or list(observation.metadata.get("tools_used", [])),
                **observation.metadata,
            },
        )

    def _log_completed(
        self,
        result: AgentRunResult,
        *,
        step_count: int,
        finished: bool,
    ) -> None:
        logger.info(
            "agent_run_completed",
            extra={
                "operation": "agent_run",
                "tool_used": result.tool_used,
                "step_count": step_count,
                "citation_count": len(result.citations),
                "answer_length": len(result.answer),
                "finished": finished,
            },
        )


def _build_request(
    query: str,
    *,
    top_k: int | None,
    filters: RetrievalFilters | None,
) -> AgentRequest:
    normalized = query.strip()
    if not normalized:
        raise QueryError(
            "Query must not be empty",
            details={"reason": "empty_query"},
        )
    kwargs: dict[str, object] = {"query": normalized, "top_k": top_k}
    if filters is not None and not filters.is_empty():
        if filters.document_ids:
            kwargs["document_ids"] = list(filters.document_ids)
        if filters.filenames:
            kwargs["filenames"] = list(filters.filenames)
        if filters.file_types:
            kwargs["file_types"] = list(filters.file_types)
        if filters.sections:
            kwargs["sections"] = list(filters.sections)
    return AgentRequest.model_validate(kwargs)


def _task_arguments(base_arguments: dict[str, object], task: AgentTask) -> dict[str, object]:
    if task.tool_name == TAVILY_WEB_SEARCH_TOOL_NAME:
        arguments: dict[str, object] = {"query": task.query}
        max_results = base_arguments.get("max_results")
        if max_results is not None:
            arguments["max_results"] = max_results
        return arguments
    arguments = dict(base_arguments)
    arguments["query"] = task.query
    return arguments


def _last_observation(history: list[AgentStep]) -> AgentObservation | None:
    for step in reversed(history):
        if step.observation is not None:
            return step.observation
    return None


def _last_tool_name(history: list[AgentStep]) -> str | None:
    for step in reversed(history):
        if step.action.type is AgentActionType.EXECUTE_PLAN:
            if step.action.tool_names:
                return "+".join(step.action.tool_names)
            return "+".join(task.tool_name for task in step.action.tasks)
        if step.action.type in {AgentActionType.CALL_TOOL, AgentActionType.CALL_TOOLS}:
            if step.action.tool_names:
                return "+".join(step.action.tool_names)
            return step.action.tool_name
    return None


def _tool_names_from_history(
    history: list[AgentStep],
    observation: AgentObservation | None,
) -> list[str]:
    if observation is not None:
        if observation.tool_names:
            return list(observation.tool_names)
        tools_used = observation.metadata.get("tools_used")
        if isinstance(tools_used, list) and tools_used:
            return [str(name) for name in tools_used]
    for step in reversed(history):
        if step.action.type is AgentActionType.EXECUTE_PLAN:
            if step.action.tool_names:
                return list(step.action.tool_names)
            return [task.tool_name for task in step.action.tasks]
        if step.action.type in {AgentActionType.CALL_TOOL, AgentActionType.CALL_TOOLS}:
            if step.action.tool_names:
                return list(step.action.tool_names)
            if step.action.tool_name:
                return [step.action.tool_name]
    return []
