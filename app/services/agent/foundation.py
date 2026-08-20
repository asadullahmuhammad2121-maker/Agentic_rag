"""Foundation agent: planning, routing, and tool selection."""

from __future__ import annotations

from collections.abc import Sequence

from app.core.config import Settings
from app.core.exceptions import AgentError, AppError
from app.core.logging import get_logger
from app.services.agent.base import Agent
from app.services.agent.models import AgentAction, AgentActionType, AgentRequest, AgentStep
from app.services.agent.planning.fallback import should_attempt_decomposition
from app.services.agent.planning.planner import QueryPlanner
from app.services.agent.recovery.navigation import maybe_document_navigation_recovery
from app.services.agent.routing.router import QueryRouter
from app.services.agent.tools.registry import ToolRegistry

logger = get_logger(__name__)


class FoundationAgent(Agent):
    """
    Decide which tool(s) to run for a user query.

    Phase 3F decomposes hybrid queries into planned sub-tasks. Simple queries
    continue to use Phase 3E routing. Planning failures fall back to routing.
    """

    def __init__(
        self,
        router: QueryRouter,
        planner: QueryPlanner,
        settings: Settings,
    ) -> None:
        self._router = router
        self._planner = planner
        self._settings = settings

    def decide(
        self,
        request: AgentRequest,
        *,
        tools: ToolRegistry,
        history: Sequence[AgentStep],
    ) -> AgentAction:
        if history:
            recovery = maybe_document_navigation_recovery(history, tools=tools)
            if recovery is not None:
                logger.info(
                    "agent_document_navigation_recovery_selected",
                    extra={
                        "operation": "agent_decide",
                        "document_id": recovery.arguments.get("document_id"),
                        "chunk_id": recovery.arguments.get("chunk_id"),
                    },
                )
                return recovery
            return self._finish_from_history(history)

        if should_attempt_decomposition(
            request,
            tools,
            planning_enabled=self._settings.agent_planning_enabled,
        ):
            try:
                plan = self._planner.plan(request, tools)
            except AppError:
                logger.warning(
                    "agent_planning_failed_using_routing",
                    extra={
                        "operation": "agent_decide",
                        "query_length": len(request.query),
                    },
                )
            else:
                if len(plan.tasks) > 1:
                    action = AgentAction(
                        type=AgentActionType.EXECUTE_PLAN,
                        tasks=list(plan.tasks),
                        tool_names=[task.tool_name for task in plan.tasks],
                        arguments=request.tool_arguments(),
                        reasoning=plan.reasoning,
                    )
                    logger.info(
                        "agent_plan_selected",
                        extra={
                            "operation": "agent_decide",
                            "task_count": len(plan.tasks),
                            "tool_names": action.tool_names,
                            "used_fallback": plan.used_fallback,
                            "query_length": len(request.query),
                        },
                    )
                    return action

        decision = self._router.route(request, tools)
        action_type = (
            AgentActionType.CALL_TOOLS
            if len(decision.tool_names) > 1
            else AgentActionType.CALL_TOOL
        )
        action = AgentAction(
            type=action_type,
            tool_name=decision.tool_names[0] if len(decision.tool_names) == 1 else None,
            tool_names=list(decision.tool_names),
            arguments=request.tool_arguments(),
            reasoning=decision.reasoning,
        )
        logger.info(
            "agent_tool_selected",
            extra={
                "operation": "agent_decide",
                "tool_names": decision.tool_names,
                "tool_count": len(decision.tool_names),
                "available_tools": tools.names(),
                "used_fallback": decision.used_fallback,
                "query_length": len(request.query),
            },
        )
        return action

    def _finish_from_history(self, history: Sequence[AgentStep]) -> AgentAction:
        last = history[-1]
        observation = last.observation
        if observation is None:
            raise AgentError(
                "Cannot finish without a tool observation",
                details={"reason": "missing_observation"},
            )
        if not observation.success:
            raise AgentError(
                "Previous tool call failed",
                details={
                    "reason": "tool_failed",
                    "tool_name": observation.tool_name,
                    "error": observation.error,
                },
            )
        return AgentAction(
            type=AgentActionType.FINISH,
            answer=observation.answer or "",
            reasoning="Finishing after successful tool execution and answer generation.",
        )
