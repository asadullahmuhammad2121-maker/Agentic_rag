"""Agent abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.services.agent.models import AgentAction, AgentRequest, AgentStep
from app.services.agent.tools.registry import ToolRegistry


class Agent(ABC):
    """Decide the next action for a user query given available tools and history."""

    @abstractmethod
    def decide(
        self,
        request: AgentRequest,
        *,
        tools: ToolRegistry,
        history: Sequence[AgentStep],
    ) -> AgentAction:
        """Return a tool call or a finish action."""
