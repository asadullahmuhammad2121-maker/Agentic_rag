"""Named registry of agent tools."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from app.core.exceptions import ConfigurationError
from app.core.logging import get_logger
from app.services.agent.tools.base import Tool

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ToolInfo:
    """Public metadata for a registered tool."""

    name: str
    description: str


class ToolRegistry:
    """Lookup table for tools the agent is allowed to call."""

    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or ():
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Register a tool. Duplicate names are configuration errors."""
        name = tool.name.strip()
        if not name:
            raise ConfigurationError(
                "Tool name must not be empty",
                details={"reason": "invalid_tool_name"},
            )
        if name in self._tools:
            raise ConfigurationError(
                f"Duplicate agent tool registered: {name}",
                details={"reason": "duplicate_tool", "tool_name": name},
            )
        self._tools[name] = tool
        logger.info(
            "agent_tool_registered",
            extra={"operation": "register_tool", "tool_name": name},
        )

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def list_tools(self) -> list[ToolInfo]:
        """Return registered tool metadata in registration order."""
        return [
            ToolInfo(name=tool.name, description=tool.description) for tool in self._tools.values()
        ]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools
