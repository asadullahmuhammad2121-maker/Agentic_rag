"""Generic tool interface used by the agent orchestrator."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ValidationError

from app.core.exceptions import AppError, QueryError
from app.core.logging import get_logger
from app.services.agent.models import ToolError, ToolResult

logger = get_logger(__name__)


class Tool(ABC):
    """A named capability with structured input and output."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable tool identifier used in actions."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description used when selecting among tools."""

    @property
    @abstractmethod
    def input_model(self) -> type[BaseModel]:
        """Pydantic model describing accepted tool arguments."""

    @property
    @abstractmethod
    def output_model(self) -> type[BaseModel]:
        """Pydantic model describing successful tool output."""

    @abstractmethod
    def execute(self, validated_input: BaseModel) -> ToolResult:
        """Run the tool with validated input and return a structured result."""

    def run(self, arguments: Mapping[str, Any]) -> ToolResult:
        """
        Validate arguments, execute the tool, and return a structured result.

        Invalid input raises ``QueryError``. Provider and infrastructure errors
        from ``AppError`` subclasses propagate unchanged.
        """
        try:
            validated = self.input_model.model_validate(dict(arguments))
        except ValidationError as exc:
            raise QueryError(
                f"Invalid input for tool '{self.name}'",
                details={"reason": "invalid_tool_input", "tool_name": self.name},
            ) from exc

        logger.info(
            "tool_execution_started",
            extra={"operation": "tool_run", "tool_name": self.name},
        )
        try:
            result = self.execute(validated)
        except AppError:
            raise
        except Exception as exc:
            logger.exception(
                "tool_execution_failed",
                extra={
                    "operation": "tool_run",
                    "tool_name": self.name,
                    "error_type": type(exc).__name__,
                },
            )
            return ToolResult(
                success=False,
                error=ToolError(
                    code="tool_execution_error",
                    message=f"Tool '{self.name}' failed",
                    details={"error_type": type(exc).__name__},
                ),
            )

        logger.info(
            "tool_execution_completed",
            extra={
                "operation": "tool_run",
                "tool_name": self.name,
                "success": result.success,
            },
        )
        return result
