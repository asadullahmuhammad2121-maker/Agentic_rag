"""Deterministic calculator tool for safe arithmetic evaluation."""

from __future__ import annotations

from pydantic import BaseModel

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.agent.models import CalculatorInput, CalculatorOutput, ToolError, ToolResult
from app.services.agent.tools.base import Tool
from app.services.agent.tools.calculator_engine import (
    CalculatorEvaluationError,
    evaluate_expression,
)
from app.services.agent.tools.expression_parser import query_to_expression

logger = get_logger(__name__)

CALCULATOR_TOOL_NAME = "calculator"


class CalculatorTool(Tool):
    """Evaluate mathematical expressions extracted from user queries."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def name(self) -> str:
        return CALCULATOR_TOOL_NAME

    @property
    def description(self) -> str:
        return (
            "Evaluate arithmetic and percentage calculations from numeric expressions or "
            "calculation-focused queries. Use this for math, averages, and percentage problems."
        )

    @property
    def input_model(self) -> type[BaseModel]:
        return CalculatorInput

    @property
    def output_model(self) -> type[BaseModel]:
        return CalculatorOutput

    def execute(self, validated_input: BaseModel) -> ToolResult:
        payload = CalculatorInput.model_validate(validated_input.model_dump())
        if not self._settings.calculator_enabled:
            return ToolResult(
                success=False,
                error=ToolError(
                    code="calculator_disabled",
                    message="Calculator tool is disabled",
                    details={"reason": "calculator_disabled"},
                ),
            )

        source_text = (payload.expression or payload.query).strip()
        logger.info(
            "calculator_tool_started",
            extra={
                "operation": "calculator_tool",
                "query_length": len(payload.query),
                "has_expression": bool(payload.expression),
            },
        )

        try:
            expression = (
                payload.expression.strip()
                if payload.expression and payload.expression.strip()
                else query_to_expression(payload.query)
            )
            evaluation = evaluate_expression(
                expression,
                max_length=self._settings.calculator_max_expression_length,
            )
        except (CalculatorEvaluationError, ValueError) as exc:
            logger.warning(
                "calculator_tool_failed",
                extra={
                    "operation": "calculator_tool",
                    "reason": str(exc),
                },
            )
            return ToolResult(
                success=False,
                error=ToolError(
                    code="calculator_error",
                    message=str(exc),
                    details={"reason": "invalid_expression"},
                ),
            )

        output = CalculatorOutput(
            query=payload.query,
            expression=evaluation.expression,
            result=evaluation.result,
            source_text=source_text,
        )
        logger.info(
            "calculator_tool_completed",
            extra={
                "operation": "calculator_tool",
                "expression_length": len(output.expression),
            },
        )
        return ToolResult(success=True, output=output)
