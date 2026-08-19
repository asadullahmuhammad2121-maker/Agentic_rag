"""Unit tests for the calculator agent tool."""

from __future__ import annotations

from app.services.agent.models import CalculatorOutput
from app.services.agent.tools.calculator import CALCULATOR_TOOL_NAME, CalculatorTool
from app.services.agent.tools.converters import tool_result_to_observation
from app.services.agent.tools.registry import ToolRegistry
from tests.conftest import make_settings


def test_calculator_tool_registered_in_registry() -> None:
    settings = make_settings(calculator_enabled=True)
    tool = CalculatorTool(settings)
    registry = ToolRegistry([tool])
    assert CALCULATOR_TOOL_NAME in registry
    assert registry.names() == [CALCULATOR_TOOL_NAME]


def test_calculator_tool_evaluates_percent_query() -> None:
    tool = CalculatorTool(make_settings(calculator_enabled=True))
    result = tool.run({"query": "What is 17.5% of 84000?"})

    assert result.success is True
    assert isinstance(result.output, CalculatorOutput)
    assert result.output.result == 14700


def test_calculator_tool_accepts_explicit_expression() -> None:
    tool = CalculatorTool(make_settings(calculator_enabled=True))
    result = tool.run({"query": "ignored", "expression": "125 * 48 + 300"})

    assert result.success is True
    assert result.output is not None
    assert result.output.result == 6300


def test_calculator_tool_rejects_malicious_expression() -> None:
    tool = CalculatorTool(make_settings(calculator_enabled=True))
    result = tool.run({"query": "x", "expression": '__import__("os")'})

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "calculator_error"


def test_calculator_tool_disabled() -> None:
    tool = CalculatorTool(make_settings(calculator_enabled=False))
    result = tool.run({"query": "2 + 2"})

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "calculator_disabled"


def test_calculator_observation_includes_expression_metadata() -> None:
    tool = CalculatorTool(make_settings(calculator_enabled=True))
    result = tool.run({"query": "What is 25 * 48?"})

    observation = tool_result_to_observation(CALCULATOR_TOOL_NAME, result)

    assert observation.success is True
    assert observation.metadata["expression"] == "25*48"
    assert observation.metadata["result"] == 1200
