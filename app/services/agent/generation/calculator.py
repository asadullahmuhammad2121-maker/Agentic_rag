"""Formatting helpers for calculator tool generation."""

from __future__ import annotations

from app.services.agent.models import CalculatorOutput


def format_calculator_answer(output: CalculatorOutput) -> str:
    """Return a concise deterministic answer for calculator-only runs."""
    return f"{output.expression} = {format_result(output.result)}"


def format_calculator_evidence(output: CalculatorOutput) -> str:
    """Return structured calculator evidence for combined generation prompts."""
    return (
        "Calculator Result\n"
        f"Expression: {output.expression}\n"
        f"Result: {format_result(output.result)}"
    )


def format_result(value: float) -> str:
    """Format numeric results without unnecessary trailing zeros."""
    if value.is_integer():
        return f"{int(value):,}"
    return f"{value:,.4f}".rstrip("0").rstrip(".")
