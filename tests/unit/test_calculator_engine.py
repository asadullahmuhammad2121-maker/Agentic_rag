"""Unit tests for the safe calculator expression engine."""

from __future__ import annotations

import pytest

from app.services.agent.tools.calculator_engine import (
    CalculatorEvaluationError,
    evaluate_expression,
)
from app.services.agent.tools.expression_parser import query_to_expression


def test_basic_addition() -> None:
    result = evaluate_expression("2 + 3", max_length=500)
    assert result.result == 5


def test_subtraction() -> None:
    result = evaluate_expression("10 - 4", max_length=500)
    assert result.result == 6


def test_multiplication() -> None:
    result = evaluate_expression("125 * 48", max_length=500)
    assert result.result == 6000


def test_division() -> None:
    result = evaluate_expression("10 / 4", max_length=500)
    assert result.result == 2.5


def test_percentage_query() -> None:
    expression = query_to_expression("What is 17.5% of 84000?")
    result = evaluate_expression(expression, max_length=500)
    assert result.result == 14700


def test_parentheses() -> None:
    expression = query_to_expression("Calculate (125 * 48) + 300.")
    result = evaluate_expression(expression, max_length=500)
    assert result.result == 6300


def test_exponentiation() -> None:
    result = evaluate_expression("2 ** 10", max_length=500)
    assert result.result == 1024


def test_decimal_arithmetic() -> None:
    result = evaluate_expression("1.5 + 2.25", max_length=500)
    assert result.result == pytest.approx(3.75)


def test_invalid_syntax() -> None:
    with pytest.raises(CalculatorEvaluationError, match="Invalid expression syntax"):
        evaluate_expression("2 + * 3", max_length=500)


def test_division_by_zero() -> None:
    with pytest.raises(CalculatorEvaluationError, match="Division by zero"):
        evaluate_expression("10 / 0", max_length=500)


def test_empty_expression() -> None:
    with pytest.raises(CalculatorEvaluationError, match="must not be empty"):
        evaluate_expression("   ", max_length=500)


def test_oversized_expression() -> None:
    with pytest.raises(CalculatorEvaluationError, match="maximum allowed length"):
        evaluate_expression("1 + 1" * 200, max_length=50)


def test_malicious_python_expression() -> None:
    with pytest.raises(CalculatorEvaluationError):
        evaluate_expression('exec("print(1)")', max_length=500)


def test_import_attempt() -> None:
    with pytest.raises(CalculatorEvaluationError):
        evaluate_expression('__import__("os").system("rm")', max_length=500)


def test_attribute_access_attempt() -> None:
    with pytest.raises(CalculatorEvaluationError):
        evaluate_expression("(1).__class__", max_length=500)


def test_unsupported_function_call() -> None:
    with pytest.raises(CalculatorEvaluationError):
        evaluate_expression("sqrt(16)", max_length=500)


def test_deterministic_results() -> None:
    first = evaluate_expression("17.5 / 100 * 84000", max_length=500)
    second = evaluate_expression("17.5 / 100 * 84000", max_length=500)
    assert first.result == second.result == 14700


def test_average_query() -> None:
    expression = query_to_expression("What is the average of 45, 67, 89 and 91?")
    result = evaluate_expression(expression, max_length=500)
    assert result.result == 73
