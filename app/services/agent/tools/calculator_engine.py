"""Safe arithmetic expression evaluation for the calculator tool."""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from typing import Any

_MAX_EXPONENT = 20
_MAX_ABS_RESULT = 1e18


@dataclass(frozen=True, slots=True)
class CalculatorEvaluation:
    """Result of evaluating a normalized expression."""

    expression: str
    result: float


class CalculatorEvaluationError(ValueError):
    """Raised when an expression cannot be evaluated safely."""


_ALLOWED_BINOPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def evaluate_expression(expression: str, *, max_length: int) -> CalculatorEvaluation:
    """Evaluate a numeric expression using a restricted AST interpreter."""
    normalized = expression.strip()
    if not normalized:
        raise CalculatorEvaluationError("Expression must not be empty")
    if len(normalized) > max_length:
        raise CalculatorEvaluationError("Expression exceeds the maximum allowed length")

    try:
        parsed = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise CalculatorEvaluationError("Invalid expression syntax") from exc

    result = _eval_node(parsed.body)
    if not isinstance(result, (int, float)):
        raise CalculatorEvaluationError("Expression did not evaluate to a number")
    float_result = float(result)
    if abs(float_result - round(float_result)) < 1e-9:
        float_result = float(int(round(float_result)))
    if abs(float_result) > _MAX_ABS_RESULT:
        raise CalculatorEvaluationError("Result exceeds the allowed numeric range")

    return CalculatorEvaluation(expression=normalized, result=float_result)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalculatorEvaluationError("Only numeric literals are allowed")
        return float(node.value)

    if isinstance(node, ast.UnaryOp):
        operator_fn = _ALLOWED_UNARYOPS.get(type(node.op))
        if operator_fn is None:
            raise CalculatorEvaluationError("Unsupported unary operator")
        return float(operator_fn(_eval_node(node.operand)))

    if isinstance(node, ast.BinOp):
        operator_fn = _ALLOWED_BINOPS.get(type(node.op))
        if operator_fn is None:
            raise CalculatorEvaluationError("Unsupported binary operator")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise CalculatorEvaluationError("Exponent is too large")
        if isinstance(node.op, ast.Div) and right == 0:
            raise CalculatorEvaluationError("Division by zero")
        if isinstance(node.op, ast.Mod) and right == 0:
            raise CalculatorEvaluationError("Division by zero")
        return float(operator_fn(left, right))

    raise CalculatorEvaluationError("Unsupported expression construct")
