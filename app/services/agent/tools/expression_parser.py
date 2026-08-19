"""Convert natural-language calculation queries into safe arithmetic expressions."""

from __future__ import annotations

import re

_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")
_AVERAGE_PATTERN = re.compile(
    r"\baverage\s+of\s+(?P<values>.+)$",
    flags=re.IGNORECASE,
)
_PERCENT_OF_PATTERN = re.compile(
    r"(?P<percent>\d+(?:\.\d+)?)\s*%\s*of\s*(?P<base>\d+(?:\.\d+)?)",
    flags=re.IGNORECASE,
)
_LEADING_PHRASES: tuple[str, ...] = (
    "please calculate",
    "please compute",
    "calculate",
    "compute",
    "what is",
    "what's",
    "tell me",
)


def query_to_expression(query: str) -> str:
    """Best-effort conversion from a user query to a calculator expression."""
    text = query.strip().rstrip("?.!")
    if not text:
        raise ValueError("Query must not be empty")

    lowered = text.casefold()
    for phrase in _LEADING_PHRASES:
        if lowered.startswith(phrase):
            text = text[len(phrase) :].strip()
            lowered = text.casefold()
            break

    average_match = _AVERAGE_PATTERN.search(text)
    if average_match:
        values = _extract_numbers(average_match.group("values"))
        if len(values) < 2:
            raise ValueError("Average queries must include at least two numbers")
        joined = " + ".join(_format_number(value) for value in values)
        return f"({joined}) / {len(values)}"

    if percent_match := _PERCENT_OF_PATTERN.search(text):
        if _PERCENT_OF_PATTERN.search(text[percent_match.end() :]) or re.search(
            r"\b(minus|plus|times|divided by)\b",
            text[percent_match.end() :],
            flags=re.IGNORECASE,
        ):
            return _normalize_percent_expression(text)
        return (
            f"({percent_match.group('percent')} / 100 * {percent_match.group('base')})"
        )

    math_like = _extract_math_expression(text)
    if math_like:
        return _normalize_math_tokens(math_like)

    raise ValueError("Could not derive a calculator expression from the query")


def _normalize_percent_expression(text: str) -> str:
    expression = text
    while True:
        match = _PERCENT_OF_PATTERN.search(expression)
        if match is None:
            break
        replacement = (
            f"({match.group('percent')} / 100 * {match.group('base')})"
        )
        expression = (
            expression[: match.start()] + replacement + expression[match.end() :]
        )

    expression = _normalize_math_tokens(expression)
    expression = re.sub(r"\bminus\b", "-", expression, flags=re.IGNORECASE)
    expression = re.sub(r"\bplus\b", "+", expression, flags=re.IGNORECASE)
    expression = re.sub(r"\btimes\b", "*", expression, flags=re.IGNORECASE)
    expression = re.sub(r"\bmultiplied by\b", "*", expression, flags=re.IGNORECASE)
    expression = re.sub(r"\bdivided by\b", "/", expression, flags=re.IGNORECASE)
    math_like = _extract_math_expression(expression)
    if math_like:
        return _normalize_math_tokens(math_like)
    expression = re.sub(r"\s+", "", expression)
    return expression


def _extract_math_expression(text: str) -> str | None:
    matches = re.findall(r"([\d\s+\-*/().,%×x÷]+)", text)
    candidates = [match.strip() for match in matches if _NUMBER_PATTERN.search(match)]
    if not candidates:
        return None
    return str(max(candidates, key=len))


def _normalize_math_tokens(expression: str) -> str:
    normalized = expression.replace(",", "")
    normalized = normalized.replace("×", "*").replace("x", "*").replace("X", "*")
    normalized = normalized.replace("÷", "/")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.replace("%", "/100*")
    return normalized


def _extract_numbers(text: str) -> list[float]:
    values = [_parse_number(match.group(0)) for match in _NUMBER_PATTERN.finditer(text)]
    if not values:
        raise ValueError("No numeric values found")
    return values


def _parse_number(value: str) -> float:
    return float(value.replace(",", ""))


def _format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return str(value)
