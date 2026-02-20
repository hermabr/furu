from __future__ import annotations

import re
from typing import TypeGuard, cast

from .ast import (
    AndNode,
    BetweenNode,
    ContainsNode,
    EndswithNode,
    EqNode,
    ExistsNode,
    FalseNode,
    GtNode,
    GteNode,
    InNode,
    IsANode,
    LtNode,
    LteNode,
    MissingNode,
    NeNode,
    NinNode,
    NotNode,
    OrNode,
    Query,
    RegexNode,
    RelatedToNode,
    Scalar,
    StartswithNode,
    TrueNode,
    TypeIsNode,
)
from .paths import PATH_MISSING, JsonValue, get_path
from .regex import compile_regex
from .types import resolve_type

_NUMBER_TYPES = (int, float)
_INT_PATTERN = re.compile(r"^[+-]?\d+$")
_FLOAT_PATTERN = re.compile(
    r"^[+-]?(?:\d+\.\d*|\d*\.\d+|\d+[eE][+-]?\d+|\d+\.\d*[eE][+-]?\d+|\d*\.\d+[eE][+-]?\d+)$"
)


def _is_number(value: Scalar) -> TypeGuard[int | float]:
    return isinstance(value, _NUMBER_TYPES) and not isinstance(value, bool)


def _is_scalar(value: JsonValue) -> TypeGuard[Scalar]:
    return isinstance(value, (str, int, float, bool)) or value is None


def _parse_numeric_string(value: str) -> int | float | None:
    stripped = value.strip()
    if stripped == "":
        return None

    if _INT_PATTERN.fullmatch(stripped) is not None:
        return int(stripped)

    if _FLOAT_PATTERN.fullmatch(stripped) is not None:
        return float(stripped)

    return None


def _coerce_expected(actual: Scalar, expected: Scalar) -> Scalar:
    if _is_number(actual) and isinstance(expected, str):
        parsed = _parse_numeric_string(expected)
        if parsed is not None:
            return parsed
    return expected


def _eq(actual: Scalar, expected: Scalar) -> bool:
    return actual == _coerce_expected(actual, expected)


def _lt(actual: Scalar, expected: Scalar) -> bool:
    coerced_expected = _coerce_expected(actual, expected)
    if _is_number(actual) and _is_number(coerced_expected):
        return actual < coerced_expected
    if isinstance(actual, str) and isinstance(coerced_expected, str):
        return actual < coerced_expected
    return False


def _lte(actual: Scalar, expected: Scalar) -> bool:
    coerced_expected = _coerce_expected(actual, expected)
    if _is_number(actual) and _is_number(coerced_expected):
        return actual <= coerced_expected
    if isinstance(actual, str) and isinstance(coerced_expected, str):
        return actual <= coerced_expected
    return False


def _gt(actual: Scalar, expected: Scalar) -> bool:
    coerced_expected = _coerce_expected(actual, expected)
    if _is_number(actual) and _is_number(coerced_expected):
        return actual > coerced_expected
    if isinstance(actual, str) and isinstance(coerced_expected, str):
        return actual > coerced_expected
    return False


def _gte(actual: Scalar, expected: Scalar) -> bool:
    coerced_expected = _coerce_expected(actual, expected)
    if _is_number(actual) and _is_number(coerced_expected):
        return actual >= coerced_expected
    if isinstance(actual, str) and isinstance(coerced_expected, str):
        return actual >= coerced_expected
    return False


def _between(
    actual: Scalar,
    low: Scalar,
    high: Scalar,
    inclusive: str,
) -> bool:
    coerced_low = _coerce_expected(actual, low)
    coerced_high = _coerce_expected(actual, high)
    if _is_number(actual) and _is_number(coerced_low) and _is_number(coerced_high):
        if inclusive == "both":
            return actual >= coerced_low and actual <= coerced_high
        if inclusive == "left":
            return actual >= coerced_low and actual < coerced_high
        if inclusive == "right":
            return actual > coerced_low and actual <= coerced_high
        return actual > coerced_low and actual < coerced_high

    if (
        isinstance(actual, str)
        and isinstance(coerced_low, str)
        and isinstance(coerced_high, str)
    ):
        if inclusive == "both":
            return actual >= coerced_low and actual <= coerced_high
        if inclusive == "left":
            return actual >= coerced_low and actual < coerced_high
        if inclusive == "right":
            return actual > coerced_low and actual <= coerced_high
        return actual > coerced_low and actual < coerced_high

    return False


def _extract_candidate_type_name(actual: JsonValue) -> str | None:
    if not isinstance(actual, dict):
        return None
    candidate = actual.get("__class__")
    if isinstance(candidate, str):
        return candidate
    return None


def _type_is(actual: JsonValue, base_type: str) -> bool:
    candidate_type = _extract_candidate_type_name(actual)
    if candidate_type is None:
        return False
    return candidate_type == base_type


def _is_a(actual: JsonValue, base_type: str) -> bool:
    candidate_type = _extract_candidate_type_name(actual)
    if candidate_type is None:
        return False
    candidate_cls = resolve_type(candidate_type)
    base_cls = resolve_type(base_type)
    if candidate_cls is None or base_cls is None:
        return False
    return issubclass(candidate_cls, base_cls)


def _related_to(actual: JsonValue, base_type: str) -> bool:
    candidate_type = _extract_candidate_type_name(actual)
    if candidate_type is None:
        return False
    candidate_cls = resolve_type(candidate_type)
    base_cls = resolve_type(base_type)
    if candidate_cls is None or base_cls is None:
        return False
    return issubclass(candidate_cls, base_cls) or issubclass(base_cls, candidate_cls)


def matches(doc: dict[str, JsonValue], query: Query) -> bool:
    if isinstance(query, TrueNode):
        return True
    if isinstance(query, FalseNode):
        return False

    if isinstance(query, AndNode):
        return all(matches(doc, node) for node in query.args)
    if isinstance(query, OrNode):
        return any(matches(doc, node) for node in query.args)
    if isinstance(query, NotNode):
        return not matches(doc, query.arg)

    if isinstance(query, ExistsNode):
        return get_path(doc, query.path) is not PATH_MISSING
    if isinstance(query, MissingNode):
        return get_path(doc, query.path) is PATH_MISSING

    actual_value = get_path(doc, query.path)
    if actual_value is PATH_MISSING:
        return False
    actual = cast(JsonValue, actual_value)

    if isinstance(query, TypeIsNode):
        return _type_is(actual, query.type)
    if isinstance(query, IsANode):
        return _is_a(actual, query.type)
    if isinstance(query, RelatedToNode):
        return _related_to(actual, query.type)

    if not _is_scalar(actual):
        return False

    if isinstance(query, EqNode):
        return _eq(actual, query.value)
    if isinstance(query, NeNode):
        return not _eq(actual, query.value)
    if isinstance(query, LtNode):
        return _lt(actual, query.value)
    if isinstance(query, LteNode):
        return _lte(actual, query.value)
    if isinstance(query, GtNode):
        return _gt(actual, query.value)
    if isinstance(query, GteNode):
        return _gte(actual, query.value)

    if isinstance(query, BetweenNode):
        return _between(actual, query.low, query.high, query.inclusive)
    if isinstance(query, InNode):
        return any(_eq(actual, value) for value in query.values)
    if isinstance(query, NinNode):
        return all(not _eq(actual, value) for value in query.values)

    if isinstance(query, ContainsNode):
        if not isinstance(actual, str):
            return False
        needle = str(query.value)
        if query.case_sensitive:
            return needle in actual
        return needle.lower() in actual.lower()

    if isinstance(query, StartswithNode):
        if not isinstance(actual, str):
            return False
        if query.case_sensitive:
            return actual.startswith(query.prefix)
        return actual.lower().startswith(query.prefix.lower())

    if isinstance(query, EndswithNode):
        if not isinstance(actual, str):
            return False
        if query.case_sensitive:
            return actual.endswith(query.suffix)
        return actual.lower().endswith(query.suffix.lower())

    if isinstance(query, RegexNode):
        if not isinstance(actual, str):
            return False
        pattern = compile_regex(
            pattern=query.pattern,
            flags=query.flags,
            path=query.path,
        )
        return pattern.search(actual) is not None

    return False
