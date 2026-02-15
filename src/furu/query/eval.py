"""Query AST evaluator."""

from __future__ import annotations

import re
from typing import cast
from typing import TypeGuard

from .ast import (
    AndQuery,
    BetweenQuery,
    ContainsQuery,
    EqQuery,
    ExistsQuery,
    FalseQuery,
    GtQuery,
    GteQuery,
    InQuery,
    IsAQuery,
    LtQuery,
    LteQuery,
    MissingQuery,
    NeQuery,
    NinQuery,
    NotQuery,
    OrQuery,
    Query,
    RegexQuery,
    RelatedToQuery,
    Scalar,
    StartsWithQuery,
    TrueQuery,
    TypeIsQuery,
    EndsWithQuery,
)
from .paths import JSONScalar, JSONValue, PATH_MISSING, get_path
from .types import resolve_type


MAX_QUERY_NODES = 200
MAX_QUERY_DEPTH = 30


def matches(document: dict[str, JSONValue], query: Query) -> bool:
    """Return whether ``document`` satisfies the AST ``query``."""

    return _matches(document, query)


def validate_query(
    query: Query,
    *,
    max_nodes: int = MAX_QUERY_NODES,
    max_depth: int = MAX_QUERY_DEPTH,
) -> None:
    """Validate query complexity limits.

    Raises ``ValueError`` when limits are exceeded.
    """

    node_count, observed_depth = _validate(query, depth=1)
    if node_count > max_nodes:
        raise ValueError(f"query has {node_count} nodes; max allowed is {max_nodes}")
    if observed_depth > max_depth:
        raise ValueError(f"query depth {observed_depth}; max allowed is {max_depth}")


def _matches(document: dict[str, JSONValue], query: Query) -> bool:
    if isinstance(query, TrueQuery):
        return True
    if isinstance(query, FalseQuery):
        return False
    if isinstance(query, AndQuery):
        return all(_matches(document, child) for child in query.args)
    if isinstance(query, OrQuery):
        return any(_matches(document, child) for child in query.args)
    if isinstance(query, NotQuery):
        return not _matches(document, query.arg)
    if isinstance(query, ExistsQuery):
        return get_path(document, query.path) is not PATH_MISSING
    if isinstance(query, MissingQuery):
        return get_path(document, query.path) is PATH_MISSING

    if isinstance(query, EqQuery):
        return _compare(document, query.path, query.value, _equal)
    if isinstance(query, NeQuery):
        return not _compare(document, query.path, query.value, _equal)
    if isinstance(query, LtQuery):
        return _compare(document, query.path, query.value, _less_than)
    if isinstance(query, LteQuery):
        return _compare(document, query.path, query.value, _less_equal)
    if isinstance(query, GtQuery):
        return _compare(document, query.path, query.value, _greater_than)
    if isinstance(query, GteQuery):
        return _compare(document, query.path, query.value, _greater_equal)
    if isinstance(query, BetweenQuery):
        return _between(document, query)
    if isinstance(query, InQuery):
        return _in(document, query.path, query.values)
    if isinstance(query, NinQuery):
        return not _in(document, query.path, query.values)
    if isinstance(query, ContainsQuery):
        return _contains(document, query)
    if isinstance(query, StartsWithQuery):
        return _starts_with(document, query)
    if isinstance(query, EndsWithQuery):
        return _ends_with(document, query)
    if isinstance(query, RegexQuery):
        return _regex(document, query)
    if isinstance(query, TypeIsQuery):
        return _type_is(document, query)
    if isinstance(query, IsAQuery):
        return _is_a(document, query)
    if isinstance(query, RelatedToQuery):
        return _related_to(document, query)

    return False


def _coerce_for_comparison(
    left: object,
    right: Scalar,
) -> tuple[JSONScalar, JSONScalar] | None:
    if not _is_scalar(left):
        return None

    if not _is_scalar(right):
        return None

    if isinstance(left, (int, float)) and isinstance(right, str):
        parsed = _parse_number(right)
        if parsed is None:
            return None
        return left, parsed

    if isinstance(left, str) and isinstance(right, (int, float)):
        parsed = _parse_number(left)
        if parsed is None:
            return None
        return parsed, right

    return left, right


def _compare(
    document: dict[str, JSONValue],
    path: str,
    expected: Scalar,
    comparator,
) -> bool:
    actual = get_path(document, path)
    if actual is PATH_MISSING:
        return False

    coerced = _coerce_for_comparison(actual, expected)
    if coerced is None:
        return False

    left, right = coerced
    return comparator(left, right)


def _in(document: dict[str, JSONValue], path: str, values: list[Scalar]) -> bool:
    actual = get_path(document, path)
    if actual is PATH_MISSING:
        return False
    return any(_equalish(actual, value) for value in values)


def _equalish(left: object, right: Scalar) -> bool:
    coerced = _coerce_for_comparison(left, right)
    if coerced is None:
        return left == right
    return coerced[0] == coerced[1]


def _between(document: dict[str, JSONValue], query: BetweenQuery) -> bool:
    actual = get_path(document, query.path)
    if actual is PATH_MISSING:
        return False

    left = _coerce_for_comparison(actual, query.low)
    if left is None:
        return False
    right = _coerce_for_comparison(actual, query.high)
    if right is None:
        return False

    value, low = left
    _, high = right
    if query.inclusive == "both":
        return _less_equal(low, value) and _less_equal(value, high)
    if query.inclusive == "left":
        return _less_equal(low, value) and _less_than(value, high)
    if query.inclusive == "right":
        return _less_than(low, value) and _less_equal(value, high)
    return _less_than(low, value) and _less_than(value, high)


def _contains(document: dict[str, JSONValue], query: ContainsQuery) -> bool:
    actual = get_path(document, query.path)
    if not isinstance(actual, str):
        return False

    if not isinstance(query.value, str):
        return False

    haystack = actual
    needle = query.value
    if not query.case_sensitive:
        haystack = haystack.lower()
        needle = needle.lower()
    return needle in haystack


def _starts_with(document: dict[str, JSONValue], query: StartsWithQuery) -> bool:
    actual = get_path(document, query.path)
    if not isinstance(actual, str):
        return False

    text = actual
    prefix = query.prefix
    if not query.case_sensitive:
        text = text.lower()
        prefix = prefix.lower()
    return text.startswith(prefix)


def _ends_with(document: dict[str, JSONValue], query: EndsWithQuery) -> bool:
    actual = get_path(document, query.path)
    if not isinstance(actual, str):
        return False

    text = actual
    suffix = query.suffix
    if not query.case_sensitive:
        text = text.lower()
        suffix = suffix.lower()
    return text.endswith(suffix)


def _regex(document: dict[str, JSONValue], query: RegexQuery) -> bool:
    actual = get_path(document, query.path)
    if not isinstance(actual, str):
        return False

    try:
        flags = _parse_regex_flags(query.flags)
        regex = re.compile(query.pattern, flags)
    except re.error:
        return False

    return regex.search(actual) is not None


def _parse_regex_flags(flags: str) -> int:
    if not flags:
        return 0

    segments = [flag.strip() for flag in flags.replace("|", ",").split(",")]
    flag_to_re = {
        "i": re.IGNORECASE,
        "ignorecase": re.IGNORECASE,
        "m": re.MULTILINE,
        "multiline": re.MULTILINE,
        "s": re.DOTALL,
        "dotall": re.DOTALL,
        "x": re.VERBOSE,
        "verbose": re.VERBOSE,
        "a": re.ASCII,
        "ascii": re.ASCII,
        "l": re.LOCALE,
        "locale": re.LOCALE,
    }

    parsed = 0
    for segment in segments:
        if not segment:
            continue
        flag = flag_to_re.get(segment.lower())
        if flag is None:
            continue
        parsed |= flag
    return parsed


def _type_is(document: dict[str, JSONValue], query: TypeIsQuery) -> bool:
    class_name = _extract_class_name(document, query.path)
    if class_name is None:
        return False

    return class_name == query.type


def _is_a(document: dict[str, JSONValue], query: IsAQuery) -> bool:
    class_name = _extract_class_name(document, query.path)
    if class_name is None:
        return False

    candidate = resolve_type(class_name)
    target = resolve_type(query.type)
    if candidate is None or target is None:
        return False

    return issubclass(candidate, target)


def _related_to(document: dict[str, JSONValue], query: RelatedToQuery) -> bool:
    class_name = _extract_class_name(document, query.path)
    if class_name is None:
        return False

    candidate = resolve_type(class_name)
    target = resolve_type(query.type)
    if candidate is None or target is None:
        return False

    return issubclass(candidate, target) or issubclass(target, candidate)


def _extract_class_name(document: dict[str, JSONValue], path: str) -> str | None:
    value = get_path(document, path)
    if isinstance(value, str):
        return value

    if not isinstance(value, dict):
        return None

    dict_value = cast(dict[str, JSONValue], value)
    if "__class__" not in dict_value:
        return None

    maybe_type = dict_value["__class__"]
    if isinstance(maybe_type, str):
        return maybe_type

    return None


def _equal(left: JSONScalar, right: JSONScalar) -> bool:
    return left == right


def _less_than(left: JSONScalar, right: JSONScalar) -> bool:
    if isinstance(left, str) and isinstance(right, str):
        return left < right
    if isinstance(left, bool) and isinstance(right, bool):
        return left < right
    if isinstance(left, int) and isinstance(right, int):
        return left < right
    if isinstance(left, float) and isinstance(right, float):
        return left < right
    if isinstance(left, int) and isinstance(right, float):
        return left < right
    if isinstance(left, float) and isinstance(right, int):
        return left < right
    return False


def _less_equal(left: JSONScalar, right: JSONScalar) -> bool:
    if isinstance(left, str) and isinstance(right, str):
        return left <= right
    if isinstance(left, bool) and isinstance(right, bool):
        return left <= right
    if isinstance(left, int) and isinstance(right, int):
        return left <= right
    if isinstance(left, float) and isinstance(right, float):
        return left <= right
    if isinstance(left, int) and isinstance(right, float):
        return left <= right
    if isinstance(left, float) and isinstance(right, int):
        return left <= right
    return False


def _greater_than(left: JSONScalar, right: JSONScalar) -> bool:
    if isinstance(left, str) and isinstance(right, str):
        return left > right
    if isinstance(left, bool) and isinstance(right, bool):
        return left > right
    if isinstance(left, int) and isinstance(right, int):
        return left > right
    if isinstance(left, float) and isinstance(right, float):
        return left > right
    if isinstance(left, int) and isinstance(right, float):
        return left > right
    if isinstance(left, float) and isinstance(right, int):
        return left > right
    return False


def _greater_equal(left: JSONScalar, right: JSONScalar) -> bool:
    if isinstance(left, str) and isinstance(right, str):
        return left >= right
    if isinstance(left, bool) and isinstance(right, bool):
        return left >= right
    if isinstance(left, int) and isinstance(right, int):
        return left >= right
    if isinstance(left, float) and isinstance(right, float):
        return left >= right
    if isinstance(left, int) and isinstance(right, float):
        return left >= right
    if isinstance(left, float) and isinstance(right, int):
        return left >= right
    return False


def _parse_number(raw: str) -> float | int | None:
    try:
        if "." in raw or "e" in raw or "E" in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return None


def _validate(query: Query, *, depth: int) -> tuple[int, int]:
    if isinstance(query, (AndQuery, OrQuery)):
        child_results = [_validate(child, depth=depth + 1) for child in query.args]
        total = 1 + sum(child_count for child_count, _ in child_results)
        deepest = max((child_depth for _, child_depth in child_results), default=depth)
        return total, deepest

    if isinstance(query, NotQuery):
        child_count, child_depth = _validate(query.arg, depth=depth + 1)
        return child_count + 1, child_depth

    return 1, depth


def _is_scalar(value: object) -> TypeGuard[JSONScalar]:
    return isinstance(value, str | int | float | bool | type(None))


__all__ = [
    "MAX_QUERY_DEPTH",
    "MAX_QUERY_NODES",
    "matches",
    "validate_query",
]
