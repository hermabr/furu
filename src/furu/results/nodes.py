from __future__ import annotations

from typing import TypeGuard, cast

from furu.utils import JsonValue

RESULT_FORMAT = "furu-result/v1"
FURU_NODE_KEY = "$furu"


def wrap_node(payload: dict[str, JsonValue]) -> JsonValue:
    return {FURU_NODE_KEY: payload}


def is_wrapped_node(value: JsonValue) -> TypeGuard[dict[str, JsonValue]]:
    if not isinstance(value, dict):
        return False
    if tuple(value.keys()) != (FURU_NODE_KEY,):
        return False
    inner = value[FURU_NODE_KEY]
    return isinstance(inner, dict)


def unwrap_node(value: JsonValue) -> dict[str, JsonValue]:
    if not is_wrapped_node(value):
        raise TypeError("Expected a wrapped Furu result node")
    return cast(dict[str, JsonValue], value[FURU_NODE_KEY])
