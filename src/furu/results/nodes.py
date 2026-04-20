from __future__ import annotations

from typing import Any, TypeGuard, cast

from furu.results.errors import ResultDeserializationError
from furu.results.paths import LogicalPath
from furu.utils import JsonValue

BUNDLE_FORMAT = "furu-result-bundle"
BUNDLE_VERSION = 1
FURU_NODE_KEY = "$furu"


def make_furu_node(kind: str, /, **payload: Any) -> dict[str, JsonValue]:
    return {FURU_NODE_KEY: {"kind": kind, **payload}}


def is_furu_node(node: object) -> TypeGuard[dict[str, dict[str, JsonValue]]]:
    if not isinstance(node, dict) or set(node) != {FURU_NODE_KEY}:
        return False
    mapping = cast(dict[str, object], node)
    payload = mapping[FURU_NODE_KEY]
    if not isinstance(payload, dict):
        return False
    payload_mapping = cast(dict[str, object], payload)
    return isinstance(payload_mapping.get("kind"), str)


def get_furu_payload(
    node: object,
    *,
    logical_path: LogicalPath,
) -> dict[str, JsonValue]:
    if not is_furu_node(node):
        raise ResultDeserializationError(
            "Invalid Furu manifest node",
            logical_path=logical_path,
        )
    return node[FURU_NODE_KEY]


def is_external_node(node: JsonValue) -> bool:
    return is_furu_node(node) and node[FURU_NODE_KEY]["kind"] == "external"
