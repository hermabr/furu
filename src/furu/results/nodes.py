from __future__ import annotations

from typing import Literal, NotRequired, TypedDict, cast

from furu.utils import JsonValue

type ManifestValue = JsonValue

FURU_KEY = "$furu"
RESULT_FORMAT = "furu-result/v1"


class ExternalNode(TypedDict):
    kind: Literal["external"]
    serializer: str
    artifact_dir: str
    lazy: bool
    python_type: str
    meta: NotRequired[JsonValue]


class DataclassNode(TypedDict):
    kind: Literal["dataclass"]
    python_type: str
    fields: dict[str, ManifestValue]


class PydanticNode(TypedDict):
    kind: Literal["pydantic"]
    python_type: str
    fields: dict[str, ManifestValue]


class TupleNode(TypedDict):
    kind: Literal["tuple"]
    items: list[ManifestValue]


class SetNode(TypedDict):
    kind: Literal["set"]
    items: list[ManifestValue]


class FrozenSetNode(TypedDict):
    kind: Literal["frozenset"]
    items: list[ManifestValue]


class MappingNode(TypedDict):
    kind: Literal["mapping"]
    items: dict[str, ManifestValue]


class ProtocolNode(TypedDict):
    kind: Literal["protocol"]
    python_type: str
    value: ManifestValue


type TaggedNode = (
    ExternalNode
    | DataclassNode
    | PydanticNode
    | TupleNode
    | SetNode
    | FrozenSetNode
    | MappingNode
    | ProtocolNode
)


class ResultManifest(TypedDict):
    format: Literal["furu-result/v1"]
    root: ManifestValue


def make_result_manifest(root: ManifestValue) -> ResultManifest:
    return ResultManifest(format=RESULT_FORMAT, root=root)


def wrap_node(node: TaggedNode) -> ManifestValue:
    return cast(ManifestValue, {FURU_KEY: cast(JsonValue, node)})


def unwrap_node(node: ManifestValue) -> TaggedNode | None:
    if not isinstance(node, dict) or set(node) != {FURU_KEY}:
        return None
    payload = node[FURU_KEY]
    if not isinstance(payload, dict):
        return None
    kind = payload.get("kind")
    if kind not in {
        "external",
        "dataclass",
        "pydantic",
        "tuple",
        "set",
        "frozenset",
        "mapping",
        "protocol",
    }:
        return None
    return cast(TaggedNode, payload)


def make_external_node(
    *,
    serializer: str,
    artifact_dir: str,
    lazy: bool,
    python_type: str,
    meta: JsonValue | None = None,
) -> ManifestValue:
    payload: ExternalNode = {
        "kind": "external",
        "serializer": serializer,
        "artifact_dir": artifact_dir,
        "lazy": lazy,
        "python_type": python_type,
    }
    if meta is not None:
        payload["meta"] = meta
    return wrap_node(payload)


def make_dataclass_node(
    python_type: str, fields: dict[str, ManifestValue]
) -> ManifestValue:
    return wrap_node(
        {"kind": "dataclass", "python_type": python_type, "fields": fields}
    )


def make_pydantic_node(
    python_type: str, fields: dict[str, ManifestValue]
) -> ManifestValue:
    return wrap_node({"kind": "pydantic", "python_type": python_type, "fields": fields})


def make_tuple_node(items: list[ManifestValue]) -> ManifestValue:
    return wrap_node({"kind": "tuple", "items": items})


def make_set_node(items: list[ManifestValue]) -> ManifestValue:
    return wrap_node({"kind": "set", "items": items})


def make_frozenset_node(items: list[ManifestValue]) -> ManifestValue:
    return wrap_node({"kind": "frozenset", "items": items})


def make_mapping_node(items: dict[str, ManifestValue]) -> ManifestValue:
    return wrap_node({"kind": "mapping", "items": items})


def make_protocol_node(python_type: str, value: ManifestValue) -> ManifestValue:
    return wrap_node({"kind": "protocol", "python_type": python_type, "value": value})
