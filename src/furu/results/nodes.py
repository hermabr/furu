from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from furu.utils import JsonValue

WRAPPER_KEY = "$furu"
RESULT_FORMAT = "furu-result/v1"

type JsonObject = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class ExternalNode:
    codec: str
    path: str
    lazy: bool
    meta: JsonObject | None = None
    type_id: str | None = None


@dataclass(frozen=True, slots=True)
class DataclassNode:
    type_id: str
    fields: dict[str, ManifestNode]


@dataclass(frozen=True, slots=True)
class PydanticNode:
    type_id: str
    fields: dict[str, ManifestNode]


@dataclass(frozen=True, slots=True)
class TupleNode:
    items: tuple[ManifestNode, ...]


@dataclass(frozen=True, slots=True)
class SetNode:
    items: tuple[ManifestNode, ...]
    frozen: bool = False


@dataclass(frozen=True, slots=True)
class MappingItem:
    key: ManifestNode
    value: ManifestNode


@dataclass(frozen=True, slots=True)
class MappingNode:
    items: tuple[MappingItem, ...]


type ManifestNode = (
    None
    | bool
    | int
    | float
    | str
    | list[ManifestNode]
    | dict[str, ManifestNode]
    | ExternalNode
    | DataclassNode
    | PydanticNode
    | TupleNode
    | SetNode
    | MappingNode
)


def manifest_to_json(node: ManifestNode) -> JsonValue:
    if node is None or isinstance(node, bool | int | float | str):
        return node
    if isinstance(node, list):
        return [manifest_to_json(item) for item in node]
    if isinstance(node, dict):
        return {key: manifest_to_json(value) for key, value in node.items()}
    if isinstance(node, ExternalNode):
        payload: JsonObject = {
            "kind": "external",
            "codec": node.codec,
            "path": node.path,
            "lazy": node.lazy,
        }
        if node.meta is not None:
            payload["meta"] = node.meta
        if node.type_id is not None:
            payload["type"] = node.type_id
        return {WRAPPER_KEY: payload}
    if isinstance(node, DataclassNode):
        return {
            WRAPPER_KEY: {
                "kind": "dataclass",
                "type": node.type_id,
                "fields": {
                    key: manifest_to_json(value) for key, value in node.fields.items()
                },
            }
        }
    if isinstance(node, PydanticNode):
        return {
            WRAPPER_KEY: {
                "kind": "pydantic",
                "type": node.type_id,
                "fields": {
                    key: manifest_to_json(value) for key, value in node.fields.items()
                },
            }
        }
    if isinstance(node, TupleNode):
        return {
            WRAPPER_KEY: {
                "kind": "tuple",
                "items": [manifest_to_json(item) for item in node.items],
            }
        }
    if isinstance(node, SetNode):
        payload: JsonObject = {
            "kind": "set",
            "items": [manifest_to_json(item) for item in node.items],
        }
        if node.frozen:
            payload["frozen"] = True
        return {WRAPPER_KEY: payload}
    if isinstance(node, MappingNode):
        return {
            WRAPPER_KEY: {
                "kind": "mapping",
                "items": [
                    {
                        "key": manifest_to_json(item.key),
                        "value": manifest_to_json(item.value),
                    }
                    for item in node.items
                ],
            }
        }
    raise TypeError(f"unexpected manifest node type: {type(node).__name__}")


def manifest_from_json(value: object) -> ManifestNode:
    if value is None or isinstance(value, bool | int | float | str):
        return cast(ManifestNode, value)
    if isinstance(value, list):
        return [manifest_from_json(item) for item in value]
    if isinstance(value, dict):
        obj = _require_object(value, "manifest object")
        if WRAPPER_KEY in obj:
            if set(obj) != {WRAPPER_KEY}:
                raise ValueError(
                    "wrapped manifest nodes must only contain the '$furu' key"
                )
            payload = _require_object(obj[WRAPPER_KEY], "wrapped manifest payload")
            kind = _require_string(payload.get("kind"), "wrapped manifest kind")
            return _wrapped_node_from_json(kind, payload)
        return {
            _require_string(key, "manifest object key"): manifest_from_json(item)
            for key, item in obj.items()
        }
    raise ValueError(f"unsupported manifest JSON value: {value!r}")


def manifest_document(root: ManifestNode) -> JsonObject:
    return {
        "format": RESULT_FORMAT,
        "root": manifest_to_json(root),
    }


def manifest_root_from_document(document: object) -> ManifestNode:
    payload = _require_object(document, "manifest document")
    format_name = _require_string(payload.get("format"), "manifest format")
    if format_name != RESULT_FORMAT:
        raise ValueError(
            f"unsupported result manifest format {format_name!r}; expected {RESULT_FORMAT!r}"
        )
    if "root" not in payload:
        raise ValueError("manifest document is missing the 'root' field")
    return manifest_from_json(payload["root"])


def _wrapped_node_from_json(kind: str, payload: dict[str, object]) -> ManifestNode:
    if kind == "external":
        meta_value = payload.get("meta")
        meta: JsonObject | None
        if meta_value is None:
            meta = None
        else:
            meta = _require_json_object(meta_value, "external node meta")
        type_value = payload.get("type")
        type_id = (
            None
            if type_value is None
            else _require_string(type_value, "external node type")
        )
        return ExternalNode(
            codec=_require_string(payload.get("codec"), "external node codec"),
            path=_require_string(payload.get("path"), "external node path"),
            lazy=_require_bool(payload.get("lazy"), "external node lazy"),
            meta=meta,
            type_id=type_id,
        )
    if kind == "dataclass":
        return DataclassNode(
            type_id=_require_string(payload.get("type"), "dataclass node type"),
            fields=_manifest_fields(payload.get("fields"), "dataclass node fields"),
        )
    if kind == "pydantic":
        return PydanticNode(
            type_id=_require_string(payload.get("type"), "pydantic node type"),
            fields=_manifest_fields(payload.get("fields"), "pydantic node fields"),
        )
    if kind == "tuple":
        items = _require_list(payload.get("items"), "tuple node items")
        return TupleNode(tuple(manifest_from_json(item) for item in items))
    if kind == "set":
        items = _require_list(payload.get("items"), "set node items")
        frozen_value = payload.get("frozen", False)
        return SetNode(
            items=tuple(manifest_from_json(item) for item in items),
            frozen=_require_bool(frozen_value, "set node frozen"),
        )
    if kind == "mapping":
        items = _require_list(payload.get("items"), "mapping node items")
        return MappingNode(tuple(_mapping_item_from_json(item) for item in items))
    raise ValueError(f"unsupported wrapped manifest node kind {kind!r}")


def _manifest_fields(value: object, label: str) -> dict[str, ManifestNode]:
    obj = _require_object(value, label)
    return {
        _require_string(key, f"{label} key"): manifest_from_json(item)
        for key, item in obj.items()
    }


def _mapping_item_from_json(value: object) -> MappingItem:
    entry = _require_object(value, "mapping entry")
    if "key" not in entry or "value" not in entry:
        raise ValueError("mapping entries must contain both 'key' and 'value'")
    return MappingItem(
        key=manifest_from_json(entry["key"]),
        value=manifest_from_json(entry["value"]),
    )


def _require_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _require_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return cast(list[object], value)


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _require_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _require_json_object(value: object, label: str) -> JsonObject:
    obj = _require_object(value, label)
    return cast(JsonObject, obj)
