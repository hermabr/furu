import enum
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, get_type_hints

from furu._declared_types import (
    child_declared_type,
    has_skip_hash,
    strip_annotated,
)
from furu._pytree import tree_node
from furu.constants import (
    CLASSMARKER,
    FIELDSMARKER,
    KINDMARKER,
    SERIALIZERMARKER,
    VALUEMARKER,
)
from furu.metadata import ArtifactSpec
from furu.serializer.registry import (
    Serializer,
    SerializerMeta,
)
from furu.utils import (
    JsonValue,
    _stable_json_dump,
    fully_qualified_name,
    resolve_fully_qualified_name,
)

if TYPE_CHECKING:
    from furu.core import Spec

_RESERVED_DICT_KEYS = frozenset(
    {CLASSMARKER, FIELDSMARKER, KINDMARKER, SERIALIZERMARKER, VALUEMARKER}
)


def to_json(  # TODO: consider caching this (but if i'm going to, I need to figure out how to cache lists and other unhashable objects)
    obj: Any,
    declared_type: object,
    artifact_serializers: tuple[type[Serializer], ...],
    for_hash: bool = False,
) -> JsonValue:
    # TODO: when writing this to metadata, make sure to escape strings etc

    def assert_correct_dict_key(x: Any) -> str:
        if not isinstance(x, str):
            raise TypeError(
                f"Cannot serialize dict key {x!r} of type {type(x).__name__!r}: "
                "artifact dict keys must be strings"
            )
        if x in _RESERVED_DICT_KEYS:
            raise ValueError(
                f"Cannot serialize dict key {x!r}: "
                "it is reserved by furu artifact serialization"
            )
        return x

    if isinstance(obj, type):
        return {
            KINDMARKER: "type_ref",
            CLASSMARKER: fully_qualified_name(obj),
        }

    if serializer := SerializerMeta.serializer_for_dump(
        obj,
        declared_type=declared_type,
        artifact_serializers=artifact_serializers,
    ):
        return {
            KINDMARKER: "custom",
            SERIALIZERMARKER: serializer._serializer_id(),
            VALUEMARKER: serializer.dump(
                obj,
                declared_type=strip_annotated(declared_type),
            ),
        }

    match obj:
        case None:
            return None
        case int() | str() | float() | bool():
            return obj
        case Path():
            return {KINDMARKER: "path", VALUEMARKER: str(obj)}
        case datetime():
            return {KINDMARKER: "datetime", VALUEMARKER: obj.isoformat()}

    if node := tree_node(obj):
        hints = (
            get_type_hints(type(obj), include_extras=True)
            if node.kind in ("dataclass", "pydantic")
            else {}
        )
        entries_and_children = [
            (entry, child)
            for entry, child in zip(node.entries, node.children, strict=True)
            if not (
                for_hash
                and node.kind in ("dataclass", "pydantic")
                and has_skip_hash(hints.get(cast(str, entry), Any))
            )
        ]
        encoded = [
            (
                entry,
                to_json(
                    child,
                    declared_type=(
                        hints.get(cast(str, entry), Any)
                        if node.kind in ("dataclass", "pydantic")
                        else child_declared_type(declared_type, entry)
                    ),
                    artifact_serializers=artifact_serializers,
                    for_hash=for_hash,
                ),
            )
            for entry, child in entries_and_children
        ]
        match node.kind:
            case "list":
                return [child for _, child in encoded]
            case "tuple":
                return {
                    KINDMARKER: "tuple",
                    VALUEMARKER: [child for _, child in encoded],
                }
            case "set" | "frozenset":
                return {
                    KINDMARKER: node.kind,
                    VALUEMARKER: sorted(
                        (child for _, child in encoded), key=_stable_json_dump
                    ),
                }
            case "dict":
                return {
                    assert_correct_dict_key(key): child
                    for key, child in encoded
                }
            case "dataclass" | "pydantic":
                cls = type(obj)
                return {
                    KINDMARKER: "instance",
                    CLASSMARKER: fully_qualified_name(cls),
                    FIELDSMARKER: {
                        cast(str, name): child
                        for name, child in encoded
                    },
                }

    if isinstance(obj, enum.Enum):
        raise TypeError(
            f"Cannot serialize enum value {obj!r}: enums are not supported "
            "in furu artifacts yet"
        )
    raise TypeError(
        f"Cannot serialize value {obj!r} of type {type(obj).__name__!r} "
        "into a furu artifact; register a Serializer for this type"
    )


def _from_json_field(value: JsonValue, expected_type: Any) -> Any:
    if (
        isinstance(value, dict)
        and value.get(KINDMARKER) == "custom"
        and isinstance(value.get(SERIALIZERMARKER), str)
    ):
        return _load_custom_value(cast(dict[str, Any], value), expected_type)

    if isinstance(value, dict) and value.get(KINDMARKER) == "path":
        return Path(cast(str, value[VALUEMARKER]))

    if isinstance(value, dict) and value.get(KINDMARKER) == "datetime":
        return datetime.fromisoformat(cast(str, value[VALUEMARKER]))

    if isinstance(value, dict) and value.get(KINDMARKER) in (
        "tuple",
        "set",
        "frozenset",
    ):
        items = [
            _from_json_field(item, child_declared_type(expected_type, i))
            for i, item in enumerate(cast(list[JsonValue], value[VALUEMARKER]))
        ]
        match value[KINDMARKER]:
            case "tuple":
                return tuple(items)
            case "set":
                return set(items)
            case _:
                return frozenset(items)

    if isinstance(value, dict) and KINDMARKER in value:
        return _from_json(value)

    if isinstance(value, list):
        return [
            _from_json_field(item, child_declared_type(expected_type, i))
            for i, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return {
            key: _from_json_field(child, child_declared_type(expected_type, key))
            for key, child in value.items()
        }
    return value


def _load_custom_value(value: dict[str, Any], expected_type: object) -> Any:
    serializer_id = value[SERIALIZERMARKER]
    if not isinstance(serializer_id, str):
        raise TypeError(
            f"Expected custom serializer id to be a string: {serializer_id}"
        )
    serializer = resolve_fully_qualified_name(serializer_id)
    if not (isinstance(serializer, type) and issubclass(serializer, Serializer)):
        raise TypeError(f"{serializer_id} is not a Serializer")
    return serializer.load(
        cast(JsonValue, value[VALUEMARKER]),
        declared_type=strip_annotated(expected_type),
    )


def _resolve_serialized_type(class_name: str) -> type:
    value = resolve_fully_qualified_name(class_name)
    if isinstance(value, type):
        return value
    raise TypeError(f"{class_name!r} resolved to a non-type value")


def _from_json(value: JsonValue) -> Any:
    match value:
        case list():
            return [_from_json(item) for item in value]
        case dict():
            class_name = value.get(CLASSMARKER)
            field_values = value.get(FIELDSMARKER)
            if value.get(KINDMARKER) == "custom" and isinstance(
                value.get(SERIALIZERMARKER), str
            ):
                return _load_custom_value(cast(dict[str, Any], value), Any)
            if value.get(KINDMARKER) in (
                "path",
                "datetime",
                "tuple",
                "set",
                "frozenset",
            ):
                return _from_json_field(value, Any)
            if value.get(KINDMARKER) == "type_ref" and isinstance(class_name, str):
                return _resolve_serialized_type(class_name)
            if (
                value.get(KINDMARKER) == "instance"
                and isinstance(class_name, str)
                and isinstance(field_values, dict)
            ):
                cls = _resolve_serialized_type(class_name)
                hints = get_type_hints(cls, include_extras=True)
                converted_fields = {
                    name: _from_json_field(field_value, hints.get(name, Any))
                    for name, field_value in field_values.items()
                }
                return cls(**converted_fields)
            return {key: _from_json(child) for key, child in value.items()}
        case _:
            return value


def _from_artifact[T: Spec](artifact: ArtifactSpec, expected_type: type[T]) -> T:
    artifact_type = _resolve_serialized_type(artifact.fully_qualified_name)
    if not issubclass(artifact_type, expected_type):
        raise TypeError(
            "Artifact described "
            + f"{artifact_type.__module__}.{artifact_type.__qualname__}, "
            + f"expected {expected_type.__module__}.{expected_type.__qualname__}"
        )

    furu_obj = _from_json_field(artifact.artifact_data, artifact_type)
    if not isinstance(furu_obj, expected_type):
        raise TypeError(
            "Artifact described "
            + f"{type(furu_obj).__module__}.{type(furu_obj).__qualname__}, "
            + f"expected {expected_type.__module__}.{expected_type.__qualname__}"
        )
    if artifact.artifact_hash != furu_obj._artifact_hash:
        raise ValueError(
            "Artifact hash did not match loaded object: "
            + f"artifact={artifact.artifact_hash[:5]}, "
            + f"loaded={furu_obj._artifact_hash[:5]}"
        )
    if artifact.schema_hash != furu_obj._artifact_schema_hash:
        raise ValueError(
            "Artifact schema hash did not match loaded object: "
            + f"artifact={artifact.schema_hash[:5]}, "
            + f"loaded={furu_obj._artifact_schema_hash[:5]}"
        )
    return furu_obj
