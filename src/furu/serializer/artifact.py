import enum
from dataclasses import fields, is_dataclass
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, get_args

from pydantic import BaseModel as PydanticBaseModel

from furu._declared_types import (
    cached_type_hints,
    child_declared_type,
    item_declared_types,
    strip_annotated,
)
from furu.constants import (
    CLASSMARKER,
    FIELDSMARKER,
    KINDMARKER,
    SERIALIZERMARKER,
    VALUEMARKER,
)
from furu.metadata import ArtifactSpec
from furu.migration.steps import naming, stored_name
from furu.serializer.registry import (
    Serializer,
    SerializerMeta,
    _annotated_serializer,
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
_PLAIN = frozenset({str, int, float, bool, type(None)})


@cache
def _declares_serializer(declared_type: object) -> bool:
    """Whether the type, or a type inside it, names a serializer for its values."""
    if _annotated_serializer(declared_type):
        return True
    declared_type = strip_annotated(declared_type)
    return hasattr(declared_type, "__furu_serializer__") or any(
        map(_declares_serializer, get_args(declared_type))
    )


def _serializer_may_apply(
    declared_type: object, artifact_serializers: tuple[type[Serializer], ...]
) -> bool:
    """Whether a serializer could take a plain value of this type, or one inside it."""
    if artifact_serializers or SerializerMeta._auto_registered_serializers:
        return True
    try:
        return _declares_serializer(declared_type)
    except TypeError:  # the type cannot be a cache key: the full lookup decides
        return True


def hashed_fields(
    obj: Any,
    artifact_serializers: tuple[type[Serializer], ...],
    nested: set[type],
) -> dict[str, JsonValue]:
    """A dataclass's field values as an id sees them.

    Each is under the name it had before any declared change, and an added
    field is left out while it equals its declared default. So a declared
    change gives the same id as before it.
    """
    hints = cached_type_hints(type(obj))
    hashed: dict[str, JsonValue] = {}
    for attribute, name, added in naming(type(obj)).fields:
        declared_type = hints.get(attribute, Any)
        value = to_json(
            getattr(obj, attribute), declared_type, artifact_serializers, nested
        )
        # Compared as text, so that a NaN equals itself.
        if added is None or _stable_json_dump(value) != _stable_json_dump(
            to_json(added.default, declared_type, artifact_serializers, nested)
        ):
            hashed[name] = value
    return hashed


def to_json(  # TODO: consider caching this (but if i'm going to, I need to figure out how to cache lists and other unhashable objects)
    obj: Any,
    declared_type: object,
    artifact_serializers: tuple[type[Serializer], ...],
    nested: set[type] | None = None,
) -> JsonValue:
    """`obj` as JSON.

    Given `nested`, it is the form an id is a hash of: a spec inside `obj` is
    written as its id, and a dataclass as `hashed_fields` names it, with its
    class added to `nested`.
    """
    # TODO: when writing this to metadata, make sure to escape strings etc

    # Most of what a spec holds is plain values and flat lists of them. JSON
    # already holds those, so they skip the lookup when no serializer could apply.
    tp = type(obj)
    if (tp in _PLAIN or tp is list or tp is tuple) and not _serializer_may_apply(
        declared_type, artifact_serializers
    ):
        if tp in _PLAIN:
            return obj
        if _PLAIN.issuperset(map(type, obj)):
            items = list(obj)
            return items if tp is list else {KINDMARKER: "tuple", VALUEMARKER: items}
    if nested is not None and _is_spec(obj):
        return {KINDMARKER: "spec", VALUEMARKER: obj.object_id}

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
        case list() | tuple():
            items: JsonValue = [
                to_json(
                    x,
                    declared_type=item_type,
                    artifact_serializers=artifact_serializers,
                    nested=nested,
                )
                for x, item_type in zip(obj, item_declared_types(declared_type))
            ]
            if isinstance(obj, list):
                return items
            return {KINDMARKER: "tuple", VALUEMARKER: items}
        case set() | frozenset():
            element_type = child_declared_type(declared_type, 0)
            return {
                KINDMARKER: "set" if isinstance(obj, set) else "frozenset",
                VALUEMARKER: sorted(
                    (
                        to_json(
                            x,
                            declared_type=element_type,
                            artifact_serializers=artifact_serializers,
                            nested=nested,
                        )
                        for x in obj
                    ),
                    key=_stable_json_dump,
                ),
            }
        case dict():
            value_type = child_declared_type(declared_type, None)
            return {
                assert_correct_dict_key(k): to_json(
                    v,
                    declared_type=value_type,
                    artifact_serializers=artifact_serializers,
                    nested=nested,
                )
                for k, v in obj.items()
            }
        case x if is_dataclass(x) and nested is not None:
            nested.add(type(x))
            return {
                KINDMARKER: "instance",
                CLASSMARKER: stored_name(type(x)),
                FIELDSMARKER: hashed_fields(x, artifact_serializers, nested),
            }
        case x if is_dataclass(x):
            hints = cached_type_hints(type(x))
            return {
                KINDMARKER: "instance",
                CLASSMARKER: fully_qualified_name(type(x)),
                FIELDSMARKER: {
                    f.name: to_json(
                        getattr(x, f.name),
                        declared_type=hints.get(f.name, Any),
                        artifact_serializers=artifact_serializers,
                    )
                    for f in fields(x)
                },
            }
        case PydanticBaseModel():
            model_cls = type(obj)
            model_fields = model_cls.model_fields
            hints = cached_type_hints(model_cls)
            return {
                KINDMARKER: "instance",
                CLASSMARKER: fully_qualified_name(model_cls),
                FIELDSMARKER: {
                    k: to_json(
                        getattr(obj, k),
                        declared_type=hints.get(k, Any),
                        artifact_serializers=artifact_serializers,
                        nested=nested,
                    )
                    for k in model_fields
                },
            }
        case enum.Enum():
            raise TypeError(
                f"Cannot serialize enum value {obj!r}: enums are not supported "
                "in furu artifacts yet"
            )
        case _:
            raise TypeError(
                f"Cannot serialize value {obj!r} of type {type(obj).__name__!r} "
                "into a furu artifact; register a Serializer for this type"
            )


def _is_spec(obj: object) -> bool:
    from furu.core import Spec

    return isinstance(obj, Spec)


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
                hints = cached_type_hints(cls)
                # Fields the class fills in itself (@furu.dependency) are
                # derived again; _from_artifact's hash check catches a change.
                derived = (
                    {f.name for f in fields(cls) if not f.init}
                    if is_dataclass(cls)
                    else ()
                )
                converted_fields = {
                    name: _from_json_field(field_value, hints.get(name, Any))
                    for name, field_value in field_values.items()
                    if name not in derived
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
            + f"loaded={furu_obj._artifact_hash[:5]}. A @furu.dependency "
            + "that now returns something else than when the spec was stored "
            + "causes this."
        )
    if artifact.class_part != furu_obj._class_part:
        raise ValueError(
            "Artifact class part did not match loaded object: "
            + f"artifact={artifact.class_part[:5]}, "
            + f"loaded={furu_obj._class_part[:5]}"
        )
    return furu_obj
