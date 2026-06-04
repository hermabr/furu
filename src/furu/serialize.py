import enum
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, get_args, get_origin, get_type_hints

from pydantic import BaseModel as PydanticBaseModel

from furu._declared_types import child_declared_type, strip_annotated
from furu.constants import (
    CLASSMARKER,
    FIELDSMARKER,
    KINDMARKER,
    SERIALIZERMARKER,
    VALUEMARKER,
)
from furu.metadata import ArtifactSpec
from furu.serializer import (
    ArtifactSerializer,
    ArtifactSerializerRegistry,
)
from furu.utils import JsonValue, fully_qualified_name, resolve_fully_qualified_name

if TYPE_CHECKING:
    from furu.core import Furu

_RESERVED_DICT_KEYS = frozenset(
    {CLASSMARKER, FIELDSMARKER, KINDMARKER, SERIALIZERMARKER, VALUEMARKER}
)


def to_json(  # TODO: consider caching this (but if i'm going to, I need to figure out how to cache lists and other unhashable objects)
    obj: Any,
    declared_type: object,
    registry: ArtifactSerializerRegistry,
) -> JsonValue:
    # TODO: when writing this to metadata, make sure to escape strings etc

    def assert_correct_dict_key(x: Any) -> str:
        if not isinstance(x, str):
            raise ValueError("TODO")
        if x in _RESERVED_DICT_KEYS:
            raise ValueError("TODO: write error msg")
        return x

    if isinstance(obj, type):
        return {
            KINDMARKER: "type_ref",
            CLASSMARKER: fully_qualified_name(obj),
        }

    if serializer := registry.for_dump(obj, declared_type=declared_type):
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
            return str(obj)
        case list() | tuple():
            return [
                to_json(
                    x,
                    declared_type=child_declared_type(declared_type, i),
                    registry=registry,
                )
                for i, x in enumerate(obj)
            ]
        case set() | frozenset():
            return sorted(
                [
                    to_json(
                        x,
                        declared_type=child_declared_type(declared_type, i),
                        registry=registry,
                    )
                    for i, x in enumerate(obj)
                ],
                key=repr,
            )
        case dict():
            return {
                assert_correct_dict_key(k): to_json(
                    v,
                    declared_type=child_declared_type(declared_type, k),
                    registry=registry,
                )
                for k, v in obj.items()
            }
        case x if is_dataclass(x):
            hints = get_type_hints(type(x), include_extras=True)
            return {
                KINDMARKER: "instance",
                CLASSMARKER: fully_qualified_name(type(x)),
                FIELDSMARKER: {
                    f.name: to_json(
                        getattr(x, f.name),
                        declared_type=hints.get(f.name, Any),
                        registry=registry,
                    )
                    for f in fields(x)
                },
            }
        case PydanticBaseModel():
            hints = get_type_hints(type(obj), include_extras=True)
            return {
                KINDMARKER: "instance",
                CLASSMARKER: fully_qualified_name(type(obj)),
                FIELDSMARKER: {
                    k: to_json(
                        getattr(obj, k),
                        declared_type=hints.get(k, Any),
                        registry=registry,
                    )
                    for k in type(obj).model_fields
                },
            }
        case enum.Enum():
            raise NotImplementedError("TODO")  #  return {"__enum__": _type_fqn(obj)}
        case _:
            raise ValueError("unexpected item", obj)  # TODO: explain the error more


def _from_json_field(value: JsonValue, expected_type: Any) -> Any:
    if (
        isinstance(value, dict)
        and value.get(KINDMARKER) == "custom"
        and isinstance(value.get(SERIALIZERMARKER), str)
    ):
        return _load_custom_value(cast(dict[str, Any], value), expected_type)

    if isinstance(value, dict) and KINDMARKER in value:
        return _from_json(value)

    expected_type = strip_annotated(expected_type)
    origin = get_origin(expected_type)
    args = get_args(expected_type)
    if isinstance(value, list):
        item_type = args[0] if args and args[0] is not Ellipsis else Any
        items = [_from_json_field(item, item_type) for item in value]
        return tuple(items) if origin is tuple else items
    if isinstance(value, dict):
        value_type = args[1] if origin is dict and len(args) == 2 else Any
        return {
            key: _from_json_field(child, value_type) for key, child in value.items()
        }
    if expected_type is Path and isinstance(value, str):
        return Path(value)
    return value


def _load_custom_value(value: dict[str, Any], expected_type: object) -> Any:
    serializer_id = value[SERIALIZERMARKER]
    if not isinstance(serializer_id, str):
        raise TypeError(
            f"Expected custom serializer id to be a string: {serializer_id}"
        )
    serializer = resolve_fully_qualified_name(serializer_id)
    if not (
        isinstance(serializer, type) and issubclass(serializer, ArtifactSerializer)
    ):
        raise TypeError(f"{serializer_id} is not an ArtifactSerializer")
    return serializer.load(
        cast(JsonValue, value[VALUEMARKER]),
        declared_type=strip_annotated(expected_type),
    )


def _resolve_serialized_type(class_name: str) -> type:
    value = resolve_fully_qualified_name(class_name)
    if isinstance(value, type):
        return value

    as_furu = getattr(value, "as_furu", None)
    if isinstance(as_furu, type):
        from furu.core import Furu

        if issubclass(as_furu, Furu):
            return as_furu

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


def _from_artifact[T: "Furu"](artifact: ArtifactSpec, expected_type: type[T]) -> T:
    load_expected_type: type[Any] = expected_type
    try:
        artifact_type = _resolve_serialized_type(artifact.fully_qualified_name)
    except Exception:
        artifact_type = expected_type
    if issubclass(artifact_type, expected_type):
        load_expected_type = artifact_type

    furu_obj = _from_json_field(artifact.artifact_data, load_expected_type)
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
