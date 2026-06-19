import enum
from dataclasses import MISSING, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, get_type_hints

from pydantic import BaseModel as PydanticBaseModel

from furu._fields import (
    dataclass_field_specs,
    pydantic_field_specs,
    split_skiphash,
    strip_skiphash,
)
from furu._declared_types import child_declared_type, strip_annotated
from furu.constants import (
    CLASSMARKER,
    FIELDSMARKER,
    KINDMARKER,
    SERIALIZERMARKER,
    VALUEMARKER,
)
from furu.metadata import ArtifactSpec
from furu.serializer.registry import (
    ArtifactSerializer,
    ArtifactSerializerMeta,
)
from furu.utils import (
    JsonValue,
    _stable_json_dump,
    fully_qualified_name,
    resolve_fully_qualified_name,
)

if TYPE_CHECKING:
    from furu.core import Furu

_RESERVED_DICT_KEYS = frozenset(
    {CLASSMARKER, FIELDSMARKER, KINDMARKER, SERIALIZERMARKER, VALUEMARKER}
)


def to_json(  # TODO: consider caching this (but if i'm going to, I need to figure out how to cache lists and other unhashable objects)
    obj: Any,
    declared_type: object,
    artifact_serializers: tuple[type[ArtifactSerializer], ...],
    *,
    include_skiphash: bool = False,
) -> JsonValue:
    # TODO: when writing this to metadata, make sure to escape strings etc
    declared_type = strip_skiphash(declared_type)

    def assert_correct_dict_key(x: Any) -> str:
        if not isinstance(x, str):
            raise TypeError(
                f"Cannot serialize dict key {x!r} of type {type(x).__name__!r}: "
                "artifact dict keys must be strings"
            )
        if x in _RESERVED_DICT_KEYS:
            raise ValueError(
                f"Cannot serialize dict key {x!r}: "
                "it is reserved by Furu artifact serialization"
            )
        return x

    if isinstance(obj, type):
        return {
            KINDMARKER: "type_ref",
            CLASSMARKER: fully_qualified_name(obj),
        }

    if serializer := ArtifactSerializerMeta.serializer_for_dump(
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
        case list():
            return [
                to_json(
                    x,
                    declared_type=child_declared_type(declared_type, i),
                    artifact_serializers=artifact_serializers,
                    include_skiphash=include_skiphash,
                )
                for i, x in enumerate(obj)
            ]
        case tuple():
            return {
                KINDMARKER: "tuple",
                VALUEMARKER: [
                    to_json(
                        x,
                        declared_type=child_declared_type(declared_type, i),
                        artifact_serializers=artifact_serializers,
                        include_skiphash=include_skiphash,
                    )
                    for i, x in enumerate(obj)
                ],
            }
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
                            include_skiphash=include_skiphash,
                        )
                        for x in obj
                    ),
                    key=_stable_json_dump,
                ),
            }
        case dict():
            return {
                assert_correct_dict_key(k): to_json(
                    v,
                    declared_type=child_declared_type(declared_type, k),
                    artifact_serializers=artifact_serializers,
                    include_skiphash=include_skiphash,
                )
                for k, v in obj.items()
            }
        case x if is_dataclass(x):
            return {
                KINDMARKER: "instance",
                CLASSMARKER: fully_qualified_name(type(x)),
                FIELDSMARKER: {
                    field.name: to_json(
                        getattr(x, field.name),
                        declared_type=field.identity_type,
                        artifact_serializers=artifact_serializers,
                        include_skiphash=include_skiphash,
                    )
                    for field in dataclass_field_specs(
                        type(x), include_skiphash=include_skiphash
                    )
                },
            }
        case PydanticBaseModel():
            model_cls = type(obj)
            return {
                KINDMARKER: "instance",
                CLASSMARKER: fully_qualified_name(model_cls),
                FIELDSMARKER: {
                    field.name: to_json(
                        getattr(obj, field.name),
                        declared_type=field.identity_type,
                        artifact_serializers=artifact_serializers,
                        include_skiphash=include_skiphash,
                    )
                    for field in pydantic_field_specs(
                        model_cls, include_skiphash=include_skiphash
                    )
                },
            }
        case enum.Enum():
            raise TypeError(
                f"Cannot serialize enum value {obj!r}: enums are not supported "
                "in Furu artifacts yet"
            )
        case _:
            raise TypeError(
                f"Cannot serialize value {obj!r} of type {type(obj).__name__!r} "
                "into a Furu artifact; register an ArtifactSerializer for this type"
            )


_NO_RUNTIME_DATA = object()


def runtime_to_json(
    obj: Any,
    declared_type: object,
    artifact_serializers: tuple[type[ArtifactSerializer], ...],
) -> dict[str, JsonValue]:
    runtime_data = _runtime_to_json(
        obj,
        declared_type=declared_type,
        artifact_serializers=artifact_serializers,
    )
    if isinstance(runtime_data, dict):
        return cast(dict[str, JsonValue], runtime_data)
    return {}


def _runtime_to_json(
    obj: Any,
    *,
    declared_type: object,
    artifact_serializers: tuple[type[ArtifactSerializer], ...],
) -> JsonValue | object:
    identity_type, skip = split_skiphash(declared_type)
    if skip:
        return to_json(
            obj,
            declared_type=identity_type,
            artifact_serializers=artifact_serializers,
            include_skiphash=True,
        )

    declared_type = strip_skiphash(declared_type)

    if is_dataclass(obj) and not isinstance(obj, type):
        field_values: dict[str, JsonValue] = {}
        for field in dataclass_field_specs(type(obj), include_skiphash=True):
            value = getattr(obj, field.name)
            if field.skiphash:
                field_values[field.name] = to_json(
                    value,
                    declared_type=field.identity_type,
                    artifact_serializers=artifact_serializers,
                    include_skiphash=True,
                )
                continue

            nested = _runtime_to_json(
                value,
                declared_type=field.identity_type,
                artifact_serializers=artifact_serializers,
            )
            if nested is not _NO_RUNTIME_DATA:
                field_values[field.name] = cast(JsonValue, nested)
        if field_values:
            return field_values
        return _NO_RUNTIME_DATA

    if isinstance(obj, PydanticBaseModel):
        field_values: dict[str, JsonValue] = {}
        for field in pydantic_field_specs(type(obj), include_skiphash=True):
            value = getattr(obj, field.name)
            if field.skiphash:
                field_values[field.name] = to_json(
                    value,
                    declared_type=field.identity_type,
                    artifact_serializers=artifact_serializers,
                    include_skiphash=True,
                )
                continue

            nested = _runtime_to_json(
                value,
                declared_type=field.identity_type,
                artifact_serializers=artifact_serializers,
            )
            if nested is not _NO_RUNTIME_DATA:
                field_values[field.name] = cast(JsonValue, nested)
        if field_values:
            return field_values
        return _NO_RUNTIME_DATA

    if isinstance(obj, list):
        items: list[JsonValue] = []
        found_runtime = False
        for i, item in enumerate(obj):
            nested = _runtime_to_json(
                item,
                declared_type=child_declared_type(declared_type, i),
                artifact_serializers=artifact_serializers,
            )
            if nested is _NO_RUNTIME_DATA:
                items.append(None)
            else:
                found_runtime = True
                items.append(cast(JsonValue, nested))
        if found_runtime:
            return items
        return _NO_RUNTIME_DATA

    if isinstance(obj, tuple):
        items: list[JsonValue] = []
        found_runtime = False
        for i, item in enumerate(obj):
            nested = _runtime_to_json(
                item,
                declared_type=child_declared_type(declared_type, i),
                artifact_serializers=artifact_serializers,
            )
            if nested is _NO_RUNTIME_DATA:
                items.append(None)
            else:
                found_runtime = True
                items.append(cast(JsonValue, nested))
        if found_runtime:
            return {KINDMARKER: "tuple", VALUEMARKER: items}
        return _NO_RUNTIME_DATA

    if isinstance(obj, dict):
        fields: dict[str, JsonValue] = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                continue
            nested = _runtime_to_json(
                value,
                declared_type=child_declared_type(declared_type, key),
                artifact_serializers=artifact_serializers,
            )
            if nested is not _NO_RUNTIME_DATA:
                fields[key] = cast(JsonValue, nested)
        if fields:
            return fields
        return _NO_RUNTIME_DATA

    return _NO_RUNTIME_DATA


def _runtime_fields(runtime_data: JsonValue | None) -> dict[str, JsonValue]:
    if isinstance(runtime_data, dict):
        return runtime_data
    return {}


def _runtime_items(runtime_data: JsonValue | None) -> dict[int, JsonValue]:
    if (
        isinstance(runtime_data, dict)
        and runtime_data.get(KINDMARKER) == "tuple"
        and isinstance(runtime_data.get(VALUEMARKER), list)
    ):
        runtime_data = cast(JsonValue, runtime_data[VALUEMARKER])

    if not isinstance(runtime_data, list):
        return {}

    return {i: item for i, item in enumerate(runtime_data) if item is not None}


def _from_json_field(
    value: JsonValue,
    expected_type: Any,
    *,
    runtime_data: JsonValue | None = None,
) -> Any:
    expected_type = strip_skiphash(expected_type)
    if (
        isinstance(value, dict)
        and value.get(KINDMARKER) == "custom"
        and isinstance(value.get(SERIALIZERMARKER), str)
    ):
        return _load_custom_value(cast(dict[str, Any], value), expected_type)

    if isinstance(value, dict) and value.get(KINDMARKER) == "path":
        return Path(cast(str, value[VALUEMARKER]))

    if isinstance(value, dict) and value.get(KINDMARKER) in (
        "tuple",
        "set",
        "frozenset",
    ):
        runtime_items = _runtime_items(runtime_data)
        items = [
            _from_json_field(
                item,
                child_declared_type(expected_type, i),
                runtime_data=runtime_items.get(i),
            )
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
        return _from_json(value, runtime_data=runtime_data)

    if isinstance(value, list):
        runtime_items = _runtime_items(runtime_data)
        return [
            _from_json_field(
                item,
                child_declared_type(expected_type, i),
                runtime_data=runtime_items.get(i),
            )
            for i, item in enumerate(value)
        ]
    if isinstance(value, dict):
        runtime_fields = _runtime_fields(runtime_data)
        return {
            key: _from_json_field(
                child,
                child_declared_type(expected_type, key),
                runtime_data=runtime_fields.get(key),
            )
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


def _raise_for_missing_required_skiphash_fields(
    cls: type, converted_fields: dict[str, Any]
) -> None:
    missing: list[str] = []
    if is_dataclass(cls):
        for field in dataclass_field_specs(cls, include_skiphash=True):
            dataclass_field = field.dataclass_field
            if (
                not field.skiphash
                or field.name in converted_fields
                or dataclass_field is None
                or not dataclass_field.init
                or dataclass_field.default is not MISSING
                or dataclass_field.default_factory is not MISSING
            ):
                continue
            missing.append(field.name)
    elif issubclass(cls, PydanticBaseModel):
        for field in pydantic_field_specs(cls, include_skiphash=True):
            if (
                field.skiphash
                and field.name not in converted_fields
                and cls.model_fields[field.name].is_required()
            ):
                missing.append(field.name)

    if not missing:
        return

    fields_text = ", ".join(missing)
    raise TypeError(
        f"Cannot load {cls.__module__}.{cls.__qualname__} from an artifact without "
        f"runtime data for required skiphash field(s): {fields_text}"
    )


def _from_json(value: JsonValue, *, runtime_data: JsonValue | None = None) -> Any:
    match value:
        case list():
            runtime_items = _runtime_items(runtime_data)
            return [
                _from_json(item, runtime_data=runtime_items.get(i))
                for i, item in enumerate(value)
            ]
        case dict():
            class_name = value.get(CLASSMARKER)
            field_values = value.get(FIELDSMARKER)
            runtime_fields = _runtime_fields(runtime_data)
            if value.get(KINDMARKER) == "custom" and isinstance(
                value.get(SERIALIZERMARKER), str
            ):
                return _load_custom_value(cast(dict[str, Any], value), Any)
            if value.get(KINDMARKER) in ("path", "tuple", "set", "frozenset"):
                return _from_json_field(value, Any, runtime_data=runtime_data)
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
                    name: _from_json_field(
                        field_value,
                        hints.get(name, Any),
                        runtime_data=runtime_fields.get(name),
                    )
                    for name, field_value in field_values.items()
                }
                for name, field_value in runtime_fields.items():
                    if name in converted_fields:
                        continue
                    converted_fields[name] = _from_json_field(
                        field_value,
                        hints.get(name, Any),
                    )
                _raise_for_missing_required_skiphash_fields(cls, converted_fields)
                return cls(**converted_fields)
            return {
                key: _from_json(child, runtime_data=runtime_fields.get(key))
                for key, child in value.items()
            }
        case _:
            return value


def _from_artifact[T: "Furu"](
    artifact: ArtifactSpec,
    expected_type: type[T],
    *,
    runtime_data: dict[str, JsonValue] | None = None,
) -> T:
    artifact_type = _resolve_serialized_type(artifact.fully_qualified_name)
    if not issubclass(artifact_type, expected_type):
        raise TypeError(
            "Artifact described "
            + f"{artifact_type.__module__}.{artifact_type.__qualname__}, "
            + f"expected {expected_type.__module__}.{expected_type.__qualname__}"
        )

    furu_obj = _from_json_field(
        artifact.artifact_data,
        artifact_type,
        runtime_data=runtime_data,
    )
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
