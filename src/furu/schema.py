import enum
import types
import typing
from dataclasses import fields, is_dataclass
from typing import (
    Any,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel as PydanticBaseModel

from furu._typing import strip_annotated
from furu.constants import (
    ARGSMARKER,
    CLASSMARKER,
    FIELDSMARKER,
    KINDMARKER,
    ORIGINMARKER,
    SCHEMAMARKER,
    SERIALIZERMARKER,
)
from furu.serializer import ArtifactSerializer, SerializerRegistry, serializer_for_type
from furu.utils import JsonValue, _stable_json_dump, fully_qualified_name


def _custom_schema(
    serializer: type[ArtifactSerializer],
    declared_type: object,
) -> JsonValue:
    return {
        KINDMARKER: "custom",
        SERIALIZERMARKER: serializer._serializer_id(),
        SCHEMAMARKER: serializer.schema(strip_annotated(declared_type)),
    }


def schema_class(
    tp: type,
    field_names: list[str],
    seen: set[type],
    *,
    registry: SerializerRegistry,
) -> JsonValue:
    if tp in seen:
        return {CLASSMARKER: fully_qualified_name(tp)}
    seen.add(tp)

    hints = get_type_hints(tp, include_extras=True)
    return {
        CLASSMARKER: fully_qualified_name(tp),
        FIELDSMARKER: {
            name: schema_type(hints[name], seen, registry=registry)
            for name in field_names
        },
    }


def schema_dataclass(
    tp: type,
    seen: set[type],
    *,
    registry: SerializerRegistry,
) -> JsonValue:
    return schema_class(
        tp,
        sorted(f.name for f in fields(tp)),
        seen,
        registry=registry,
    )


def schema_pydantic_model(
    tp: type[PydanticBaseModel],
    seen: set[type],
    *,
    registry: SerializerRegistry,
) -> JsonValue:
    return schema_class(tp, sorted(tp.model_fields), seen, registry=registry)


def schema_type(
    tp: Any,
    seen: set[type],
    *,
    registry: SerializerRegistry | None = None,
) -> JsonValue:
    registry = SerializerRegistry.default() if registry is None else registry

    if serializer := serializer_for_type(tp, registry=registry):
        return _custom_schema(serializer, tp)

    origin = get_origin(tp)

    if origin is typing.Annotated:
        return schema_type(get_args(tp)[0], seen, registry=registry)
    if tp is Ellipsis:
        return fully_qualified_name(types.EllipsisType)
    if tp is Any:
        return "typing.Any"
    if isinstance(tp, typing.TypeAliasType):
        return schema_type(tp.__value__, seen, registry=registry)

    if isinstance(tp, type) and is_dataclass(tp):
        return schema_dataclass(tp, seen, registry=registry)
    if origin is not None and is_dataclass(origin):
        return schema_dataclass(origin, seen, registry=registry)
    if isinstance(tp, type) and issubclass(tp, PydanticBaseModel):
        return schema_pydantic_model(tp, seen, registry=registry)

    if origin in (typing.Union, types.UnionType):
        return sorted(
            [schema_type(a, seen, registry=registry) for a in get_args(tp)],
            key=_stable_json_dump,
        )
    elif origin is not None:
        assert (args := get_args(tp))  # TODO: maybe i need to remove this?
        return {
            ORIGINMARKER: fully_qualified_name(origin),
            ARGSMARKER: sorted(
                [schema_type(a, seen, registry=registry) for a in args],
                key=_stable_json_dump,
            ),
        }

    if tp is None or isinstance(tp, (str, bool, int, float)):
        return tp
    if tp in [str, float, int, bool, types.NoneType]:
        return fully_qualified_name(tp)
    elif isinstance(tp, str):
        return tp
    elif tp in [list, tuple, dict]:
        if get_args(tp) != ():
            raise ValueError(f"Expected bare {tp.__name__}, got parameterized type")
        return fully_qualified_name(tp)
    elif isinstance(tp, typing.TypeVar):
        return repr(tp)
    elif isinstance(tp, enum.EnumType):
        return fully_qualified_name(tp)
    elif isinstance(tp, type):
        return fully_qualified_name(tp)
    assert False, f"TODO: unexpected type value {tp=}"
