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

from furu.constants import ARGSMARKER, CLASSMARKER, ORIGINMARKER
from furu.utils import JsonValue, _stable_json_dump, fully_qualified_name


def schema_class(tp: type, field_names: list[str], seen: set[type]) -> JsonValue:
    if tp in seen:
        return {CLASSMARKER: fully_qualified_name(tp)}
    seen.add(tp)

    hints = get_type_hints(tp, include_extras=True)
    return {
        CLASSMARKER: fully_qualified_name(tp),
        "fields": {
            name: schema_type(hints[name], seen) for name in field_names
        },
    }


def schema_dataclass(tp: type, seen: set[type]) -> JsonValue:
    return schema_class(
        tp,
        sorted(f.name for f in fields(tp)),
        seen,
    )


def schema_pydantic_model(tp: type[PydanticBaseModel], seen: set[type]) -> JsonValue:
    return schema_class(tp, sorted(tp.model_fields), seen)


def schema_type(tp: Any, seen: set[type]) -> JsonValue:
    origin = get_origin(tp)

    if tp is Ellipsis:
        return fully_qualified_name(types.EllipsisType)
    if isinstance(tp, typing.TypeAliasType):
        return schema_type(tp.__value__, seen)

    if isinstance(tp, type) and is_dataclass(tp):
        return schema_dataclass(tp, seen)
    if origin is not None and is_dataclass(origin):
        return schema_dataclass(origin, seen)
    if isinstance(tp, type) and issubclass(tp, PydanticBaseModel):
        return schema_pydantic_model(tp, seen)

    if origin in (typing.Union, types.UnionType):
        return sorted(
            [schema_type(a, seen) for a in get_args(tp)], key=_stable_json_dump
        )
    elif origin is not None:
        assert (args := get_args(tp))  # TODO: maybe i need to remove this?
        return {
            ORIGINMARKER: fully_qualified_name(origin),
            ARGSMARKER: sorted(
                [schema_type(a, seen) for a in args], key=_stable_json_dump
            ),
        }

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
