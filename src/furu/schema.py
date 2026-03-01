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

from furu.constants import ARGSMARKER, CLASSMARKER, ORIGINMARKER
from furu.utils import JsonValue, _stable_json_dump, fully_qualified_name


def schema_dataclass(tp: type, seen: set[type]) -> JsonValue:
    if tp in seen:
        return {CLASSMARKER: fully_qualified_name(tp)}
    seen.add(tp)

    hints = get_type_hints(tp, include_extras=True)
    return {
        CLASSMARKER: fully_qualified_name(tp),
        "fields": {
            f.name: schema_type(hints[f.name], seen)
            for f in sorted(fields(tp), key=lambda f: f.name)
            if not f.name.startswith("_")
        },
    }


def schema_type(tp: Any, seen: set[type]) -> JsonValue:
    origin = get_origin(tp)

    if isinstance(tp, type) and is_dataclass(tp):
        return schema_dataclass(tp, seen)
    if origin is not None and is_dataclass(origin):
        return schema_dataclass(origin, seen)

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

    if tp in [str, float, int, bool]:
        return fully_qualified_name(tp)
    elif isinstance(tp, str):
        return tp
    elif tp in [list, tuple, dict]:
        assert get_args(tp) == ()
        return fully_qualified_name(tp)
    elif isinstance(tp, typing.TypeVar):
        return repr(tp)
    elif isinstance(tp, enum.EnumType):
        return fully_qualified_name(tp)
    assert False, f"TODO: unexpected type value {tp=}"
