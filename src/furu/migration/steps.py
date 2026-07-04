from __future__ import annotations

import dataclasses
import types
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from furu.utils import JsonValue, fully_qualified_name


@dataclass(frozen=True, slots=True)
class Renamed:
    field: str
    _: dataclasses.KW_ONLY
    to: str


@dataclass(frozen=True, slots=True)
class Added:
    field: str


@dataclass(frozen=True, slots=True)
class MovedFrom:
    fully_qualified_name: str


@dataclass(frozen=True, slots=True)
class Retyped:
    field: str
    _: dataclasses.KW_ONLY
    was: Any


@dataclass(frozen=True, slots=True)
class Rewrite:
    transform: Callable[[Mapping[str, JsonValue]], Mapping[str, JsonValue]]


type MigrationStep = Renamed | Added | MovedFrom | Retyped | Rewrite


class Stale(RuntimeError):
    pass


class MigrationError(RuntimeError):
    pass


def _type_label(tp: object) -> str:
    if typing.get_origin(tp) in (typing.Union, types.UnionType):
        return " | ".join(_type_label(a) for a in typing.get_args(tp))
    if isinstance(tp, type):
        return fully_qualified_name(tp)
    return repr(tp)


def _describe_step(step: MigrationStep) -> str:
    match step:
        case Renamed(field=field, to=to):
            return f"Renamed({field!r}, to={to!r})"
        case Added(field=field):
            return f"Added({field!r})"
        case MovedFrom(fully_qualified_name=name):
            return f"MovedFrom({name!r})"
        case Retyped(field=field, was=was):
            return f"Retyped({field!r}, was={_type_label(was)})"
        case Rewrite(transform=transform):
            return f"Rewrite({getattr(transform, '__qualname__', repr(transform))})"
