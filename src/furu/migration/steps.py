from __future__ import annotations

import dataclasses
import types
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias, assert_never

from furu.utils import JsonValue, fully_qualified_name

if TYPE_CHECKING:
    from furu.core import Spec


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


MigrationStep: TypeAlias = Renamed | Added | MovedFrom | Retyped | Rewrite


class Stale(RuntimeError):
    pass


class MigrationError(RuntimeError):
    pass


def _type_label(tp: object) -> str:
    if typing.get_origin(tp) in (typing.Union, types.UnionType):
        return " | ".join(_type_label(arg) for arg in typing.get_args(tp))
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


def validate_migration_declaration(cls: type[Spec[Any]]) -> None:
    steps = cls.migrations
    if not isinstance(steps, tuple) or not all(
        isinstance(step, MigrationStep) for step in steps
    ):
        raise TypeError(
            f"{cls.__name__}.migrations must be a tuple of "
            "Renamed/Added/MovedFrom/Retyped/Rewrite steps"
        )

    fields_by_name = {field.name: field for field in dataclasses.fields(cls)}
    names = {name: name for name in fields_by_name}

    for index in reversed(range(len(steps))):
        match steps[index]:
            case Renamed(field=field, to=to):
                if to not in names:
                    raise TypeError(
                        f"{cls.__name__}.migrations[{index}] ({_describe_step(steps[index])}): "
                        f"{to!r} is not a field; fields at that point in the chain: {sorted(names)}"
                    )
                if field in names:
                    raise TypeError(
                        f"{cls.__name__}.migrations[{index}] ({_describe_step(steps[index])}): "
                        f"{field!r} already exists; fields at that point in the chain: {sorted(names)}"
                    )
                names[field] = names.pop(to)
            case Added(field=field):
                if field not in names:
                    raise TypeError(
                        f"{cls.__name__}.migrations[{index}] ({_describe_step(steps[index])}): "
                        f"{field!r} is not a field; fields at that point in the chain: {sorted(names)}"
                    )
                current = fields_by_name[names.pop(field)]
                if (
                    current.default is dataclasses.MISSING
                    and current.default_factory is dataclasses.MISSING
                ):
                    raise TypeError(
                        f"{cls.__name__}.migrations[{index}] ({_describe_step(steps[index])}): "
                        f"field {current.name!r} has no default; Added can only "
                        "backfill a field with a default value; fields at that "
                        f"point in the chain: {sorted(names)}"
                    )
            case Retyped(field=field):
                if field not in names:
                    raise TypeError(
                        f"{cls.__name__}.migrations[{index}] ({_describe_step(steps[index])}): "
                        f"{field!r} is not a field; fields at that point in the chain: {sorted(names)}"
                    )
            case MovedFrom() | Rewrite():
                pass
            case unreachable:
                assert_never(unreachable)
