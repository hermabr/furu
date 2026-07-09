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


_NO_DEFAULT: Any = object()


@dataclass(frozen=True, slots=True)
class ResultRenamed:
    field: str
    _: dataclasses.KW_ONLY
    to: str


@dataclass(frozen=True, slots=True)
class ResultAdded:
    field: str
    _: dataclasses.KW_ONLY
    default: JsonValue


@dataclass(frozen=True, slots=True)
class ResultRewrite:
    transform: Callable[[JsonValue], JsonValue]


ResultMigration: TypeAlias = ResultRenamed | ResultAdded | ResultRewrite


@dataclass(frozen=True, slots=True)
class Renamed:
    field: str
    _: dataclasses.KW_ONLY
    to: str
    breaking: bool = False
    result: ResultMigration | None = None


@dataclass(frozen=True, slots=True)
class Added:
    field: str
    _: dataclasses.KW_ONLY
    default: Any = _NO_DEFAULT
    breaking: bool = False
    result: ResultMigration | None = None


@dataclass(frozen=True, slots=True)
class MovedFrom:
    fully_qualified_name: str
    _: dataclasses.KW_ONLY
    breaking: bool = False
    result: ResultMigration | None = None


@dataclass(frozen=True, slots=True)
class Retyped:
    field: str
    _: dataclasses.KW_ONLY
    was: Any
    breaking: bool = False
    result: ResultMigration | None = None


@dataclass(frozen=True, slots=True)
class Rewrite:
    transform: Callable[[Mapping[str, JsonValue]], Mapping[str, JsonValue]]
    _: dataclasses.KW_ONLY
    result: ResultMigration | None = None


MigrationStep: TypeAlias = Renamed | Added | MovedFrom | Retyped | Rewrite


def _is_breaking(step: MigrationStep) -> bool:
    return not isinstance(step, Rewrite) and step.breaking


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
            body = f"{field!r}, to={to!r}"
        case Added(field=field, default=default):
            body = (
                f"{field!r}"
                if default is _NO_DEFAULT
                else f"{field!r}, default={default!r}"
            )
        case MovedFrom(fully_qualified_name=name):
            body = f"{name!r}"
        case Retyped(field=field, was=was):
            body = f"{field!r}, was={_type_label(was)}"
        case Rewrite(transform=transform):
            body = f"{getattr(transform, '__qualname__', repr(transform))}"
    suffix = ", breaking=True" if _is_breaking(step) else ""
    if step.result is not None:
        suffix += f", result={_describe_result_migration(step.result)}"
    return f"{type(step).__name__}({body}{suffix})"


def _describe_result_migration(migration: ResultMigration) -> str:
    match migration:
        case ResultRenamed(field=field, to=to):
            body = f"{field!r}, to={to!r}"
        case ResultAdded(field=field, default=default):
            body = f"{field!r}, default={default!r}"
        case ResultRewrite(transform=transform):
            body = f"{getattr(transform, '__qualname__', repr(transform))}"
        case unreachable:
            assert_never(unreachable)
    return f"{type(migration).__name__}({body})"


def validate_migration_declaration(cls: type[Spec[Any]]) -> None:
    steps = cls.migrations
    if not isinstance(steps, tuple) or not all(
        isinstance(step, MigrationStep) for step in steps
    ):
        raise TypeError(
            f"{cls.__name__}.migrations must be a tuple of "
            "Renamed/Added/MovedFrom/Retyped/Rewrite steps"
        )

    names = {field.name: field.name for field in dataclasses.fields(cls)}

    result_migrations: list[tuple[int, MigrationStep]] = []

    for index in reversed(range(len(steps))):
        step = steps[index]
        if not (step.result is None or isinstance(step.result, ResultMigration)):
            raise TypeError(
                f"{cls.__name__}.migrations[{index}]: "
                "result= must be a ResultRenamed/ResultAdded/ResultRewrite step"
            )
        if step.result is not None:
            if _is_breaking(step):
                raise TypeError(
                    f"{cls.__name__}.migrations[{index}] ({_describe_step(step)}): "
                    "a breaking migration discards old results, so result= can "
                    "never migrate anything; drop one of the two"
                )
            result_migrations.append((index, step))

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
            case Added(field=field) as step:
                if field not in names:
                    raise TypeError(
                        f"{cls.__name__}.migrations[{index}] ({_describe_step(steps[index])}): "
                        f"{field!r} is not a field; fields at that point in the chain: {sorted(names)}"
                    )
                del names[field]
                if step.breaking and step.default is not _NO_DEFAULT:
                    raise TypeError(
                        f"{cls.__name__}.migrations[{index}] ({_describe_step(steps[index])}): "
                        "a breaking Added discards old results, so default= can "
                        "never backfill anything; drop one of the two"
                    )
                if not step.breaking and step.default is _NO_DEFAULT:
                    raise TypeError(
                        f"{cls.__name__}.migrations[{index}] ({_describe_step(steps[index])}): "
                        "Added needs default= (the value old runs behaved as, "
                        "pinned independently of the field's own default), or "
                        "breaking=True to discard the old results"
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

    if result_migrations:
        from furu.core import Spec

        if not issubclass(cls, Spec):
            index, step = result_migrations[-1]
            raise TypeError(
                f"{cls.__name__}.migrations[{index}] ({_describe_step(step)}): "
                "result= is only valid on a Spec migration; embedded plain "
                "dataclasses do not own cached results"
            )
