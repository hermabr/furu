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


_NO_DEFAULT = object()


@dataclass(frozen=True, slots=True)
class Renamed:
    field: str
    _: dataclasses.KW_ONLY
    to: str
    breaking: bool = False


@dataclass(frozen=True, slots=True)
class Added:
    """Backfill ``field`` on old results.

    ``default`` is the one value every old run behaved as. ``default_factory``
    computes it per run from that run's stored fields (as ``Rewrite`` sees
    them): ``Added("model", default_factory=lambda f: ModelConfig(width=f["width"]))``.
    """

    field: str
    _: dataclasses.KW_ONLY
    default: object = _NO_DEFAULT
    default_factory: Callable[[Mapping[str, JsonValue]], object] | None = None
    breaking: bool = False


@dataclass(frozen=True, slots=True)
class MovedFrom:
    fully_qualified_name: str
    _: dataclasses.KW_ONLY
    breaking: bool = False


@dataclass(frozen=True, slots=True)
class Retyped:
    field: str
    _: dataclasses.KW_ONLY
    was: Any
    breaking: bool = False


@dataclass(frozen=True, slots=True)
class Rewrite:
    transform: Callable[[Mapping[str, JsonValue]], Mapping[str, JsonValue]]


MigrationStep: TypeAlias = Renamed | Added | MovedFrom | Retyped | Rewrite  # noqa: UP040


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
        case Added(field=field, default=default, default_factory=factory):
            body = f"{field!r}"
            if default is not _NO_DEFAULT:
                body += f", default={default!r}"
            if factory is not None:
                body += f", default_factory={getattr(factory, '__qualname__', repr(factory))}"
        case MovedFrom(fully_qualified_name=name):
            body = f"{name!r}"
        case Retyped(field=field, was=was):
            body = f"{field!r}, was={_type_label(was)}"
        case Rewrite(transform=transform):
            body = f"{getattr(transform, '__qualname__', repr(transform))}"
    suffix = ", breaking=True" if _is_breaking(step) else ""
    return f"{type(step).__name__}({body}{suffix})"


def validate_migration_declaration(cls: type[Spec]) -> None:
    steps = cls.migrations
    if not isinstance(steps, tuple) or not all(
        isinstance(step, MigrationStep) for step in steps
    ):
        raise TypeError(
            f"{cls.__name__}.migrations must be a tuple of "
            "Renamed/Added/MovedFrom/Retyped/Rewrite steps"
        )

    names = {field.name: field.name for field in dataclasses.fields(cls)}

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
            case Added(field=field) as step:
                if field not in names:
                    raise TypeError(
                        f"{cls.__name__}.migrations[{index}] ({_describe_step(steps[index])}): "
                        f"{field!r} is not a field; fields at that point in the chain: {sorted(names)}"
                    )
                del names[field]
                backfills = (step.default is not _NO_DEFAULT) + (
                    step.default_factory is not None
                )
                if step.breaking and backfills:
                    raise TypeError(
                        f"{cls.__name__}.migrations[{index}] ({_describe_step(steps[index])}): "
                        "a breaking Added discards old results, so a default can "
                        "never backfill anything; drop one of the two"
                    )
                if not step.breaking and backfills != 1:
                    raise TypeError(
                        f"{cls.__name__}.migrations[{index}] ({_describe_step(steps[index])}): "
                        "Added needs exactly one of default= (the value old runs "
                        "behaved as, pinned independently of the field's own "
                        "default) or default_factory= (computing it from the old "
                        "run's stored fields), or breaking=True to discard the "
                        "old results"
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
