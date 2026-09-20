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

# Every non-breaking step may carry ``result_rewrite``: a function applied to
# an old run's stored *result* when it is loaded through this step. Like
# ``Rewrite.transform`` it works on the raw JSON as written to the result
# manifest (a dataclass result is ``{"$furu": {"|kind": "dataclass", "|type":
# ..., "|fields": {...}}}``), before anything is decoded or instantiated. So a
# chain that grew a result dataclass field by field declares one rewrite per
# step, each adding its field to ``|fields``, and the class is built once, from
# the final JSON. Rewrites run in declaration order, oldest generation first;
# the JSON is a private copy, so mutating it in place is fine.
ResultRewrite: TypeAlias = Callable[[JsonValue], JsonValue]  # noqa: UP040


@dataclass(frozen=True, slots=True)
class Renamed:
    field: str
    _: dataclasses.KW_ONLY
    to: str
    breaking: bool = False
    result_rewrite: ResultRewrite | None = None


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
    result_rewrite: ResultRewrite | None = None


@dataclass(frozen=True, slots=True)
class MovedFrom:
    fully_qualified_name: str
    _: dataclasses.KW_ONLY
    breaking: bool = False
    result_rewrite: ResultRewrite | None = None


@dataclass(frozen=True, slots=True)
class Retyped:
    field: str
    _: dataclasses.KW_ONLY
    was: Any
    breaking: bool = False
    result_rewrite: ResultRewrite | None = None


@dataclass(frozen=True, slots=True)
class Rewrite:
    transform: Callable[[Mapping[str, JsonValue]], Mapping[str, JsonValue]]
    _: dataclasses.KW_ONLY
    result_rewrite: ResultRewrite | None = None


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


def _callable_label(fn: Callable[..., object]) -> str:
    return getattr(fn, "__qualname__", repr(fn))


def _describe_step(step: MigrationStep) -> str:
    match step:
        case Renamed(field=field, to=to):
            body = f"{field!r}, to={to!r}"
        case Added(field=field, default=default, default_factory=factory):
            body = f"{field!r}"
            if default is not _NO_DEFAULT:
                body += f", default={default!r}"
            if factory is not None:
                body += f", default_factory={_callable_label(factory)}"
        case MovedFrom(fully_qualified_name=name):
            body = f"{name!r}"
        case Retyped(field=field, was=was):
            body = f"{field!r}, was={_type_label(was)}"
        case Rewrite(transform=transform):
            body = _callable_label(transform)
    suffix = ", breaking=True" if _is_breaking(step) else ""
    if step.result_rewrite is not None:
        suffix += f", result_rewrite={_callable_label(step.result_rewrite)}"
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

    from furu.core import Spec

    for index, step in enumerate(steps):
        if step.result_rewrite is None:
            continue
        if _is_breaking(step):
            raise TypeError(
                f"{cls.__name__}.migrations[{index}] ({_describe_step(step)}): "
                "a breaking step discards old results, so result_rewrite can "
                "never run; drop one of the two"
            )
        if not issubclass(cls, Spec):
            raise TypeError(
                f"{cls.__name__}.migrations[{index}] ({_describe_step(step)}): "
                f"{cls.__name__} has no results to rewrite; result_rewrite "
                "belongs on a Spec"
            )

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
