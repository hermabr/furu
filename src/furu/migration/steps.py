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
class ResultAdded:
    field: str
    _: dataclasses.KW_ONLY
    value: Any


@dataclass(frozen=True, slots=True)
class ResultRenamed:
    field: str
    _: dataclasses.KW_ONLY
    to: str


@dataclass(frozen=True, slots=True)
class ResultRemoved:
    field: str


@dataclass(frozen=True, slots=True)
class ResultRewrite:
    transform: Callable[..., Mapping[str, JsonValue]]


ResultChange: TypeAlias = ResultAdded | ResultRenamed | ResultRemoved | ResultRewrite


@dataclass(frozen=True, slots=True)
class Renamed:
    field: str
    _: dataclasses.KW_ONLY
    to: str
    breaking: bool = False
    result_changes: tuple[ResultChange, ...] = ()


@dataclass(frozen=True, slots=True)
class Added:
    field: str
    _: dataclasses.KW_ONLY
    default: Any = _NO_DEFAULT
    breaking: bool = False
    result_changes: tuple[ResultChange, ...] = ()


@dataclass(frozen=True, slots=True)
class MovedFrom:
    fully_qualified_name: str
    _: dataclasses.KW_ONLY
    breaking: bool = False
    result_changes: tuple[ResultChange, ...] = ()


@dataclass(frozen=True, slots=True)
class Retyped:
    field: str
    _: dataclasses.KW_ONLY
    was: Any
    breaking: bool = False
    result_changes: tuple[ResultChange, ...] = ()


@dataclass(frozen=True, slots=True)
class Rewrite:
    transform: Callable[[Mapping[str, JsonValue]], Mapping[str, JsonValue]]
    _: dataclasses.KW_ONLY
    result_changes: tuple[ResultChange, ...] = ()


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


def _describe_result_change(change: ResultChange) -> str:
    match change:
        case ResultAdded(field=field, value=value):
            body = f"{field!r}, value={value!r}"
        case ResultRenamed(field=field, to=to):
            body = f"{field!r}, to={to!r}"
        case ResultRemoved(field=field):
            body = f"{field!r}"
        case ResultRewrite(transform=transform):
            body = f"{getattr(transform, '__qualname__', repr(transform))}"
    return f"{type(change).__name__}({body})"


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
    if step.result_changes:
        inner = ", ".join(map(_describe_result_change, step.result_changes))
        comma = "," if len(step.result_changes) == 1 else ""
        suffix += f", result_changes=({inner}{comma})"
    return f"{type(step).__name__}({body}{suffix})"


def _result_field_names(cls: type[Spec[Any]]) -> set[str] | None:
    """Field names of cls's declared result type, or None when not introspectable."""
    from furu._declared_types import declared_result_type, strip_annotated

    import pydantic

    try:
        declared = strip_annotated(declared_result_type(cls))
    except TypeError:
        return None
    if isinstance(declared, type) and dataclasses.is_dataclass(declared):
        return {field.name for field in dataclasses.fields(declared)}
    if isinstance(declared, type) and issubclass(declared, pydantic.BaseModel):
        return set(declared.model_fields)
    return None


def _validate_result_changes_declaration(
    cls: type[Spec[Any]], steps: tuple[MigrationStep, ...]
) -> None:
    for index, step in enumerate(steps):
        if not isinstance(step.result_changes, tuple) or not all(
            isinstance(change, ResultChange) for change in step.result_changes
        ):
            raise TypeError(
                f"{cls.__name__}.migrations[{index}]: result_changes must be a "
                "tuple of ResultAdded/ResultRenamed/ResultRemoved/ResultRewrite steps"
            )
        if step.result_changes and _is_breaking(step):
            raise TypeError(
                f"{cls.__name__}.migrations[{index}] ({_describe_step(steps[index])}): "
                "a breaking step discards old results, so result_changes can "
                "never apply to anything; drop one of the two"
            )

    result_names = _result_field_names(cls)
    if result_names is None:
        return
    for index in reversed(range(len(steps))):
        for change in reversed(steps[index].result_changes):
            prefix = (
                f"{cls.__name__}.migrations[{index}] "
                f"({_describe_result_change(change)})"
            )
            match change:
                case ResultAdded(field=field):
                    if field not in result_names:
                        raise TypeError(
                            f"{prefix}: {field!r} is not a result field; result "
                            f"fields at that point in the chain: {sorted(result_names)}"
                        )
                    result_names.remove(field)
                case ResultRenamed(field=field, to=to):
                    if to not in result_names:
                        raise TypeError(
                            f"{prefix}: {to!r} is not a result field; result "
                            f"fields at that point in the chain: {sorted(result_names)}"
                        )
                    if field in result_names:
                        raise TypeError(
                            f"{prefix}: {field!r} already exists; result fields "
                            f"at that point in the chain: {sorted(result_names)}"
                        )
                    result_names.remove(to)
                    result_names.add(field)
                case ResultRemoved(field=field):
                    if field in result_names:
                        raise TypeError(
                            f"{prefix}: {field!r} still exists at that point in "
                            "the chain, so the step can never remove anything"
                        )
                    result_names.add(field)
                case ResultRewrite():
                    pass
                case unreachable:
                    assert_never(unreachable)


def validate_migration_declaration(cls: type[Spec[Any]]) -> None:
    steps = cls.migrations
    if not isinstance(steps, tuple) or not all(
        isinstance(step, MigrationStep) for step in steps
    ):
        raise TypeError(
            f"{cls.__name__}.migrations must be a tuple of "
            "Renamed/Added/MovedFrom/Retyped/Rewrite steps"
        )

    _validate_result_changes_declaration(cls, steps)

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
