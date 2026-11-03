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

    Removing a field together with its ``Added`` makes the results stored with
    that field recompute quietly instead of going stale: to furu they look like
    a sibling branch's that added the field. Keep the step, or delete those
    results, if you meant something else.
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
    """Reshape old results' stored fields with ``transform``.

    Furu recognises a Rewrite by the step just before it, not by its function,
    so a lambda is fine and renaming the function changes nothing. Once results
    depend on it, keep what it does: an edited ``transform`` goes unnoticed.
    """

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


def _chain_identities(steps: tuple[MigrationStep, ...]) -> tuple[tuple[str, ...], ...]:
    """What each step does to a stored schema, so the same step declared on two
    branches is recognised as one.

    Defaults, ``breaking``, ``was=`` types and functions are left out, so
    editing or renaming them keeps a step's identity. A Retyped is told apart
    from earlier retypes of its field by count: the retype shows in the stored
    schema, so a mix-up is caught there. A Rewrite shows in no schema, so it is
    identified by the step before it, which two branches rarely share.
    """
    identities: list[tuple[str, ...]] = []
    retypes: dict[str, int] = {}
    for step in steps:
        match step:
            case Renamed(field=field, to=to):
                identity: tuple[str, ...] = ("Renamed", field, to)
            case Added(field=field):
                identity = ("Added", field)
            case Retyped(field=field):
                retypes[field] = retypes.get(field, 0) + 1
                identity = ("Retyped", field, str(retypes[field]))
            case MovedFrom(fully_qualified_name=name):
                identity = ("MovedFrom", name)
            case Rewrite():
                identity = ("Rewrite", *(identities[-1] if identities else ()))
        identities.append(identity)
    return tuple(identities)


def _describe_identity(identity: tuple[str, ...]) -> str:
    match identity:
        case ():
            return "the start of the chain"
        case ("Rewrite", *before):
            return f"a Rewrite after {_describe_identity(tuple(before))}"
        case ("Retyped", field, count):
            return f"retype {count} of {field!r}"
        case _:
            kind, *fields = identity
            return f"{kind}({', '.join(repr(field) for field in fields)})"


def _touched_fields(step: MigrationStep) -> frozenset[str] | None:
    """The fields a step reads or writes; None when it sees the whole record."""
    match step:
        case Renamed(field=field, to=to):
            return frozenset((field, to))
        case Added(field=field, default_factory=factory):
            return None if factory is not None else frozenset((field,))
        case Retyped(field=field):
            return frozenset((field,))
        case MovedFrom():
            return frozenset()
        case Rewrite():
            return None


def _commutes(earlier: MigrationStep, later: MigrationStep) -> bool:
    """Whether ``earlier`` may run after ``later`` without changing the result,
    as when two branches made them in parallel."""
    if earlier.result_rewrite is not None and later.result_rewrite is not None:
        return False
    if isinstance(earlier, MovedFrom) or isinstance(later, MovedFrom):
        # A move only changes where old schemas live, not their fields.
        return not (isinstance(earlier, MovedFrom) and isinstance(later, MovedFrom))
    earlier_fields, later_fields = _touched_fields(earlier), _touched_fields(later)
    if earlier_fields is None or later_fields is None:
        return False
    return not earlier_fields & later_fields


def validate_migration_declaration(cls: type[Spec]) -> None:
    steps = cls.migrations
    if not isinstance(steps, tuple) or not all(
        isinstance(step, MigrationStep) for step in steps
    ):
        raise TypeError(
            f"{cls.__name__}.migrations must be a tuple of "
            "Renamed/Added/MovedFrom/Retyped/Rewrite steps"
        )

    seen: dict[tuple[str, ...], int] = {}
    for index, identity in enumerate(_chain_identities(steps)):
        if (first := seen.setdefault(identity, index)) != index:
            raise TypeError(
                f"{cls.__name__}.migrations[{index}] "
                f"({_describe_step(steps[index])}) repeats migrations[{first}]; "
                "furu tells which steps a stored schema already has by what they "
                "do, so each step must differ"
            )

    names = {field.name for field in dataclasses.fields(cls)}

    from furu.core import Spec

    def invalid(index: int, problem: str) -> TypeError:
        return TypeError(
            f"{cls.__name__}.migrations[{index}] ({_describe_step(steps[index])}): "
            f"{problem}"
        )

    def not_a_field(index: int, field: str) -> TypeError:
        return invalid(
            index,
            f"{field!r} is not a field; fields at that point in the chain: "
            f"{sorted(names)}",
        )

    for index, step in enumerate(steps):
        if step.result_rewrite is None:
            continue
        if _is_breaking(step):
            raise invalid(
                index,
                "a breaking step discards old results, so result_rewrite can "
                "never run; drop one of the two",
            )
        if not issubclass(cls, Spec):
            raise invalid(
                index,
                f"{cls.__name__} has no results to rewrite; result_rewrite "
                "belongs on a Spec",
            )

    for index in reversed(range(len(steps))):
        match steps[index]:
            case Renamed(field=field, to=to):
                if to not in names:
                    raise not_a_field(index, to)
                if field in names:
                    raise invalid(
                        index,
                        f"{field!r} already exists; fields at that point in the "
                        f"chain: {sorted(names)}",
                    )
                names.remove(to)
                names.add(field)
            case Added(field=field) as step:
                if field not in names:
                    raise not_a_field(index, field)
                names.remove(field)
                backfills = (step.default is not _NO_DEFAULT) + (
                    step.default_factory is not None
                )
                if step.breaking and backfills:
                    raise invalid(
                        index,
                        "a breaking Added discards old results, so a default can "
                        "never backfill anything; drop one of the two",
                    )
                if not step.breaking and backfills != 1:
                    raise invalid(
                        index,
                        "Added needs exactly one of default= (the value old runs "
                        "behaved as, pinned independently of the field's own "
                        "default) or default_factory= (computing it from the old "
                        "run's stored fields), or breaking=True to discard the "
                        "old results",
                    )
            case Retyped(field=field):
                if field not in names:
                    raise not_a_field(index, field)
            case MovedFrom() | Rewrite():
                pass
            case unreachable:
                assert_never(unreachable)
