from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, Literal, overload

from pydantic import BaseModel as PydanticBaseModel

if TYPE_CHECKING:
    from furu.core import Furu


class _CachedDependency[T](cached_property):
    __furu_dependency__ = True


class _UncachedDependency[T](property):
    __furu_dependency__ = True


@overload
def dependency[T](func: Callable[[Any], T], /) -> _CachedDependency[T]: ...


@overload
def dependency[T](
    *, cached: Literal[True] = True
) -> Callable[[Callable[[Any], T]], _CachedDependency[T]]: ...


@overload
def dependency[T](
    *, cached: Literal[False]
) -> Callable[[Callable[[Any], T]], _UncachedDependency[T]]: ...


@overload
def dependency[T](
    *, cached: bool
) -> Callable[[Callable[[Any], T]], _CachedDependency[T] | _UncachedDependency[T]]: ...


def dependency[T](
    func: Callable[[Any], T] | None = None, /, *, cached: bool = True
) -> (
    _CachedDependency[T]
    | _UncachedDependency[T]
    | Callable[[Callable[[Any], T]], _CachedDependency[T] | _UncachedDependency[T]]
):
    def decorate(
        func: Callable[[Any], T],
    ) -> _CachedDependency[T] | _UncachedDependency[T]:
        if cached:
            return _CachedDependency(func)
        return _UncachedDependency(func)

    if func is not None:
        return decorate(func)
    return decorate


def find_nested_furu_objects(value: object) -> Iterator[Furu[Any]]:
    from furu.core import Furu

    match value:
        case Furu():
            yield value
        case _ if is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                yield from find_nested_furu_objects(getattr(value, field.name))
        case PydanticBaseModel():
            for name in type(value).model_fields:
                yield from find_nested_furu_objects(getattr(value, name))
        case tuple() | list() | set() | frozenset():
            for item in value:
                yield from find_nested_furu_objects(item)
        case dict():
            for item in value.values():
                yield from find_nested_furu_objects(item)


def collect_declared_refs(obj: Furu[Any]) -> tuple[Furu[Any], ...]:
    refs_by_id: dict[str, Furu[Any]] = {}

    for field in fields(obj):
        for ref in find_nested_furu_objects(getattr(obj, field.name)):
            refs_by_id.setdefault(ref.object_id, ref)

    for base in reversed(type(obj).__mro__):
        for name, value in base.__dict__.items():
            if getattr(value, "__furu_dependency__", False):
                for ref in find_nested_furu_objects(getattr(obj, name)):
                    refs_by_id.setdefault(ref.object_id, ref)

    return tuple(
        ref
        for _, ref in sorted(
            refs_by_id.items(),
            key=lambda item: item[0],
        )
    )


class DependencyRecorder:
    def __init__(self) -> None:
        self._observed_ids: set[str] = set()

    def record(self, obj: Furu[Any]) -> None:
        self._observed_ids.add(obj.object_id)

    def finalize(self) -> tuple[str, ...]:
        return tuple(sorted(self._observed_ids))


# TODO: ContextVar state does not propagate to new threads. If _create() or
# _create_batched() runs child loads in worker threads, those loads will not be
# recorded unless recorder context is propagated explicitly.
_active_dependency_recorder: ContextVar[DependencyRecorder | None] = ContextVar(
    "_active_dependency_recorder",
    default=None,
)


def record_dependency_call(obj: Furu[Any]) -> None:
    recorder = _active_dependency_recorder.get()
    if recorder is not None:
        recorder.record(obj)


@contextmanager
def dependency_recorder() -> Iterator[DependencyRecorder]:
    recorder = DependencyRecorder()
    token = _active_dependency_recorder.set(recorder)
    try:
        yield recorder
    finally:
        _active_dependency_recorder.reset(token)
