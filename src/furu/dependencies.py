from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import is_dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, Self, overload

from pydantic import BaseModel as PydanticBaseModel

from furu._fields import dataclass_field_specs, pydantic_field_specs

if TYPE_CHECKING:
    from furu.core import Furu


class _CachedDependency[TOwner, T](cached_property):
    __furu_dependency__ = True

    if TYPE_CHECKING:

        @overload
        def __get__(
            self, instance: None, owner: type[TOwner] | None = None
        ) -> Self: ...

        @overload
        def __get__(self, instance: object, owner: type[Any] | None = None) -> T: ...


class _UncachedDependency[TOwner, T](property):
    __furu_dependency__ = True

    if TYPE_CHECKING:

        @overload
        def __get__(
            self, instance: None, owner: type[TOwner] | None = None
        ) -> Self: ...

        @overload
        def __get__(self, instance: object, owner: type[Any] | None = None) -> T: ...


@overload
def dependency[TFuru: Furu[Any], T](
    func: Callable[[TFuru], T], /
) -> _CachedDependency[TFuru, T]: ...


@overload
def dependency[TFuru: Furu[Any], T](
    *, cached: Literal[True] = True
) -> Callable[[Callable[[TFuru], T]], _CachedDependency[TFuru, T]]: ...


@overload
def dependency[TFuru: Furu[Any], T](
    *, cached: Literal[False]
) -> Callable[[Callable[[TFuru], T]], _UncachedDependency[TFuru, T]]: ...


@overload
def dependency[TFuru: Furu[Any], T](
    *, cached: bool
) -> Callable[
    [Callable[[TFuru], T]], _CachedDependency[TFuru, T] | _UncachedDependency[TFuru, T]
]: ...


def dependency[TFuru: Furu[Any], T](
    func: Callable[[TFuru], T] | None = None, /, *, cached: bool = True
) -> (
    _CachedDependency[TFuru, T]
    | _UncachedDependency[TFuru, T]
    | Callable[
        [Callable[[TFuru], T]],
        _CachedDependency[TFuru, T] | _UncachedDependency[TFuru, T],
    ]
):
    def decorate(
        func: Callable[[TFuru], T],
    ) -> _CachedDependency[TFuru, T] | _UncachedDependency[TFuru, T]:
        if cached:
            return _CachedDependency(func)
        return _UncachedDependency(func)

    if func is not None:
        return decorate(func)
    return decorate


def find_nested_furu_objects(value: object) -> Iterator[Furu]:
    from furu.core import Furu

    match value:
        case Furu():
            yield value
        case _ if is_dataclass(value) and not isinstance(value, type):
            for field in dataclass_field_specs(type(value), include_skip_hash=False):
                yield from find_nested_furu_objects(getattr(value, field.name))
        case PydanticBaseModel():
            for field in pydantic_field_specs(type(value), include_skip_hash=False):
                yield from find_nested_furu_objects(getattr(value, field.name))
        case tuple() | list() | set() | frozenset():
            for item in value:
                yield from find_nested_furu_objects(item)
        case dict():
            for item in value.values():
                yield from find_nested_furu_objects(item)


def collect_declared_refs(obj: Furu) -> tuple[Furu, ...]:
    refs_by_id: dict[str, Furu] = {}

    for field in dataclass_field_specs(type(obj), include_skip_hash=False):
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

    def record[T](self, obj: Furu[T]) -> None:
        self._observed_ids.add(obj.object_id)

    def finalize(self) -> tuple[str, ...]:
        return tuple(sorted(self._observed_ids))


# TODO: ContextVar state does not propagate to new threads. If create() or
# create_batched() runs child loads in worker threads, those loads will not be
# recorded unless recorder context is propagated explicitly.
_active_dependency_recorder: ContextVar[DependencyRecorder | None] = ContextVar(
    "_active_dependency_recorder",
    default=None,
)


def record_dependency_call[T](obj: Furu[T]) -> None:
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
