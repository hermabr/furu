from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Self, overload

from pydantic import BaseModel as PydanticBaseModel

if TYPE_CHECKING:
    from furu.core import Spec


class _CachedDependency[TOwner, T](cached_property):
    __furu_dependency__ = True

    if TYPE_CHECKING:

        @overload
        def __get__(
            self, instance: None, owner: type[TOwner] | None = None
        ) -> Self: ...

        @overload
        def __get__(self, instance: object, owner: type[Any] | None = None) -> T: ...


@overload
def dependency[TSpec: Spec[Any], T](
    func: Callable[[TSpec], T], /
) -> _CachedDependency[TSpec, T]: ...
@overload
def dependency[TSpec: Spec[Any], T]() -> Callable[
    [Callable[[TSpec], T]], _CachedDependency[TSpec, T]
]: ...
def dependency[TSpec: Spec[Any], T](
    func: Callable[[TSpec], T] | None = None, /
) -> (
    _CachedDependency[TSpec, T]
    | Callable[[Callable[[TSpec], T]], _CachedDependency[TSpec, T]]
):
    return dependency if func is None else _CachedDependency(func)


def find_nested_furu_objects(value: object) -> Iterator[Spec]:
    from furu.core import Spec

    match value:
        case Spec():
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


def collect_declared_refs(obj: Spec) -> tuple[Spec, ...]:
    refs_by_id: dict[str, Spec] = {}

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

    def record[T](self, obj: Spec[T]) -> None:
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


def record_dependency_call[T](obj: Spec[T]) -> None:
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
