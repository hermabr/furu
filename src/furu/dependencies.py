from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel as PydanticBaseModel

if TYPE_CHECKING:
    from furu.core import Furu


class dependency[T](cached_property):
    __furu_dependency__ = True

    def __init__(self, func: Callable[..., T]):
        super().__init__(func)
        self.__name__ = getattr(func, "__name__", type(func).__name__)
        self.__doc__ = getattr(func, "__doc__", None)


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


def collect_eager_dependencies(obj: Furu[Any]) -> tuple[str, ...]:
    dependency_ids: set[str] = set()

    for field in fields(obj):
        for dep in find_nested_furu_objects(getattr(obj, field.name)):
            dependency_ids.add(dep.object_id)

    for base in reversed(type(obj).__mro__):
        for name, value in base.__dict__.items():
            if getattr(value, "__furu_dependency__", False):
                for dep in find_nested_furu_objects(getattr(obj, name)):
                    dependency_ids.add(dep.object_id)

    return tuple(sorted(dependency_ids))


class DependencyRecorder:
    def __init__(self) -> None:
        self._observed_ids: set[str] = set()

    def record(self, obj: Furu[Any]) -> None:
        self._observed_ids.add(obj.object_id)

    def finalize(self) -> tuple[str, ...]:
        return tuple(sorted(self._observed_ids))


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
