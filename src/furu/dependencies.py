from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
import time
from typing import TYPE_CHECKING, Any, Callable, Literal, overload

from pydantic import BaseModel as PydanticBaseModel

if TYPE_CHECKING:
    from furu.core import Furu


type RecheckInterval = int | Literal["never"]

_NOT_FOUND = object()


class _Dependency[T]:
    __furu_dependency__ = True

    def __init__(
        self,
        func: Callable[..., T],
        *,
        recheck_interval: RecheckInterval = "never",
    ) -> None:
        if recheck_interval != "never" and recheck_interval < 0:
            raise ValueError("recheck_interval must be >= 0 or 'never'")

        self.func = func
        self.recheck_interval = recheck_interval
        self.attrname: str | None = None
        self.__doc__ = func.__doc__
        self.__module__ = func.__module__

    def __set_name__(self, owner: type[object], name: str) -> None:
        self.attrname = name

    def __set__(self, instance: object, value: object) -> None:
        raise AttributeError("can't set dependency attribute")

    @property
    def _checked_at_key(self) -> str:
        assert self.attrname is not None
        return f"__furu_dependency_checked_at_{self.attrname}"

    @overload
    def __get__(
        self, instance: None, owner: type[object] | None = None
    ) -> _Dependency[T]: ...

    @overload
    def __get__(self, instance: object, owner: type[object] | None = None) -> T: ...

    def __get__(
        self, instance: object | None, owner: type[object] | None = None
    ) -> T | _Dependency[T]:
        if instance is None:
            return self

        assert self.attrname is not None
        cache = instance.__dict__

        value = cache.get(self.attrname, _NOT_FOUND)

        if self.recheck_interval == "never":
            if value is _NOT_FOUND:
                value = self.func(instance)
                cache[self.attrname] = value
            return value

        checked_at = cache.get(self._checked_at_key, _NOT_FOUND)
        now = time.monotonic()

        if (
            value is _NOT_FOUND
            or checked_at is _NOT_FOUND
            or now - checked_at >= self.recheck_interval
        ):
            value = self.func(instance)
            cache[self.attrname] = value
            cache[self._checked_at_key] = now

        return value

    def recheck_due(self, instance: object) -> bool:
        if self.recheck_interval == "never":
            return False

        assert self.attrname is not None
        cache = instance.__dict__

        if self.attrname not in cache:
            return True

        checked_at = cache.get(self._checked_at_key, _NOT_FOUND)
        return (
            checked_at is _NOT_FOUND
            or time.monotonic() - checked_at >= self.recheck_interval
        )


@overload
def dependency[TFuru: Furu[Any], T](
    func: Callable[[TFuru], T], /
) -> _Dependency[T]: ...


@overload
def dependency[TFuru: Furu[Any], T](
    *,
    recheck_interval: RecheckInterval = "never",
) -> Callable[[Callable[[TFuru], T]], _Dependency[T]]: ...


def dependency[TFuru: Furu[Any], T](
    func: Callable[[TFuru], T] | None = None,
    /,
    *,
    recheck_interval: RecheckInterval = "never",
):
    def decorate(func: Callable[[TFuru], T]) -> _Dependency[T]:
        return _Dependency(func, recheck_interval=recheck_interval)

    if func is not None:
        return decorate(func)
    return decorate


def find_nested_furu_objects(value: object) -> Iterator[Furu]:
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


def collect_declared_refs(obj: Furu) -> tuple[Furu, ...]:
    refs_by_id: dict[str, Furu] = {}

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


def declared_dependency_recheck_due(obj: Furu) -> bool:
    for base in reversed(type(obj).__mro__):
        for value in base.__dict__.values():
            if isinstance(value, _Dependency) and value.recheck_due(obj):
                return True
    return False


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
