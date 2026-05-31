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
type _DependencyCacheEntry[T] = tuple[float | None, T]


class _Dependency[T](property):
    __furu_dependency__ = True

    def __init__(
        self,
        func: Callable[[Any], T],
        *,
        recheck_interval: RecheckInterval,
    ) -> None:
        if isinstance(recheck_interval, bool):
            raise TypeError(
                "recheck_interval must be a non-negative integer or 'never'"
            )
        if isinstance(recheck_interval, int):
            if recheck_interval < 0:
                raise ValueError("recheck_interval must be non-negative")
        elif recheck_interval != "never":
            raise TypeError(
                "recheck_interval must be a non-negative integer or 'never'"
            )

        super().__init__(func)
        self._recheck_interval = recheck_interval
        func_name = getattr(func, "__name__", type(func).__name__)
        self._cache_key = f"__furu_dependency_cache_{func_name}"

    def __set_name__(self, owner: type[object], name: str) -> None:
        self._cache_key = f"__furu_dependency_cache_{name}"

    @overload
    def __get__(
        self,
        obj: None,
        objtype: type[object] | None = None,
    ) -> _Dependency[T]: ...

    @overload
    def __get__(
        self,
        obj: object,
        objtype: type[object] | None = None,
    ) -> T: ...

    def __get__(
        self,
        obj: object | None,
        objtype: type[object] | None = None,
    ) -> T | _Dependency[T]:
        if obj is None:
            return self

        cached: _DependencyCacheEntry[T] | None = getattr(obj, self._cache_key, None)
        now: float | None = None
        if cached is not None:
            expires_at, value = cached
            if expires_at is None:
                return value

            now = time.monotonic()
            if now < expires_at:
                return value

        if self.fget is None:
            raise AttributeError("unreadable dependency")

        recheck_interval = self._recheck_interval
        if isinstance(recheck_interval, int) and now is None:
            now = time.monotonic()

        value = self.fget(obj)
        if recheck_interval == "never":
            expires_at = None
        else:
            assert isinstance(recheck_interval, int)
            assert now is not None
            expires_at = now + recheck_interval
        object.__setattr__(obj, self._cache_key, (expires_at, value))
        return value


@overload
def dependency[TFuru: Furu[Any], T](
    func: Callable[[TFuru], T], /
) -> _Dependency[T]: ...


@overload
def dependency[TFuru: Furu[Any], T](
    *, recheck_interval: RecheckInterval = "never"
) -> Callable[[Callable[[TFuru], T]], _Dependency[T]]: ...


def dependency[TFuru: Furu[Any], T](
    func: Callable[[TFuru], T] | None = None,
    /,
    *,
    recheck_interval: RecheckInterval = "never",
) -> _Dependency[T] | Callable[[Callable[[TFuru], T]], _Dependency[T]]:
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
