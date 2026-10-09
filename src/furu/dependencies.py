from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields
from functools import cached_property
from typing import TYPE_CHECKING, Any, Self, overload

from furu._tree import specs_in

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
def dependency[TSpec: Spec, T](
    func: Callable[[TSpec], T], /
) -> _CachedDependency[TSpec, T]: ...
@overload
def dependency[TSpec: Spec, T]() -> Callable[
    [Callable[[TSpec], T]], _CachedDependency[TSpec, T]
]: ...
def dependency[TSpec: Spec, T](
    func: Callable[[TSpec], T] | None = None, /
) -> (
    _CachedDependency[TSpec, T]
    | Callable[[Callable[[TSpec], T]], _CachedDependency[TSpec, T]]
):
    return dependency if func is None else _CachedDependency(func)


def collect_declared_refs(obj: Spec) -> tuple[Spec, ...]:
    refs_by_id: dict[str, Spec] = {}

    for field in fields(obj):
        for ref in specs_in(getattr(obj, field.name)):
            refs_by_id.setdefault(ref.object_id, ref)

    for base in reversed(type(obj).__mro__):
        for name, value in base.__dict__.items():
            if getattr(value, "__furu_dependency__", False):
                for ref in specs_in(getattr(obj, name)):
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


# TODO: ContextVar state does not propagate to new threads. If a create()
# hook runs child loads in worker threads, those loads will not be
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
def dependency_recorder() -> Generator[DependencyRecorder]:
    recorder = DependencyRecorder()
    token = _active_dependency_recorder.set(recorder)
    try:
        yield recorder
    finally:
        _active_dependency_recorder.reset(token)


_specs_under_creation: ContextVar[tuple[Spec, ...]] = ContextVar(
    "_specs_under_creation",
    default=(),
)


def specs_under_creation() -> tuple[Spec, ...]:
    return _specs_under_creation.get()


def is_under_creation(obj: Spec) -> bool:
    return any(spec.object_id == obj.object_id for spec in specs_under_creation())


@contextmanager
def under_creation(objs: Sequence[Spec]) -> Generator[None]:
    token = _specs_under_creation.set(tuple(objs))
    try:
        yield
    finally:
        _specs_under_creation.reset(token)


class _DependencyNotReady(BaseException):
    """create() inside a create hook found missing results.

    A BaseException so user ``except Exception`` blocks don't swallow it. The
    caller builds ``dependencies``, then reruns ``dependents`` from scratch.
    """

    def __init__(self, dependencies: Sequence[Spec], dependents: Sequence[Spec]):
        self.dependencies = tuple(dependencies)
        self.dependents = tuple(dependents)
        super().__init__(
            f"{self.dependents[0]._log_label} is blocked on "
            + missing_dependencies(len(self.dependencies))
        )


def missing_dependencies(count: int) -> str:
    return f"{count} missing {'dependency' if count == 1 else 'dependencies'}"
