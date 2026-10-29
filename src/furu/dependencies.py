from __future__ import annotations

import os
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields
from functools import cached_property
from pathlib import Path
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
    """Which version of each dependency a create() loaded.

    Versions are stored relative to the creating spec's storage root, so a
    lookup can re-check them without a Spec object.
    """

    def __init__(
        self,
        storage_root: Path,
        versions: dict[str, str],
        log: Callable[[list[str]], None] | None,
    ) -> None:
        self._storage_root = storage_root
        self.versions = versions
        self._log = log

    def record(self, obj: Spec, version: Path) -> None:
        relative = os.path.relpath(version, self._storage_root)
        if self.versions.get(obj.object_id) != relative:
            self.versions[obj.object_id] = relative
            if self._log is not None:
                self._log(["dependency", obj.object_id, relative])


# TODO: dependencies loaded in threads a create() starts aren't recorded
# (ContextVar state doesn't reach them). Workaround: submit the work with
# contextvars.copy_context().run. Check with a human before adding such threads.
_active_dependency_recorder: ContextVar[DependencyRecorder | None] = ContextVar(
    "_active_dependency_recorder",
    default=None,
)


def record_dependency(obj: Spec, version: Path) -> None:
    recorder = _active_dependency_recorder.get()
    if recorder is not None:
        recorder.record(obj, version)


@contextmanager
def dependency_recorder(
    storage_root: Path,
    versions: dict[str, str],
    log: Callable[[list[str]], None] | None,
) -> Generator[DependencyRecorder]:
    recorder = DependencyRecorder(storage_root, versions, log)
    token = _active_dependency_recorder.set(recorder)
    try:
        yield recorder
    finally:
        _active_dependency_recorder.reset(token)


_specs_under_creation: ContextVar[frozenset[str]] = ContextVar(
    "_specs_under_creation",
    default=frozenset(),
)


def is_under_creation(obj: Spec) -> bool:
    return obj.object_id in _specs_under_creation.get()


def any_under_creation() -> bool:
    return bool(_specs_under_creation.get())


@contextmanager
def under_creation(objs: Sequence[Spec]) -> Generator[None]:
    token = _specs_under_creation.set(frozenset(obj.object_id for obj in objs))
    try:
        yield
    finally:
        _specs_under_creation.reset(token)


class _DependencyNotReady(BaseException):
    """create() inside a create hook found missing results.

    A BaseException so user ``except Exception`` blocks don't swallow it. The
    runner builds ``dependencies``, then reruns the blocked create from scratch.
    """

    def __init__(self, dependencies: Sequence[Spec]):
        self.dependencies = tuple(dependencies)
        super().__init__(missing_dependencies(len(self.dependencies)))


def missing_dependencies(count: int) -> str:
    return f"{count} missing {'dependency' if count == 1 else 'dependencies'}"
