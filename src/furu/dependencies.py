from __future__ import annotations

import os
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, overload

from furu._tree import specs_in

if TYPE_CHECKING:
    from furu.core import Spec

_DERIVE = "furu_dependency"  # field metadata key: the method that computes it


class _Dependency[TSpec, T]:
    """A ``@furu.dependency`` method, until its Spec class turns it into a field.

    The field is filled from ``func`` when a spec is constructed, so what the
    method returns is hashed, stored and planned like any other field.
    """

    def __init__(self, func: Callable[[TSpec], T]) -> None:
        self.func = func

    if TYPE_CHECKING:

        @overload
        def __get__(self, instance: None, owner: type[TSpec] | None = None) -> Self: ...

        @overload
        def __get__(self, instance: object, owner: type[Any] | None = None) -> T: ...


@overload
def dependency[TSpec: Spec, T](
    func: Callable[[TSpec], T], /
) -> _Dependency[TSpec, T]: ...
@overload
def dependency[TSpec: Spec, T]() -> Callable[
    [Callable[[TSpec], T]], _Dependency[TSpec, T]
]: ...
def dependency[TSpec: Spec, T](
    func: Callable[[TSpec], T] | None = None, /
) -> _Dependency[TSpec, T] | Callable[[Callable[[TSpec], T]], _Dependency[TSpec, T]]:
    """Declare specs a create() uses that are computed from the spec's fields.

    The method becomes a field that furu fills in when the spec is constructed.
    """
    return dependency if func is None else _Dependency(func)


def dependency_field(cls: type, name: str, declared: _Dependency) -> Any:
    """The dataclass field ``cls.name`` becomes; its type is the return annotation."""
    if "return" not in declared.func.__annotations__:
        raise TypeError(
            f"{cls.__name__}.{name}: @furu.dependency needs a return annotation; "
            "it is the type of the field furu stores."
        )
    return field(init=False, metadata={_DERIVE: declared.func})


def fill_dependencies(obj: Spec) -> None:
    for f in fields(obj):
        if (derive := f.metadata.get(_DERIVE)) is not None:
            object.__setattr__(obj, f.name, derive(obj))


def collect_declared_refs(obj: Spec) -> tuple[Spec, ...]:
    refs_by_id: dict[str, Spec] = {}
    for f in fields(obj):
        for ref in specs_in(getattr(obj, f.name)):
            refs_by_id.setdefault(ref.object_id, ref)
    return tuple(refs_by_id[object_id] for object_id in sorted(refs_by_id))


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
