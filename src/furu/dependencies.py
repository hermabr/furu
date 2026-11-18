from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, overload

from furu._tree import specs_in

if TYPE_CHECKING:
    from furu.core import Spec


class _Dependency[TSpec, T]:
    """A ``@furu.dependency`` method, until its Spec class turns it into a field.

    The field is filled from ``func`` the first time it is read, so what the
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

    The method becomes a field. furu calls it the first time the field is read,
    which naming the spec does: the ids it returns are part of the spec's id.
    """
    return dependency if func is None else _Dependency(func)


def dependency_field(cls: type, name: str, declared: _Dependency) -> Any:
    """The dataclass field ``cls.name`` becomes; its type is the return annotation."""
    if "return" not in declared.func.__annotations__:
        raise TypeError(
            f"{cls.__name__}.{name}: @furu.dependency needs a return annotation; "
            "it is the type of the field furu stores."
        )
    # Left out of repr and equality, which would call the method; it only
    # restates the other fields.
    return field(init=False, repr=False, compare=False)


def collect_declared_refs(obj: Spec) -> tuple[Spec, ...]:
    refs_by_id: dict[str, Spec] = {}
    for f in fields(obj):
        for ref in specs_in(getattr(obj, f.name)):
            refs_by_id.setdefault(ref.object_id, ref)
    return tuple(refs_by_id[object_id] for object_id in sorted(refs_by_id))


# What to tell about each published run the create() in this context loads.
# TODO: results loaded in threads a create() starts aren't recorded (ContextVar
# state doesn't reach them). Workaround: submit the work with
# contextvars.copy_context().run. Check with a human before adding such threads.
_on_load: ContextVar[Callable[[Path], None] | None] = ContextVar(
    "_on_load", default=None
)


def record_loaded(run: Path) -> None:
    if (on_load := _on_load.get()) is not None:
        on_load(run)


@contextmanager
def recording_loads(on_load: Callable[[Path], None]) -> Generator[None]:
    token = _on_load.set(on_load)
    try:
        yield
    finally:
        _on_load.reset(token)


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
