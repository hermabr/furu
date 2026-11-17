from __future__ import annotations

import inspect
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import field, fields, is_dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, overload

from furu._declared_types import cached_type_hints
from furu._tree import specs_in
from furu.code_trace import paused, tracing
from furu.migration.schema import held
from furu.serializer.artifact import to_json
from furu.utils import _hash_dict_deterministically

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


class Returned:
    """The field a ``@furu.dependency`` method fills, on its class.

    The first read calls the method and keeps what it returned. A spec built
    from a plan line has the fingerprint of what the method returned when the
    launch was planned, and a result with another fingerprint stops the read.
    Code tracing is paused for that call: the fingerprint already covers the
    result, and the spec gets the code version it gets where the method ran
    before create() started.
    """

    def __init__(self, func: Callable[[Any], object]) -> None:
        self.func = func

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, obj: Spec | None, owner: type | None = None) -> Any:
        if obj is None:
            return self
        cls = type(obj)
        if obj._from_line:
            loads = _on_load.set(None)
            try:
                with paused():
                    value = self.func(obj)
                    fingerprint = _hash_dict_deterministically(
                        to_json(
                            value,
                            cached_type_hints(cls).get(self.name, Any),
                            obj.artifact_serializers,
                            held[cls],
                        )
                    )
            finally:
                _on_load.reset(loads)
            if fingerprint != (planned := obj._returned[self.name]):
                raise RuntimeError(
                    f"{obj._log_label}.{self.name} returned something else than "
                    f"when the launch was planned (fingerprint {fingerprint}, "
                    f"planned {planned}). A @furu.dependency method has to "
                    "return the same specs in the same order for as long as "
                    "what it reads is unchanged."
                )
        elif tracing():
            creating = min(specs_under_creation()).partition(":")[0]
            raise RuntimeError(
                f"the create() of {creating} traces its code and built a "
                f"{cls.__name__}, whose @furu.dependency method {self.name} "
                "would run as part of it. furu does not trace a dependency "
                f"method into the create() that builds its spec yet: turn code "
                f"tracing off for {creating}, or declare the {cls.__name__} as "
                "a field or dependency of it."
            )
        else:
            value = self.func(obj)
        vars(obj)[self.name] = value
        return value


@cache
def returned_attributes(cls: type) -> frozenset[str]:
    """The fields of a spec class that its ``@furu.dependency`` methods fill."""
    assert is_dataclass(cls)
    return frozenset(
        f.name
        for f in fields(cls)
        if isinstance(inspect.getattr_static(cls, f.name, None), Returned)
    )


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


def specs_under_creation() -> frozenset[str]:
    """The ids of the specs whose create() runs in this context."""
    return _specs_under_creation.get()


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
    ``lines`` describe them and are saved in ``plan``.
    """

    def __init__(
        self,
        dependencies: Sequence[Spec],
        plan: Path | None = None,
        lines: Sequence[dict[str, Any]] = (),
    ):
        self.dependencies = tuple(dependencies)
        self.plan = plan
        self.lines = list(lines)
        super().__init__(missing_dependencies(len(self.dependencies)))


def missing_dependencies(count: int) -> str:
    return f"{count} missing {'dependency' if count == 1 else 'dependencies'}"
