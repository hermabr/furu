from __future__ import annotations

import inspect
from collections.abc import Callable, Generator, Iterable, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, fields, is_dataclass, replace
from functools import cache, partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, overload

from furu import remember
from furu._declared_types import cached_type_hints
from furu._tree import specs_in
from furu.code_trace import (
    LogState,
    StaleCodeError,
    build_trace,
    paused,
    recording,
    tracing,
)
from furu.logging import get_logger
from furu.migration.schema import held
from furu.serializer.artifact import fingerprint
from furu.storage._layout import remembered_dir_in

if TYPE_CHECKING:
    from furu.core import Spec


class _Dependency[TSpec, T]:
    """A ``@furu.dependency`` method, until its Spec class turns it into a field.

    The field is filled from ``func`` the first time it is read, so what the
    method returns is hashed, stored and planned like any other field.
    """

    def __init__(self, func: Callable[[TSpec], T], remember: bool = False) -> None:
        self.func = func
        self.remember = remember

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
def dependency[TSpec: Spec, T](
    *, remember: bool = False
) -> Callable[[Callable[[TSpec], T]], _Dependency[TSpec, T]]: ...
def dependency[TSpec: Spec, T](
    func: Callable[[TSpec], T] | None = None, /, *, remember: bool = False
) -> _Dependency[TSpec, T] | Callable[[Callable[[TSpec], T]], _Dependency[TSpec, T]]:
    """Declare specs a create() uses that are computed from the spec's fields.

    The method becomes a field. furu calls it the first time the field is read,
    which naming the spec does: the ids it returns are part of the spec's id.

    With ``remember=True`` furu keeps a fingerprint of what the method returned
    and names the spec from it. The method is called again only when another
    field of the spec, code the method ran or a result it loaded has changed,
    or when something reads the field. It changes speed and nothing else, as
    long as the method reads nothing but those: furu does not notice a file, an
    environment variable or a web page that changes, nor whether a result is
    stored as ``.done`` or a caught ``furu.Missing`` told it, which it warns
    about. A spec the method uses that was already named or filled, and that
    no field this one was built with holds, directly or through another spec,
    has its own dependency methods called again in each such call, so that a
    change to their code is noticed too.
    """
    if func is None:
        return partial(_Dependency, remember=remember)
    return _Dependency(func, remember)


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

    The first read calls the method and keeps what it returned. A spec can be
    named without that: one built from a plan line has the fingerprint of what
    the method returned when the launch was planned, and one whose method has
    ``remember=True`` the fingerprint furu remembers. The first read then
    compares, and a result with another fingerprint stops it. Code tracing is
    paused for that call: the fingerprint already covers the result, and the
    spec gets the code version it gets where the method ran before create()
    started.
    """

    def __init__(self, func: Callable[[Any], object], remember: bool) -> None:
        self.func = func
        self.remember = remember

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def _fingerprint(self, obj: Spec, value: object) -> str:
        cls = type(obj)
        declared = cached_type_hints(cls).get(self.name, Any)
        return fingerprint(
            value, cls, self.name, declared, obj.artifact_serializers, held[cls]
        )

    def __get__(self, obj: Spec | None, owner: type | None = None) -> Any:
        if obj is None:
            return self
        if (copy := fresh(obj)) is not obj:
            return getattr(copy, self.name)
        if self.name in vars(obj):
            return vars(obj)[self.name]
        cls = type(obj)
        named = "_named" in vars(obj)
        if self.remember and not named and remembers():
            # Naming calls the method when nothing remembered holds, which
            # fills the field.
            named = bool(obj._named)
            if self.name in vars(obj):
                return vars(obj)[self.name]
        if named:
            with creating(on_load=_ignore_loads), paused():
                value = self.func(obj)
                found = self._fingerprint(obj, value)
            if found != (known := obj._returned[self.name]):
                label = f"{obj.object_id}.{self.name}"
                if not self.remember:
                    raise RuntimeError(
                        f"{label} returned something else than when the launch "
                        f"was planned (fingerprint {found}, planned {known}). A "
                        "@furu.dependency method has to return the same specs "
                        "in the same order for as long as what it reads is "
                        "unchanged."
                    )
                raise RuntimeError(
                    f"{label} returned something else than furu remembers "
                    f"(fingerprint {found}, remembered {known}), though the "
                    "spec's fields, the code the method ran and the results "
                    "it loaded are unchanged. A method with remember=True "
                    "reads something furu does not check, such as a file, an "
                    "environment variable or a web page: put what it reads "
                    "in a field of the spec, or take remember=True off. "
                    f"Deleting {remembered_dir_in(obj._class_dir)} makes furu "
                    "call the method again."
                )
        elif tracing() and any_under_creation():
            creator = min(creation().specs).partition(":")[0]
            raise RuntimeError(
                f"the create() of {creator} traces its code and built a "
                f"{cls.__name__}, whose @furu.dependency method {self.name} "
                "would run as part of it. furu does not trace a dependency "
                f"method into the create() that builds its spec yet: turn code "
                f"tracing off for {creator}, or declare the {cls.__name__} as "
                "a field or dependency of it."
            )
        else:
            value = self.func(obj)
        vars(obj)[self.name] = value
        return value

    def __set__(self, obj: Spec, value: object) -> None:
        # A data descriptor, so that every read goes through __get__.
        vars(obj)[self.name] = value

    def remembered(self, obj: Spec, name: str, fields: str) -> str:
        """The fingerprint of what the method returns for ``obj``, for its id.

        ``fields`` is a fingerprint of the spec's other fields and ``name``
        what the id calls this one. The method is called only when no
        remembered line holds. That call is traced as a create() is, together
        with naming what it returned, which calls the dependency methods of
        the specs in it, on a copy of each that was named or filled before
        (`fresh`). It fills the field and is remembered.
        """
        class_dir, storage = obj._class_dir, obj._metadata.storage
        if (known := remember.lookup(class_dir, name, fields, storage)) is not None:
            return known
        with (
            recording(None, storage_root=storage, resumed=LogState()) as run,
            creating(on_load=lambda loaded, specs: run.loaded(loaded)),
        ):
            # ``fields`` covers the spec and the specs its fields hold, directly
            # or through another spec: no copies.
            met, pending = {}, [obj]
            while pending:
                if id(spec := pending.pop()) not in met:
                    met[id(spec)] = spec, spec
                    pending += specs_in(_field_values(spec))
            token, checks = _met.set(met), _checked.set(run.checked)
            try:
                value = vars(obj)[self.name] = self.func(obj)
                found = self._fingerprint(obj, value)
            finally:
                _met.reset(token)
                _checked.reset(checks)
        try:
            trace = build_trace(
                run,
                label=f"{type(obj).__name__}.{self.name}",
                warn=get_logger().warning,
            )
        except StaleCodeError:
            # This process runs older code than the files hold, so the next
            # launch calls the method again.
            return found
        remember.add(class_dir, name, fields, found, trace)
        return found


# Inside the traced call of a ``remember=True`` method: each spec the call has
# met, by identity, with the spec that stands in for it there.
_met: ContextVar[dict[int, tuple[Spec, Spec]] | None] = ContextVar("_met", default=None)
# And what the call checked that its trace does not see.
_checked: ContextVar[set[str] | None] = ContextVar("_checked", default=None)


def checked(what: str) -> None:
    """Note that the traced ``remember=True`` call this runs in, if any, did ``what``."""
    if (checks := _checked.get()) is not None and not any_under_creation():
        checks.add(what)


def fresh(obj: Spec) -> Spec:
    """``obj``, or the copy that stands in for it in the traced call of a ``remember=True`` method.

    A spec that was named, or had a dependency list filled, before the call
    would keep the code that did so out of the call's trace. The call names
    and reads a copy of it instead, which runs its dependency methods again:
    an object of the same class that has the values of the other fields and
    nothing else.
    A create() the call runs is traced on its own and gets the spec itself.
    """
    if (met := _met.get()) is None or any_under_creation():
        return obj
    if id(obj) not in met:
        copy = obj
        if vars(obj).keys() & {"_named", *returned_attributes(type(obj))}:
            # Without __init__: a __post_init__ can take a value no field keeps.
            copy = object.__new__(type(obj))
            vars(copy).update(_field_values(obj))
        met[id(obj)] = obj, copy
    return met[id(obj)][1]


def _field_values(obj: Spec) -> dict[str, Any]:
    """The values of the fields of ``obj`` that no dependency method fills."""
    names = {f.name for f in fields(obj)} - returned_attributes(type(obj))
    return {k: v for k, v in vars(obj).items() if k in names}


def remembers() -> bool:
    """Whether a spec named here takes what furu remembers for a ``remember=True`` method.

    Not inside a create(), where a dependency list is what its method returns,
    and not inside the traced call of another such method, whose trace has to
    cover this one's code.
    """
    return not tracing() and not any_under_creation()


@cache
def returned_attributes(cls: type) -> frozenset[str]:
    """The fields of a spec class that its ``@furu.dependency`` methods fill."""
    assert is_dataclass(cls)
    return frozenset(
        f.name
        for f in fields(cls)
        if isinstance(inspect.getattr_static(cls, f.name, None), Returned)
    )


@cache
def remembered_methods(cls: type) -> dict[str, Returned]:
    """The ``remember=True`` methods of a spec class, by the field each fills."""
    methods = {
        name: inspect.getattr_static(cls, name) for name in returned_attributes(cls)
    }
    return {name: method for name, method in methods.items() if method.remember}


type OnLoad = Callable[[Path, Iterable[Spec]], None]


def _ignore_loads(run: Path, specs: Iterable[Spec]) -> None:
    pass


@dataclass(frozen=True)
class Creation:
    """What code that runs in this context knows of the create() it runs in.

    ``specs`` are the ids of the specs whose create() runs, ``plan`` the plan
    file that lists them and ``asked`` the stored names of the classes their
    launch asked about. ``on_load`` is told about each published run the
    create(), or the traced call of a ``remember=True`` method, loads, and the
    specs whose rows it loads from it.
    """

    attempt_dir: Path | None = None
    specs: frozenset[str] = frozenset()
    plan: Path | None = None
    asked: frozenset[str] = frozenset()
    on_load: OnLoad = _ignore_loads


# TODO: a thread a create() starts sees Creation(), as if no create() ran
# (ContextVar state doesn't reach it): its loads aren't recorded, a create() in it
# starts a launch instead of blocking, and .directory raises. Workaround: submit
# the work with contextvars.copy_context().run. Check with a human before adding
# such threads.
_creation: ContextVar[Creation] = ContextVar(
    "furu_creation",
    default=Creation(),  # noqa: B039 -- frozen
)


def creation() -> Creation:
    return _creation.get()


@contextmanager
def creating(
    *,
    attempt_dir: Path | None = None,
    specs: frozenset[str] | None = None,
    plan: Path | None = None,
    asked: frozenset[str] | None = None,
    on_load: OnLoad | None = None,
) -> Generator[None]:
    """Run code with ``creation()`` changed in each field that is given."""
    given = {
        "attempt_dir": attempt_dir,
        "specs": specs,
        "plan": plan,
        "asked": asked,
        "on_load": on_load,
    }
    changes = {name: value for name, value in given.items() if value is not None}
    token = _creation.set(replace(_creation.get(), **changes))
    try:
        yield
    finally:
        _creation.reset(token)


def record_loaded(run: Path, specs: Iterable[Spec]) -> None:
    creation().on_load(run, specs)


def any_under_creation() -> bool:
    return bool(creation().specs)


class _DependencyNotReady(BaseException):
    """create() inside a create hook found missing results.

    A BaseException so user ``except Exception`` blocks don't swallow it. The
    runner builds ``dependencies``, then reruns the blocked create from scratch.
    ``lines`` describe them and are saved in ``plan``.
    """

    def __init__(
        self,
        dependencies: Sequence[Spec],
        plan: Path,
        lines: Sequence[dict[str, Any]] = (),
    ):
        self.dependencies = tuple(dependencies)
        self.plan = plan
        self.lines = list(lines)
        super().__init__(missing_dependencies(len(self.dependencies)))


def missing_dependencies(count: int) -> str:
    return f"{count} missing {'dependency' if count == 1 else 'dependencies'}"
