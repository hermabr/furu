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
    from furu.metadata import DependencyRef, DependencyVia


class dependency[T](cached_property):
    __furu_dependency__ = True

    def __init__(self, func: Callable[..., T]):
        super().__init__(func)
        self.__name__ = getattr(func, "__name__", type(func).__name__)
        self.__doc__ = getattr(func, "__doc__", None)


def find_nested_furu_objects(
    value: object, *, path: str | None = None
) -> Iterator[tuple[Furu[Any], str | None]]:
    from furu.core import Furu

    match value:
        case Furu():
            yield value, path
        case _ if is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                yield from find_nested_furu_objects(
                    getattr(value, field.name),
                    path=f"{path}.{field.name}" if path else field.name,
                )
        case PydanticBaseModel():
            for name in type(value).model_fields:
                yield from find_nested_furu_objects(
                    getattr(value, name),
                    path=f"{path}.{name}" if path else name,
                )
        case tuple() | list() | set() | frozenset():
            for index, item in enumerate(value):
                item_path = f"[{index}]"
                yield from find_nested_furu_objects(
                    item,
                    path=f"{path}[{index}]" if path else item_path,
                )
        case dict():
            for key, item in value.items():
                item_path = f"[{key!r}]"
                yield from find_nested_furu_objects(
                    item,
                    path=f"{path}[{key!r}]" if path else item_path,
                )


def collect_eager_dependencies(obj: Furu[Any]) -> tuple[DependencyRef, ...]:
    from furu.metadata import DependencyRef

    refs: list[DependencyRef] = []

    for field in fields(obj):
        refs.extend(
            DependencyRef.from_furu(dep, via="field", path=dep_path)
            for dep, dep_path in find_nested_furu_objects(
                getattr(obj, field.name), path=field.name
            )
        )

    for base in reversed(type(obj).__mro__):
        for name, value in base.__dict__.items():
            if getattr(value, "__furu_dependency__", False):
                refs.extend(
                    DependencyRef.from_furu(dep, via="dependency", path=dep_path)
                    for dep, dep_path in find_nested_furu_objects(
                        getattr(obj, name), path=name
                    )
                )

    return normalize_dependency_refs(refs)


def normalize_dependency_refs(
    refs: list[DependencyRef] | tuple[DependencyRef, ...],
) -> tuple[DependencyRef, ...]:
    by_id: dict[str, DependencyRef] = {}
    for ref in refs:
        by_id.setdefault(ref.object_id, ref)
    return tuple(sorted(by_id.values(), key=lambda ref: ref.object_id))


class DependencyRecorder:
    def __init__(self) -> None:
        self._observed_by_id: dict[str, DependencyRef] = {}

    def record(self, obj: Furu[Any], *, via: DependencyVia) -> None:
        from furu.metadata import DependencyRef

        ref = DependencyRef.from_furu(obj, via=via)
        self._observed_by_id.setdefault(ref.object_id, ref)

    @property
    def observed(self) -> tuple[DependencyRef, ...]:
        return tuple(
            sorted(self._observed_by_id.values(), key=lambda ref: ref.object_id)
        )


_active_dependency_recorder: ContextVar[DependencyRecorder | None] = ContextVar(
    "_active_dependency_recorder",
    default=None,
)


def record_dependency_call(obj: Furu[Any], *, via: DependencyVia) -> None:
    recorder = _active_dependency_recorder.get()
    if recorder is not None:
        recorder.record(obj, via=via)


@contextmanager
def dependency_recorder() -> Iterator[DependencyRecorder]:
    recorder = DependencyRecorder()
    token = _active_dependency_recorder.set(recorder)
    try:
        yield recorder
    finally:
        _active_dependency_recorder.reset(token)


def resolve_dependencies(
    *,
    eager: tuple[DependencyRef, ...],
    observed: tuple[DependencyRef, ...],
) -> tuple[DependencyRef, ...]:
    eager_ids = {ref.object_id for ref in eager}
    return normalize_dependency_refs(
        tuple(ref for ref in observed if ref.object_id not in eager_ids)
    )
