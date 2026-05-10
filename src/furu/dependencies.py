from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict

from furu.utils import fully_qualified_name

if TYPE_CHECKING:
    from furu.core import Furu


type DependencyVia = Literal["field", "dependency", "load_or_create", "try_load"]


_DEPENDENCY_MARKER = "__furu_dependency__"


class DependencyRef(PydanticBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    object_id: str
    class_name: str
    data_path: Path
    artifact_hash: str
    artifact_schema_hash: str
    via: DependencyVia
    path: str | None = None


class DependencyMetadata(PydanticBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    eager: tuple[DependencyRef, ...] = ()
    lazy: tuple[DependencyRef, ...] = ()


class dependency[T](cached_property):
    def __init__(self, func: Callable[[Any], T]) -> None:
        super().__init__(func)
        setattr(self, _DEPENDENCY_MARKER, True)


def _is_dependency_descriptor(value: object) -> bool:
    return getattr(value, _DEPENDENCY_MARKER, False) is True


def _iter_dependency_descriptors(cls: type) -> Iterator[tuple[str, Any]]:
    descriptors: dict[str, Any] = {}
    shadowed: set[str] = set()
    for base in cls.__mro__:
        for name, value in base.__dict__.items():
            if name in shadowed or name in descriptors:
                continue
            if _is_dependency_descriptor(value):
                descriptors[name] = value
            else:
                shadowed.add(name)
    yield from descriptors.items()


def _find_nested_furu_objects(
    value: object, *, path: str
) -> Iterator[tuple[Furu, str]]:
    from furu.core import Furu

    if isinstance(value, Furu):
        yield value, path
        return
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from _find_nested_furu_objects(
                item, path=f"{path}[{i}]" if path else f"[{i}]"
            )
        return
    if isinstance(value, (set, frozenset)):
        for i, item in enumerate(sorted(value, key=repr)):
            yield from _find_nested_furu_objects(
                item, path=f"{path}[{i}]" if path else f"[{i}]"
            )
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _find_nested_furu_objects(
                item, path=f"{path}[{key!r}]" if path else f"[{key!r}]"
            )
        return
    if isinstance(value, PydanticBaseModel):
        for field_name in type(value).model_fields:
            yield from _find_nested_furu_objects(
                getattr(value, field_name),
                path=f"{path}.{field_name}" if path else field_name,
            )
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from _find_nested_furu_objects(
                getattr(value, field.name),
                path=f"{path}.{field.name}" if path else field.name,
            )
        return


def _make_ref(furu_obj: Furu, *, via: DependencyVia, path: str | None) -> DependencyRef:
    return DependencyRef(
        object_id=furu_obj.object_id,
        class_name=fully_qualified_name(type(furu_obj)),
        data_path=furu_obj.data_dir,
        artifact_hash=furu_obj.artifact_hash,
        artifact_schema_hash=furu_obj.artifact_schema_hash,
        via=via,
        path=path,
    )


def collect_eager_dependencies(obj: Furu) -> list[DependencyRef]:
    refs: list[DependencyRef] = []
    seen_ids: set[str] = set()

    def add(furu_obj: Furu, *, via: DependencyVia, path: str) -> None:
        if furu_obj.object_id in seen_ids:
            return
        seen_ids.add(furu_obj.object_id)
        refs.append(_make_ref(furu_obj, via=via, path=path))

    for field in dataclasses.fields(type(obj)):
        value = getattr(obj, field.name)
        for furu_obj, found_path in _find_nested_furu_objects(value, path=field.name):
            add(furu_obj, via="field", path=found_path)

    for name, _descriptor in _iter_dependency_descriptors(type(obj)):
        value = getattr(obj, name)
        for furu_obj, found_path in _find_nested_furu_objects(value, path=name):
            add(furu_obj, via="dependency", path=found_path)

    return refs


class _DependencyRecorder:
    def __init__(self) -> None:
        self._observed: list[tuple[str, DependencyVia, Furu]] = []

    def record(self, obj: Furu, *, via: DependencyVia) -> None:
        self._observed.append((obj.object_id, via, obj))

    def lazy_refs(self, *, exclude_object_ids: set[str]) -> list[DependencyRef]:
        seen: set[str] = set()
        out: list[DependencyRef] = []
        for object_id, via, obj in self._observed:
            if object_id in exclude_object_ids or object_id in seen:
                continue
            seen.add(object_id)
            out.append(_make_ref(obj, via=via, path=None))
        return out


_active_recorder: ContextVar[_DependencyRecorder | None] = ContextVar(
    "furu_active_dependency_recorder",
    default=None,
)


def _record_call(obj: Furu, *, via: DependencyVia) -> None:
    recorder = _active_recorder.get()
    if recorder is not None:
        recorder.record(obj, via=via)


@contextmanager
def _scoped_recorder() -> Iterator[_DependencyRecorder]:
    recorder = _DependencyRecorder()
    token = _active_recorder.set(recorder)
    try:
        yield recorder
    finally:
        _active_recorder.reset(token)
