from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, fields, is_dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence, overload

from pydantic import BaseModel as PydanticBaseModel

if TYPE_CHECKING:
    from furu.core import Furu


class _CachedDependency[T](cached_property):
    __furu_dependency__ = True


class _UncachedDependency[T](property):
    __furu_dependency__ = True


@overload
def dependency[T](func: Callable[[Any], T], /) -> _CachedDependency[T]: ...


@overload
def dependency[T](
    *, cached: Literal[True] = True
) -> Callable[[Callable[[Any], T]], _CachedDependency[T]]: ...


@overload
def dependency[T](
    *, cached: Literal[False]
) -> Callable[[Callable[[Any], T]], _UncachedDependency[T]]: ...


@overload
def dependency[T](
    *, cached: bool
) -> Callable[[Callable[[Any], T]], _CachedDependency[T] | _UncachedDependency[T]]: ...


def dependency[T](
    func: Callable[[Any], T] | None = None, /, *, cached: bool = True
) -> (
    _CachedDependency[T]
    | _UncachedDependency[T]
    | Callable[[Callable[[Any], T]], _CachedDependency[T] | _UncachedDependency[T]]
):
    def decorate(
        func: Callable[[Any], T],
    ) -> _CachedDependency[T] | _UncachedDependency[T]:
        if cached:
            return _CachedDependency(func)
        return _UncachedDependency(func)

    if func is not None:
        return decorate(func)
    return decorate


def find_nested_furu_objects(value: object) -> Iterator[Furu[Any]]:
    from furu.core import Furu

    match value:
        case Furu():
            yield value
        case _ if is_dataclass(value) and not isinstance(value, type):
            for dataclass_field in fields(value):
                yield from find_nested_furu_objects(
                    getattr(value, dataclass_field.name)
                )
        case PydanticBaseModel():
            for name in type(value).model_fields:
                yield from find_nested_furu_objects(getattr(value, name))
        case tuple() | list() | set() | frozenset():
            for item in value:
                yield from find_nested_furu_objects(item)
        case dict():
            for item in value.values():
                yield from find_nested_furu_objects(item)


def collect_declared_refs(obj: Furu[Any]) -> tuple[Furu[Any], ...]:
    refs_by_id: dict[str, Furu[Any]] = {}

    for dataclass_field in fields(obj):
        for ref in find_nested_furu_objects(getattr(obj, dataclass_field.name)):
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


@dataclass(frozen=True, eq=False)
class FuruDependencyNode[TFuru: Furu]:
    obj: TFuru
    dependencies: tuple[FuruDependencyNode[Furu], ...] = field(
        default=(),
        repr=False,
    )
    dependents: tuple[FuruDependencyNode[Furu], ...] = field(
        default=(),
        repr=False,
    )


@overload
def _normalize_topological_sort_input[TFuru: Furu](
    obj_or_objs: TFuru,
) -> list[TFuru]: ...


@overload
def _normalize_topological_sort_input[TFuru: Furu](
    obj_or_objs: Sequence[TFuru],
) -> list[TFuru]: ...


def _normalize_topological_sort_input(
    obj_or_objs: Furu | Sequence[Furu],
) -> list[Furu]:
    from furu.core import Furu

    if isinstance(obj_or_objs, Furu):
        return [obj_or_objs]
    if not isinstance(obj_or_objs, Sequence):
        raise TypeError(
            "make_topological_sort() expected a Furu object or a sequence of "
            "Furu objects"
        )

    objs = list(obj_or_objs)
    if any(not isinstance(obj, Furu) for obj in objs):
        raise TypeError("make_topological_sort() expected Furu objects")
    return objs


def _is_success_status[TFuru: Furu](obj: TFuru) -> bool:
    return obj.status() == "completed"


@overload
def make_topological_sort[TFuru: Furu](
    obj_or_objs: TFuru,
) -> list[FuruDependencyNode[Furu]]: ...


@overload
def make_topological_sort[TFuru: Furu](
    obj_or_objs: Sequence[TFuru],
) -> list[FuruDependencyNode[Furu]]: ...


def make_topological_sort(
    obj_or_objs: Furu | Sequence[Furu],
) -> list[FuruDependencyNode[Furu]]:
    objs = _normalize_topological_sort_input(obj_or_objs)

    nodes_by_id: dict[str, FuruDependencyNode[Furu]] = {}
    dependency_ids_by_id: dict[str, tuple[str, ...]] = {}
    dependent_ids_by_id: dict[str, list[str]] = {}
    visiting: set[str] = set()

    def get_node(obj: Furu) -> FuruDependencyNode[Furu]:
        obj_id = obj.object_id
        if obj_id in nodes_by_id:
            return nodes_by_id[obj_id]
        node = FuruDependencyNode(obj=obj)
        nodes_by_id[obj_id] = node
        return node

    def visit(obj: Furu) -> None:
        obj_id = obj.object_id
        get_node(obj)
        if obj_id in visiting:
            raise ValueError(f"declared Furu dependencies contain a cycle at {obj_id}")
        if obj_id in dependency_ids_by_id:
            return

        visiting.add(obj_id)
        refs = () if _is_success_status(obj) else collect_declared_refs(obj)
        for ref in refs:
            visit(ref)
            dependent_ids_by_id.setdefault(ref.object_id, []).append(obj_id)
        dependency_ids_by_id[obj_id] = tuple(ref.object_id for ref in refs)
        visiting.remove(obj_id)

    for obj in objs:
        visit(obj)

    for obj_id, node in nodes_by_id.items():
        object.__setattr__(
            node,
            "dependencies",
            tuple(
                nodes_by_id[dependency_id]
                for dependency_id in dependency_ids_by_id[obj_id]
            ),
        )
        object.__setattr__(
            node,
            "dependents",
            tuple(
                nodes_by_id[dependent_id]
                for dependent_id in dependent_ids_by_id.get(obj_id, [])
            ),
        )

    return [node for node in nodes_by_id.values() if not node.dependencies]


class DependencyRecorder:
    def __init__(self) -> None:
        self._observed_ids: set[str] = set()

    def record(self, obj: Furu[Any]) -> None:
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


def record_dependency_call(obj: Furu[Any]) -> None:
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
