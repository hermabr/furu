from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from furu.dependencies import collect_declared_refs

if TYPE_CHECKING:
    from furu.core import Furu


@dataclass(frozen=True, kw_only=True)
class FuruDag:
    nodes: dict[str, Furu[Any]]
    dependencies: dict[str, tuple[str, ...]]
    roots: tuple[str, ...]


def make_dag(obj_or_objs: Furu[Any] | Sequence[Furu[Any]]) -> FuruDag:
    from furu.core import Furu

    if isinstance(obj_or_objs, Furu):
        inputs: list[Furu[Any]] = [obj_or_objs]
    elif isinstance(obj_or_objs, Sequence):
        inputs = list(obj_or_objs)
        if any(not isinstance(obj, Furu) for obj in inputs):
            raise TypeError("make_dag() expected Furu objects")
    else:
        raise TypeError(
            "make_dag() expected a Furu object or a sequence of Furu objects"
        )

    roots: dict[str, None] = {}
    for obj in inputs:
        roots.setdefault(obj.object_id, None)

    nodes: dict[str, Furu[Any]] = {}
    dependencies: dict[str, tuple[str, ...]] = {}
    pending: list[Furu[Any]] = list(inputs)

    while pending:
        obj = pending.pop()
        if obj.object_id in nodes:
            continue
        nodes[obj.object_id] = obj
        if obj.status() == "completed":
            dependencies[obj.object_id] = ()
            continue
        refs = collect_declared_refs(obj)
        dependencies[obj.object_id] = tuple(ref.object_id for ref in refs)
        pending.extend(refs)

    return FuruDag(
        nodes=nodes,
        dependencies=dependencies,
        roots=tuple(roots),
    )
