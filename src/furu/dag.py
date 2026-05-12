from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from furu.dependencies import collect_declared_refs

if TYPE_CHECKING:
    from furu.core import Furu


@dataclass(frozen=True, kw_only=True)
class FuruDag[TFuru: Furu]:
    nodes: dict[str, TFuru]
    dependencies: dict[str, tuple[str, ...]]


def make_dag[TFuru: Furu](
    obj_or_objs: TFuru | Sequence[TFuru],
) -> FuruDag[TFuru]:
    from furu.core import Furu

    if isinstance(obj_or_objs, Furu):
        inputs: list[TFuru] = [cast("TFuru", obj_or_objs)]
    elif isinstance(obj_or_objs, Sequence):
        inputs = list(obj_or_objs)
        if any(not isinstance(obj, Furu) for obj in inputs):
            raise TypeError("make_dag() expected Furu objects")
    else:
        raise TypeError(
            "make_dag() expected a Furu object or a sequence of Furu objects"
        )

    nodes: dict[str, TFuru] = {}
    dependencies: dict[str, tuple[str, ...]] = {}
    pending: list[TFuru] = list(inputs)

    while pending:
        obj = pending.pop()
        if obj.object_id in nodes:
            continue
        nodes[obj.object_id] = obj
        if obj.status() == "completed":
            dependencies[obj.object_id] = ()
            continue
        refs = cast("tuple[TFuru, ...]", collect_declared_refs(obj))
        dependencies[obj.object_id] = tuple(ref.object_id for ref in refs)
        pending.extend(refs)

    return FuruDag(
        nodes=nodes,
        dependencies=dependencies,
    )
