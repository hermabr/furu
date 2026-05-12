from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from furu.dependencies import collect_declared_refs

if TYPE_CHECKING:
    from furu.core import Furu


@dataclass(eq=False)
class FuruDagNode[TFuru: Furu]:
    obj: TFuru
    dependencies: list[FuruDagNode[TFuru]] = field(default_factory=list)
    dependents: list[FuruDagNode[TFuru]] = field(default_factory=list)


def make_execution_dag[TFuru: Furu](
    obj_or_objs: TFuru | Sequence[TFuru],
) -> list[FuruDagNode[TFuru]]:
    from furu.core import Furu

    if isinstance(obj_or_objs, Furu):
        inputs: list[TFuru] = [cast("TFuru", obj_or_objs)]
    elif isinstance(obj_or_objs, Sequence):
        inputs = list(obj_or_objs)
        if any(not isinstance(obj, Furu) for obj in inputs):
            raise TypeError("make_execution_dag() expected Furu objects")
    else:
        raise TypeError(
            "make_execution_dag() expected a Furu object or a sequence of Furu objects"
        )

    nodes_by_id: dict[str, FuruDagNode[TFuru]] = {}
    refs_by_id: dict[str, tuple[TFuru, ...]] = {}
    pending: list[TFuru] = list(inputs)

    while pending:
        obj = pending.pop()
        if obj.object_id in nodes_by_id:
            continue
        nodes_by_id[obj.object_id] = FuruDagNode(obj=obj)
        if obj.status() == "completed":
            refs_by_id[obj.object_id] = ()
            continue
        refs = cast("tuple[TFuru, ...]", collect_declared_refs(obj))
        refs_by_id[obj.object_id] = refs
        pending.extend(refs)

    for obj_id, refs in refs_by_id.items():
        node = nodes_by_id[obj_id]
        for ref in refs:
            dep_node = nodes_by_id[ref.object_id]
            node.dependencies.append(dep_node)
            dep_node.dependents.append(node)

    return [node for node in nodes_by_id.values() if not node.dependencies]
