from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from furu.core import Furu
from furu.dependencies import collect_declared_refs

if TYPE_CHECKING:
    from furu.execution.manager import Manager


@dataclass(eq=False)
class DagNode[TFuru: Furu]:
    obj: TFuru
    dependencies: list[DagNode[TFuru]] = field(default_factory=list)
    dependents: list[DagNode[TFuru]] = field(default_factory=list)


def _add_to_dag(manager: Manager, objs: Sequence[Furu[Any]]) -> None:
    if any(not isinstance(obj, Furu) for obj in objs):
        # TODO: accept pytrees of Furu objects (e.g. nested lists/dicts/dataclasses)
        # and flatten them before walking dependencies.
        raise TypeError("expected Furu objects")

    refs_by_id: dict[str, tuple[Furu[Any], ...]] = {}
    newly_added: list[DagNode[Furu[Any]]] = []
    # TODO: detect cycles and raise a clear error
    pending = list(objs)

    while pending:
        obj = pending.pop()
        if obj.object_id in manager.nodes_by_id:
            continue
        node = DagNode(obj=obj)
        manager.nodes_by_id[obj.object_id] = node
        newly_added.append(node)
        if obj.status() == "completed":
            refs_by_id[obj.object_id] = ()
            continue
        refs = collect_declared_refs(obj)
        refs_by_id[obj.object_id] = refs
        pending.extend(refs)

    for obj_id, refs in refs_by_id.items():
        node = manager.nodes_by_id[obj_id]
        for ref in refs:
            dep_node = manager.nodes_by_id[ref.object_id]
            node.dependencies.append(dep_node)
            dep_node.dependents.append(node)

    for node in newly_added:
        target = manager.ready if not node.dependencies else manager.blocked
        target[node.obj.object_id] = node
