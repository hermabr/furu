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


def _extend_execution_dag[TFuru: Furu](
    objs: Sequence[TFuru],
    nodes_by_id: dict[str, FuruDagNode[TFuru]],
) -> list[FuruDagNode[TFuru]]:
    from furu.core import Furu

    if any(not isinstance(obj, Furu) for obj in objs):
        # TODO: accept pytrees of Furu objects (e.g. nested lists/dicts/dataclasses)
        # and flatten them before walking dependencies.
        raise TypeError("expected Furu objects")

    refs_by_id: dict[str, tuple[TFuru, ...]] = {}
    newly_added: list[FuruDagNode[TFuru]] = []
    # TODO: detect cycles and raise a clear error
    pending: list[TFuru] = list(objs)

    while pending:
        obj = pending.pop()
        if obj.object_id in nodes_by_id:
            continue
        node = FuruDagNode(obj=obj)
        nodes_by_id[obj.object_id] = node
        newly_added.append(node)
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

    return [node for node in newly_added if not node.dependencies]


def make_execution_dag[TFuru: Furu](
    objs: Sequence[TFuru],
) -> tuple[list[FuruDagNode[TFuru]], dict[str, FuruDagNode[TFuru]]]:
    nodes_by_id: dict[str, FuruDagNode[TFuru]] = {}
    zero_dependency_nodes = _extend_execution_dag(objs, nodes_by_id)
    return zero_dependency_nodes, nodes_by_id
