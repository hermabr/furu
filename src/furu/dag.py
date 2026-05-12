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


def add_dependency[TFuru: Furu](
    node: FuruDagNode[TFuru],
    dependency: FuruDagNode[TFuru],
) -> None:
    if all(
        existing.obj.object_id != dependency.obj.object_id
        for existing in node.dependencies
    ):
        node.dependencies.append(dependency)
    if all(
        existing.obj.object_id != node.obj.object_id
        for existing in dependency.dependents
    ):
        dependency.dependents.append(node)


def add_execution_dag_nodes[TFuru: Furu](
    objs: Sequence[TFuru],
    nodes_by_id: dict[str, FuruDagNode[TFuru]],
    *,
    type_error_message: str = "add_execution_dag_nodes() expected Furu objects",
) -> list[FuruDagNode[TFuru]]:
    from furu.core import Furu

    if any(not isinstance(obj, Furu) for obj in objs):
        raise TypeError(type_error_message)

    added_nodes: list[FuruDagNode[TFuru]] = []
    refs_by_id: dict[str, tuple[TFuru, ...]] = {}
    # TODO: detect cycles and raise a clear error
    pending: list[TFuru] = list(objs)

    while pending:
        obj = pending.pop()
        if obj.object_id in nodes_by_id:
            continue
        node = FuruDagNode(obj=obj)
        nodes_by_id[obj.object_id] = node
        added_nodes.append(node)
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
            add_dependency(node, dep_node)

    return added_nodes


def make_execution_dag[TFuru: Furu](
    objs: Sequence[TFuru],
) -> tuple[list[FuruDagNode[TFuru]], dict[str, FuruDagNode[TFuru]]]:
    nodes_by_id: dict[str, FuruDagNode[TFuru]] = {}
    add_execution_dag_nodes(
        objs,
        nodes_by_id,
        type_error_message="make_execution_dag() expected Furu objects",
    )

    zero_dependency_nodes = [
        node for node in nodes_by_id.values() if not node.dependencies
    ]
    return zero_dependency_nodes, nodes_by_id
