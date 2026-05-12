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


def add_dag_dependency[TFuru: Furu](
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


def ensure_execution_dag_node[TFuru: Furu](
    obj: TFuru,
    nodes_by_id: dict[str, FuruDagNode[TFuru]],
) -> FuruDagNode[TFuru]:
    from furu.core import Furu

    if not isinstance(obj, Furu):
        raise TypeError("make_execution_dag() expected Furu objects")

    if existing := nodes_by_id.get(obj.object_id):
        return existing

    node = FuruDagNode(obj=obj)
    nodes_by_id[obj.object_id] = node

    if obj.status() == "completed":
        return node

    refs = cast("tuple[TFuru, ...]", collect_declared_refs(obj))
    for ref in refs:
        dep_node = ensure_execution_dag_node(ref, nodes_by_id)
        add_dag_dependency(node, dep_node)

    return node


def make_execution_dag[TFuru: Furu](
    objs: Sequence[TFuru],
) -> tuple[list[FuruDagNode[TFuru]], dict[str, FuruDagNode[TFuru]]]:
    nodes_by_id: dict[str, FuruDagNode[TFuru]] = {}
    for obj in objs:
        # TODO: accept pytrees of Furu objects (e.g. nested lists/dicts/dataclasses)
        # and flatten them before walking dependencies.
        # TODO: detect cycles and raise a clear error
        ensure_execution_dag_node(obj, nodes_by_id)

    zero_dependency_nodes = [
        node for node in nodes_by_id.values() if not node.dependencies
    ]
    return zero_dependency_nodes, nodes_by_id
