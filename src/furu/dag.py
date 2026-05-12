from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from furu.dependencies import collect_declared_refs
from furu.worker_execution import _DependencyNotReady, worker_execution_context

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


def submit(objs: Sequence[Furu[Any]]) -> None:
    from furu.execution import _load_or_create_local

    zero_dependency_nodes, nodes_by_id = make_execution_dag(objs)

    while zero_dependency_nodes:
        node = zero_dependency_nodes.pop(0)
        try:
            with worker_execution_context(lease_id=uuid.uuid4().hex):
                _load_or_create_local(node.obj)
        except _DependencyNotReady as exc:
            new_objs = [
                dep for dep in exc.dependencies if dep.object_id not in nodes_by_id
            ]
            zero_dependency_nodes.extend(_extend_execution_dag(new_objs, nodes_by_id))
            for lazy_dep in exc.dependencies:
                dep_node = nodes_by_id[lazy_dep.object_id]
                if dep_node not in node.dependencies:
                    node.dependencies.append(dep_node)
                    dep_node.dependents.append(node)
        else:
            del nodes_by_id[node.obj.object_id]
            for dependent in list(node.dependents):
                dependent.dependencies.remove(node)
                if not dependent.dependencies:
                    zero_dependency_nodes.append(dependent)
