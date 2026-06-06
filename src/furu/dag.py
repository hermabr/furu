from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, assert_never

from furu.core import Furu
from furu.dependencies import collect_declared_refs
from furu.metadata import ArtifactSpec

if TYPE_CHECKING:
    from furu.execution.execution_coordinator import ExecutionCoordinator


@dataclass(eq=False)
class DagNode:
    obj: Furu
    dependencies: list[DagNode] = field(default_factory=list)
    dependents: list[DagNode] = field(default_factory=list)


def _add_to_dag(coordinator: ExecutionCoordinator, objs: Sequence[Furu]) -> None:
    if any(not isinstance(obj, Furu) for obj in objs):
        # TODO: accept pytrees of Furu objects (e.g. nested lists/dicts/dataclasses)
        # and flatten them before walking dependencies.
        raise TypeError("expected Furu objects")

    refs_by_id: dict[str, tuple[Furu, ...]] = {}
    newly_added: list[DagNode] = []
    # TODO: detect cycles and raise a clear error
    pending = list(objs)

    while pending:
        obj = pending.pop()
        if obj.object_id in coordinator.nodes_by_id:
            continue
        match obj.status():
            case "completed":
                continue
            case "running":
                # TODO: handle already-running objects as external dependencies.
                raise RuntimeError(f"cannot add running object to DAG: {obj.object_id}")
            case "missing" | "failed":
                pass
            case x:
                assert_never(x)
        node = DagNode(obj=obj)
        coordinator.nodes_by_id[obj.object_id] = node
        newly_added.append(node)
        refs = collect_declared_refs(obj)
        refs_by_id[obj.object_id] = refs
        pending.extend(refs)

    for obj_id, refs in refs_by_id.items():
        node = coordinator.nodes_by_id[obj_id]
        for ref in refs:
            if dep_node := coordinator.nodes_by_id.get(ref.object_id):
                node.dependencies.append(dep_node)
                dep_node.dependents.append(node)

    for node in newly_added:
        if node.dependencies:
            coordinator.blocked[node.obj.object_id] = node
        else:
            coordinator.ready[node.obj.object_id] = node


def _update_dag_blocking_dependencies(
    coordinator: ExecutionCoordinator,
    node: DagNode,
    dependencies: Sequence[ArtifactSpec],
) -> None:
    dependency_ids: dict[str, None] = {}
    missing_dependencies: list[Furu] = []
    for artifact in dependencies:
        object_id = artifact.object_id
        if object_id in coordinator.completed or object_id in dependency_ids:
            continue

        dep_node = coordinator.nodes_by_id.get(object_id)
        if dep_node is not None:
            if dep_node.obj.status() != "completed":
                dependency_ids[object_id] = None
            continue

        dependency = Furu.from_artifact(artifact)
        if dependency.status() == "completed":
            continue

        dependency_ids[object_id] = None
        missing_dependencies.append(dependency)

    _add_to_dag(coordinator, missing_dependencies)

    for dependency_id in dependency_ids:
        dep_node = coordinator.nodes_by_id[dependency_id]
        if dep_node not in node.dependencies:
            node.dependencies.append(dep_node)
        if node not in dep_node.dependents:
            dep_node.dependents.append(node)

    if node.dependencies:
        coordinator.blocked[node.obj.object_id] = node
    else:
        coordinator.ready[node.obj.object_id] = node
