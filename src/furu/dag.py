from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, assert_never

from furu.core import Furu
from furu.dependencies import collect_declared_refs
from furu.metadata import ArtifactSpec

if TYPE_CHECKING:
    from furu.execution.manager import Manager


@dataclass(eq=False)
class DagNode:
    obj: Furu
    dependencies: list[DagNode] = field(default_factory=list)
    dependents: list[DagNode] = field(default_factory=list)
    declared_dependency_ids: set[str] = field(default_factory=set)
    runtime_dependency_ids: set[str] = field(default_factory=set)


def _ensure_edge(node: DagNode, dep_node: DagNode) -> None:
    if dep_node not in node.dependencies:
        node.dependencies.append(dep_node)
    if node not in dep_node.dependents:
        dep_node.dependents.append(node)


def _remove_edge(node: DagNode, dep_node: DagNode) -> None:
    if dep_node in node.dependencies:
        node.dependencies.remove(dep_node)
    if node in dep_node.dependents:
        dep_node.dependents.remove(node)


def _is_running(manager: Manager, node: DagNode) -> bool:
    return any(running.node is node for running in manager.running.values())


def _move_ready_or_blocked(manager: Manager, node: DagNode) -> None:
    object_id = node.obj.object_id
    if (
        object_id in manager.completed
        or object_id in manager.failed
        or _is_running(manager, node)
    ):
        return

    if node.dependencies:
        manager.ready.pop(object_id, None)
        manager.blocked[object_id] = node
    else:
        manager.blocked.pop(object_id, None)
        manager.ready[object_id] = node


def _add_to_dag(manager: Manager, objs: Sequence[Furu]) -> None:
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
        if obj.object_id in manager.nodes_by_id:
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
        manager.nodes_by_id[obj.object_id] = node
        newly_added.append(node)
        refs = collect_declared_refs(obj)
        refs_by_id[obj.object_id] = refs
        pending.extend(refs)

    for obj_id, refs in refs_by_id.items():
        node = manager.nodes_by_id[obj_id]
        for ref in refs:
            if ref.object_id in manager.completed:
                continue
            if dep_node := manager.nodes_by_id.get(ref.object_id):
                if dep_node.obj.status() == "completed":
                    continue
                node.declared_dependency_ids.add(ref.object_id)
                _ensure_edge(node, dep_node)

    for node in newly_added:
        if node.dependencies:
            manager.blocked[node.obj.object_id] = node
        else:
            manager.ready[node.obj.object_id] = node


def _update_dag_blocking_dependencies(
    manager: Manager,
    node: DagNode,
    dependencies: Sequence[ArtifactSpec],
) -> None:
    dependency_ids: dict[str, None] = {}
    missing_dependencies: list[Furu] = []
    for artifact in dependencies:
        object_id = artifact.object_id
        if object_id in manager.completed or object_id in dependency_ids:
            continue

        dep_node = manager.nodes_by_id.get(object_id)
        if dep_node is not None:
            if dep_node.obj.status() != "completed":
                dependency_ids[object_id] = None
            continue

        dependency = Furu.from_artifact(artifact)
        if dependency.status() == "completed":
            continue

        dependency_ids[object_id] = None
        missing_dependencies.append(dependency)

    _add_to_dag(manager, missing_dependencies)

    for dependency_id in dependency_ids:
        dep_node = manager.nodes_by_id[dependency_id]
        node.runtime_dependency_ids.add(dependency_id)
        _ensure_edge(node, dep_node)

    _move_ready_or_blocked(manager, node)


def _refresh_dag_declared_dependencies(manager: Manager) -> None:
    nodes = tuple(manager.blocked.values()) + tuple(manager.ready.values())
    for node in nodes:
        if manager.nodes_by_id.get(node.obj.object_id) is not node:
            continue
        refs = collect_declared_refs(node.obj)
        _reconcile_declared_dependencies(manager, node, refs)


def _reconcile_declared_dependencies(
    manager: Manager,
    node: DagNode,
    refs: Sequence[Furu],
) -> None:
    wanted_declared_ids: set[str] = set()
    wanted_declared_ids_in_order: list[str] = []
    missing_dependencies: list[Furu] = []

    for ref in refs:
        object_id = ref.object_id
        if object_id == node.obj.object_id or object_id in manager.completed:
            continue

        dep_node = manager.nodes_by_id.get(object_id)
        if dep_node is not None:
            if dep_node.obj.status() != "completed":
                wanted_declared_ids.add(object_id)
                wanted_declared_ids_in_order.append(object_id)
            continue

        match ref.status():
            case "completed":
                continue
            case "running" | "missing" | "failed":
                wanted_declared_ids.add(object_id)
                wanted_declared_ids_in_order.append(object_id)
                missing_dependencies.append(ref)
            case x:
                assert_never(x)

    _add_to_dag(manager, missing_dependencies)

    for dependency_id in tuple(node.declared_dependency_ids - wanted_declared_ids):
        node.declared_dependency_ids.discard(dependency_id)
        if dependency_id in node.runtime_dependency_ids:
            continue

        dep_node = manager.nodes_by_id.get(dependency_id)
        if dep_node is None:
            continue

        _remove_edge(node, dep_node)
        _prune_orphan_dependency(manager, dep_node)

    for dependency_id in wanted_declared_ids_in_order:
        dep_node = manager.nodes_by_id.get(dependency_id)
        if dep_node is None:
            continue

        node.declared_dependency_ids.add(dependency_id)
        _ensure_edge(node, dep_node)

    _move_ready_or_blocked(manager, node)


def _prune_orphan_dependency(manager: Manager, node: DagNode) -> None:
    object_id = node.obj.object_id
    if (
        object_id in manager.root_ids
        or node.dependents
        or _is_running(manager, node)
        or manager.nodes_by_id.get(object_id) is not node
    ):
        return

    manager.ready.pop(object_id, None)
    manager.blocked.pop(object_id, None)
    manager.completed.pop(object_id, None)
    manager.failed.pop(object_id, None)
    manager.nodes_by_id.pop(object_id, None)

    for dep_node in tuple(node.dependencies):
        node.declared_dependency_ids.discard(dep_node.obj.object_id)
        node.runtime_dependency_ids.discard(dep_node.obj.object_id)
        _remove_edge(node, dep_node)
        _prune_orphan_dependency(manager, dep_node)
