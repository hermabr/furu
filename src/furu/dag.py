from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, assert_never

from furu.core import Furu
from furu.dependencies import collect_declared_refs, dependency_recheck_interval
from furu.metadata import ArtifactSpec

if TYPE_CHECKING:
    from furu.execution.manager import Manager


@dataclass(eq=False)
class DagNode:
    obj: Furu
    dependencies: list[DagNode] = field(default_factory=list)
    dependents: list[DagNode] = field(default_factory=list)
    dependency_recheck_interval: float | None = None
    next_dependency_recheck_at: float | None = None
    rechecked_declared_dependency_ids: set[str] = field(default_factory=set)


def _schedule_next_dependency_recheck(node: DagNode, *, now: float) -> None:
    if node.dependency_recheck_interval is None:
        node.next_dependency_recheck_at = None
    else:
        node.next_dependency_recheck_at = now + node.dependency_recheck_interval


def _set_node_dependencies(
    node: DagNode,
    dependencies: Sequence[DagNode],
    *,
    replaceable_dependency_ids: set[str],
) -> None:
    dependency_ids = {dependency.obj.object_id for dependency in dependencies}

    for old_dependency in tuple(node.dependencies):
        if old_dependency.obj.object_id not in replaceable_dependency_ids:
            continue
        if old_dependency.obj.object_id in dependency_ids:
            continue
        node.dependencies.remove(old_dependency)
        if node in old_dependency.dependents:
            old_dependency.dependents.remove(node)

    existing_ids = {dependency.obj.object_id for dependency in node.dependencies}
    for dependency in dependencies:
        if dependency.obj.object_id in existing_ids:
            continue
        node.dependencies.append(dependency)
        dependency.dependents.append(node)
        existing_ids.add(dependency.obj.object_id)


def _declared_dependency_nodes(
    manager: Manager,
    refs: Sequence[Furu],
) -> list[DagNode]:
    dependencies: list[DagNode] = []
    missing_dependencies: list[Furu] = []

    for ref in refs:
        if ref.object_id in manager.completed:
            continue

        dep_node = manager.nodes_by_id.get(ref.object_id)
        if dep_node is not None:
            if dep_node.obj.status() != "completed":
                dependencies.append(dep_node)
            continue

        if ref.status() == "completed":
            continue

        missing_dependencies.append(ref)

    _add_to_dag(manager, missing_dependencies)

    for ref in refs:
        if dep_node := manager.nodes_by_id.get(ref.object_id):
            if dep_node.obj.status() != "completed":
                dependencies.append(dep_node)

    deduped: dict[str, DagNode] = {}
    for dependency in dependencies:
        deduped.setdefault(dependency.obj.object_id, dependency)
    return list(deduped.values())


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
        node = DagNode(
            obj=obj, dependency_recheck_interval=dependency_recheck_interval(obj)
        )
        manager.nodes_by_id[obj.object_id] = node
        newly_added.append(node)
        refs = collect_declared_refs(obj)
        refs_by_id[obj.object_id] = refs
        pending.extend(refs)

    for obj_id, refs in refs_by_id.items():
        node = manager.nodes_by_id[obj_id]
        for ref in refs:
            if dep_node := manager.nodes_by_id.get(ref.object_id):
                node.dependencies.append(dep_node)
                dep_node.dependents.append(node)
                node.rechecked_declared_dependency_ids.add(ref.object_id)

    for node in newly_added:
        if node.dependencies:
            manager.blocked[node.obj.object_id] = node
            _schedule_next_dependency_recheck(node, now=time.monotonic())
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
        if dep_node not in node.dependencies:
            node.dependencies.append(dep_node)
        if node not in dep_node.dependents:
            dep_node.dependents.append(node)

    if node.dependencies:
        manager.blocked[node.obj.object_id] = node
        _schedule_next_dependency_recheck(node, now=time.monotonic())
    else:
        manager.ready[node.obj.object_id] = node


def _recheck_blocked_declared_dependencies(manager: Manager, *, now: float) -> None:
    for node in tuple(manager.blocked.values()):
        if (
            node.next_dependency_recheck_at is None
            or node.next_dependency_recheck_at > now
        ):
            continue

        refs = collect_declared_refs(node.obj)
        dependencies = _declared_dependency_nodes(manager, refs)
        _set_node_dependencies(
            node,
            dependencies,
            replaceable_dependency_ids=node.rechecked_declared_dependency_ids,
        )
        node.rechecked_declared_dependency_ids = {
            dependency.obj.object_id for dependency in dependencies
        }

        if node.dependencies:
            _schedule_next_dependency_recheck(node, now=now)
        else:
            manager.ready[node.obj.object_id] = manager.blocked.pop(node.obj.object_id)
