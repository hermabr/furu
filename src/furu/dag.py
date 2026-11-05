from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import assert_never

from furu._batched import _BatchedHook
from furu.core import Spec
from furu.dependencies import collect_declared_refs
from furu.migration.stale import raise_if_stale
from furu.resources import Worker

# Allow two minutes of delayed replays for cross-node filesystem visibility.
_VISIBILITY_RETRY_SECONDS = 5.0
_MAX_VISIBILITY_RETRIES = 24


class DependencyVisibilityError(RuntimeError):
    """The creator persistently cannot load dependencies the runner sees built."""


@dataclass(eq=False)
class DagNode:
    obj: Spec
    dependencies: list[DagNode] = field(default_factory=list)
    dependents: list[DagNode] = field(default_factory=list)

    # Never reset on dependency changes or progress elsewhere in the graph.
    visibility_retries: int = 0
    ready_at: float = 0.0

    _batch_groups: dict[Worker, tuple[object, int] | None] = field(default_factory=dict)

    def batch_group(self, worker: Worker) -> tuple[object, int] | None:
        if worker in self._batch_groups:
            return self._batch_groups[worker]
        obj = self.obj
        group = None
        hook = getattr(type(obj), "_furu_create_hook", None)
        if isinstance(hook, _BatchedHook):
            group_hash, cap = hook.batch_fn(obj, worker)
            if type(cap) is not int or cap < 1:
                raise TypeError(
                    f"{type(obj).__qualname__} batch key cap must be a positive int, "
                    f"got {cap!r} on {worker}"
                )
            group = (type(obj), group_hash, cap, obj._metadata), cap
        self._batch_groups[worker] = group
        return group


@dataclass(slots=True, kw_only=True)
class Dag:
    """Specs still to build and what each waits on, shared by local and worker runs.

    The runner takes nodes out of ``ready``, then hands each back through
    ``complete`` or ``block``.
    """

    nodes_by_id: dict[str, DagNode] = field(default_factory=dict)
    ready: dict[str, DagNode] = field(default_factory=dict)
    blocked: dict[str, DagNode] = field(default_factory=dict)
    completed: dict[str, DagNode] = field(default_factory=dict)

    def runnable(self) -> list[DagNode]:
        """Ready nodes whose filesystem visibility delay has elapsed."""
        now = time.monotonic()
        return [node for node in self.ready.values() if node.ready_at <= now]

    def _check_new(self, nodes: Sequence[DagNode]) -> None:
        """Reject new nodes before they join the graph; runners override this."""

    def add(self, objs: Sequence[Spec]) -> None:
        """Add the missing ``objs`` and, recursively, their declared dependencies."""
        refs_by_id: dict[str, tuple[Spec, ...]] = {}
        new: list[DagNode] = []
        pending = list(reversed(objs))
        seen_object_ids = set(self.nodes_by_id)

        while pending:
            obj = pending.pop()
            if obj.object_id in seen_object_ids:
                continue
            seen_object_ids.add(obj.object_id)
            match obj.status:
                case "done":
                    continue
                case "running" | "missing" | "incomplete":
                    pass
                case "stale":
                    raise_if_stale(obj)
                case x:
                    assert_never(x)
            new.append(DagNode(obj=obj))
            refs = collect_declared_refs(obj)
            refs_by_id[obj.object_id] = refs
            pending.extend(refs)

        self._check_new(new)
        for node in new:
            self.nodes_by_id[node.obj.object_id] = node
        for node in new:
            for ref in refs_by_id[node.obj.object_id]:
                if dep := self.nodes_by_id.get(ref.object_id):
                    self._link(node, dep)
        for node in new:
            (self.blocked if node.dependencies else self.ready)[node.obj.object_id] = (
                node
            )

    def block(self, node: DagNode, dependencies: Sequence[Spec]) -> None:
        """``node`` found ``dependencies`` missing; rerun it once they're built."""
        self.add(dependencies)
        for dep in dependencies:
            dep_node = self.nodes_by_id.get(dep.object_id)
            if dep_node is not None:
                self._link(node, dep_node)
        if not node.dependencies:
            # Nothing to build: let remote directory listings catch up, finitely.
            if node.visibility_retries >= _MAX_VISIBILITY_RETRIES:
                labels = (
                    ", ".join(dep._log_label for dep in dependencies)
                    or "reported dependencies"
                )
                raise DependencyVisibilityError(
                    f"{node.obj._log_label} still reports missing dependencies "
                    f"({labels}) that the runner considers built after "
                    f"{_MAX_VISIBILITY_RETRIES} delayed reruns "
                    f"({_MAX_VISIBILITY_RETRIES * _VISIBILITY_RETRY_SECONDS:g}s waiting). "
                    "Check filesystem visibility, storage mounts, and code versions."
                )
            node.visibility_retries += 1
            node.ready_at = time.monotonic() + _VISIBILITY_RETRY_SECONDS
        (self.blocked if node.dependencies else self.ready)[node.obj.object_id] = node

    def complete(self, node: DagNode) -> None:
        self.completed[node.obj.object_id] = node
        for dependent in node.dependents:
            if node in dependent.dependencies:
                dependent.dependencies.remove(node)
            dependent_id = dependent.obj.object_id
            if not dependent.dependencies and dependent_id in self.blocked:
                self.ready[dependent_id] = self.blocked.pop(dependent_id)

    def _link(self, node: DagNode, dep: DagNode) -> None:
        if dep.obj.object_id in self.completed or dep.obj.status == "done":
            return
        if dep in node.dependencies:
            return
        if (path := _path(dep, node)) is not None:
            raise RuntimeError(
                "dependency cycle: "
                + " → ".join(step.obj._log_label for step in (*path, dep))
            )
        node.dependencies.append(dep)
        dep.dependents.append(node)


def _path(start: DagNode, goal: DagNode) -> list[DagNode] | None:
    """The nodes from ``start`` to ``goal`` along dependency edges, if any."""
    parents: dict[int, DagNode | None] = {id(start): None}
    stack = [start]
    while stack:
        node = stack.pop()
        if node is goal:
            path = [node]
            while (parent := parents[id(path[-1])]) is not None:
                path.append(parent)
            return path[::-1]
        for dep in node.dependencies:
            if id(dep) not in parents:
                parents[id(dep)] = node
                stack.append(dep)
    return None
