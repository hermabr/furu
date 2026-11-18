from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import ClassVar

from furu._batched import _BatchedHook
from furu.core import Spec
from furu.dependencies import collect_declared_refs
from furu.resources import Worker
from furu.spec_metadata import Metadata
from furu.storage.stored import Stored
from furu.utils import format_duration


@dataclass(eq=False)
class DagNode:
    obj: Spec
    # The plan file that lists it: the launch's, or the one of the process
    # whose create() asked for it.
    plan: Path
    dependencies: list[DagNode] = field(default_factory=list)
    dependents: list[DagNode] = field(default_factory=list)
    # When it first found missing each dependency its run sees built, since it
    # last got anywhere: a block on unbuilt ones or a failed attempt clears it.
    unseen_since: dict[str, float] = field(default_factory=dict)

    _batch_groups: dict[Worker, tuple[object, int] | None] = field(default_factory=dict)


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
    # The first spec seen with each batch key on each worker.
    batch_firsts: dict[tuple[Worker, object], Spec] = field(default_factory=dict)

    # How long a spec may keep finding missing a dependency the run sees built.
    # In one process that can only be a race with another run publishing it,
    # which one rerun settles.
    unseen_grace_s: ClassVar[float] = 0.0

    def batch_group(self, node: DagNode, worker: Worker) -> tuple[object, int] | None:
        """The batch ``node`` joins on ``worker`` and its cap; None when it runs alone.

        A batch is one create() call in one process that publishes one folder,
        so two specs with one batch key and different Metadata stop the run.
        """
        if worker in node._batch_groups:
            return node._batch_groups[worker]
        obj = node.obj
        group = None
        hook = getattr(type(obj), "_furu_create_hook", None)
        if isinstance(hook, _BatchedHook):
            key, cap = hook.batch_fn(obj, worker)
            if type(cap) is not int or cap < 1:
                raise TypeError(
                    f"{type(obj).__qualname__} batch key cap must be a positive int, "
                    f"got {cap!r} on {worker}"
                )
            group = (type(obj), key, cap), cap
            first = self.batch_firsts.setdefault((worker, group[0]), obj)
            if differs := [
                f"{f.name} ({getattr(first._metadata, f.name)!r} and "
                f"{getattr(obj._metadata, f.name)!r})"
                for f in fields(Metadata)
                if getattr(first._metadata, f.name) != getattr(obj._metadata, f.name)
            ]:
                raise ValueError(
                    f"{first._log_label} and {obj._log_label} have the same batch "
                    f"key and differ in Metadata {', '.join(differs)}. A batch is "
                    "one create() call that publishes one folder, so it has one "
                    "Metadata: give the specs different batch keys or the same "
                    "Metadata."
                )
        node._batch_groups[worker] = group
        return group

    def built(self, obj: Spec) -> bool:
        """Completed by this run, or done on disk (perhaps by another run)."""
        return obj.object_id in self.completed or obj.done

    def _check_new(self, nodes: Sequence[DagNode]) -> None:
        """Reject new nodes before they join the graph; runners override this."""

    def add(self, objs: Sequence[Spec], plan: Path) -> None:
        """Add the missing ``objs`` and, recursively, their declared dependencies.

        ``plan`` is the plan file that lists them.
        """
        refs_by_id: dict[str, tuple[Spec, ...]] = {}
        new: list[DagNode] = []
        pending = list(reversed(objs))
        seen_object_ids = set(self.nodes_by_id)
        look = Stored()  # one listing per class, however many specs it has

        while pending:
            obj = pending.pop()
            if obj.object_id in seen_object_ids:
                continue
            seen_object_ids.add(obj.object_id)
            if look.row(obj) is not None:
                continue
            new.append(DagNode(obj=obj, plan=plan))
            refs = collect_declared_refs(obj)
            refs_by_id[obj.object_id] = refs
            pending.extend(refs)

        self._check_new(new)
        for node in new:
            self.nodes_by_id[node.obj.object_id] = node
        # New nodes were just found unbuilt; older ones may be built by now.
        built = dict.fromkeys(refs_by_id, False)
        for node in new:
            for ref in refs_by_id[node.obj.object_id]:
                if (dep := self.nodes_by_id.get(ref.object_id)) is None:
                    continue
                if ref.object_id not in built:
                    built[ref.object_id] = self.built(dep.obj)
                if not built[ref.object_id]:
                    self._link(node, dep)
        for node in new:
            (self.blocked if node.dependencies else self.ready)[node.obj.object_id] = (
                node
            )

    def block(
        self, node: DagNode, dependencies: Sequence[Spec], plan: Path
    ) -> str | None:
        """``node`` found ``dependencies`` missing; rerun it once they're built.

        ``plan`` is the plan file in which its create() saved their lines.

        Ones this run sees built may not be visible to ``node`` yet, so if it
        names only those it just reruns; once it has found one missing for over
        ``unseen_grace_s`` without getting anywhere in between, this returns why
        ``node`` can't run and leaves it to the caller.
        """
        if not dependencies:
            return "reported itself blocked without naming a missing dependency"
        unbuilt = [dep for dep in dependencies if not self.built(dep)]
        if unbuilt:
            node.unseen_since.clear()  # it got somewhere: start its clocks over
        else:
            now = time.monotonic()
            for dep in dependencies:
                waited = now - node.unseen_since.setdefault(dep.object_id, now)
                if waited > self.unseen_grace_s:
                    span = (
                        f"for {format_duration(waited)}"
                        if self.unseen_grace_s
                        else "after a rerun"
                    )
                    return (
                        f"still finds {dep._log_label} missing {span}, "
                        "though this run sees it built"
                    )
        self.add(unbuilt, plan)
        for dep in unbuilt:
            if dep_node := self.nodes_by_id.get(dep.object_id):
                self._link(node, dep_node)
        (self.blocked if node.dependencies else self.ready)[node.obj.object_id] = node
        return None

    def complete(self, node: DagNode) -> None:
        self.completed[node.obj.object_id] = node
        for dependent in node.dependents:
            if node in dependent.dependencies:
                dependent.dependencies.remove(node)
            dependent_id = dependent.obj.object_id
            if not dependent.dependencies and dependent_id in self.blocked:
                self.ready[dependent_id] = self.blocked.pop(dependent_id)

    def _link(self, node: DagNode, dep: DagNode) -> None:
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
