"""The graph of specs a run builds, and the one state machine both runners drive.

A node's state lives on the node and changes only through ``Dag._move``, which
checks it against ``TRANSITIONS``::

    (new)    → blocked | ready             add
    blocked  → ready                       its last dependency completed
    ready    → running | locked | done     start: is it free, held elsewhere, or built?
    locked   → ready                       poll: the other run let go of it
    deferred → ready                       poll: rerun it
    running  → done                        it built
    running  → locked                      another run took its lock first
    running  → ready | failed              a failed attempt: retry, or out of retries
    running  → blocked | deferred | failed it found dependencies missing

Runners differ only in policy: how many retries, how long a spec may keep
missing dependencies this run sees built, and how often they poll.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import assert_never

from furu._batched import _BatchedHook
from furu.core import Spec
from furu.dependencies import collect_declared_refs
from furu.logging import get_logger
from furu.migration.stale import raise_if_stale
from furu.resources import Worker
from furu.utils import format_duration

logger = get_logger("run")


class State(StrEnum):
    BLOCKED = "blocked"  # waits on dependencies in this graph
    READY = "ready"
    RUNNING = "running"  # this run is building it
    LOCKED = "locked"  # another run holds its lock
    DEFERRED = "deferred"  # missed only specs this run sees built; rerun next poll
    DONE = "done"
    FAILED = "failed"


TRANSITIONS: dict[State | None, frozenset[State]] = {
    None: frozenset({State.BLOCKED, State.READY}),
    State.BLOCKED: frozenset({State.READY}),
    State.READY: frozenset({State.RUNNING, State.LOCKED, State.DONE}),
    State.LOCKED: frozenset({State.READY}),
    State.DEFERRED: frozenset({State.READY}),
    State.RUNNING: frozenset(
        {
            State.DONE,
            State.LOCKED,
            State.READY,
            State.FAILED,
            State.BLOCKED,
            State.DEFERRED,
        }
    ),
    State.DONE: frozenset(),
    State.FAILED: frozenset(),
}

# Ranks failed specs in order of first failure, like the run's log.
_failure_order = itertools.count()


@dataclass(eq=False)
class DagNode:
    obj: Spec
    state: State | None = None  # None until it joins the graph
    dependencies: list[DagNode] = field(default_factory=list)
    dependents: list[DagNode] = field(default_factory=list)
    started_at: float = 0.0
    attempts: int = 0  # failed attempts
    # The last failed attempt's error, or why it can't run.
    error: str | None = None
    first_failure: int | None = None
    # When it first found missing each dependency its run sees built, since it
    # last got anywhere: a block on unbuilt ones or a failed attempt clears it.
    unseen_since: dict[str, float] = field(default_factory=dict)

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


def _accept(nodes: Sequence[DagNode]) -> None:
    pass


@dataclass(slots=True, kw_only=True)
class Dag:
    """Specs still to build, what each waits on, and where each is in its life."""

    # Sees new nodes before they join the graph, and may reject them by raising.
    check_new: Callable[[Sequence[DagNode]], None] = _accept
    nodes_by_id: dict[str, DagNode] = field(default_factory=dict)
    _by_state: dict[State, dict[str, DagNode]] = field(
        default_factory=lambda: {state: {} for state in State}
    )
    # Nodes with a failed attempt that are neither done nor failed.
    retrying: int = 0

    def nodes(self, state: State) -> Mapping[str, DagNode]:
        """The nodes in ``state``, in the order they entered it."""
        return self._by_state[state]

    def active(self) -> bool:
        """Whether anything can still move without new information."""
        return any(
            self._by_state[state]
            for state in (State.READY, State.RUNNING, State.LOCKED, State.DEFERRED)
        )

    def built(self, obj: Spec) -> bool:
        """Completed by this run, or done on disk (perhaps by another run)."""
        node = self.nodes_by_id.get(obj.object_id)
        return (node is not None and node.state is State.DONE) or obj.status == "done"

    def _move(self, node: DagNode, state: State) -> None:
        """The one place a node changes state."""
        if state not in TRANSITIONS[node.state]:
            raise AssertionError(f"{node.obj._log_label}: {node.state} → {state}")
        object_id = node.obj.object_id
        if node.state is not None:
            del self._by_state[node.state][object_id]
        if node.attempts and state in (State.DONE, State.FAILED):
            self.retrying -= 1
        node.state = state
        self._by_state[state][object_id] = node

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

        self.check_new(new)
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
            self._move(node, State.BLOCKED if node.dependencies else State.READY)

    def start(self, node: DagNode) -> State:
        """Take a ready node: run it, or not if another run holds or built it."""
        match node.obj.status:
            case "done":
                logger.info("found %s done", node.obj._log_label)
                self.complete(node)
            case "running":
                self._move(node, State.LOCKED)
            case "missing" | "incomplete" | "stale":
                node.started_at = time.monotonic()
                self._move(node, State.RUNNING)
            case unreachable:
                assert_never(unreachable)
        assert node.state is not None
        return node.state

    def poll(self) -> None:
        """Ready the locked nodes whose lock went, and every deferred one."""
        for node in list(self._by_state[State.LOCKED].values()):
            if node.obj.status != "running":
                self._move(node, State.READY)
        for node in list(self._by_state[State.DEFERRED].values()):
            self._move(node, State.READY)

    def complete(self, node: DagNode) -> None:
        self._move(node, State.DONE)
        for dependent in node.dependents:
            if node in dependent.dependencies:
                dependent.dependencies.remove(node)
            if not dependent.dependencies and dependent.state is State.BLOCKED:
                self._move(dependent, State.READY)

    def lost_lock_race(self, node: DagNode) -> None:
        """Another run took ``node``'s lock between our status check and ours."""
        self._move(node, State.LOCKED)

    def attempt_failed(self, node: DagNode, error: str, *, max_retries: int) -> bool:
        """``node``'s attempt failed; retry it unless it is out of retries."""
        node.unseen_since.clear()  # its earlier misses are behind it
        node.attempts += 1
        if node.attempts == 1:
            self.retrying += 1
        self._failed(node, error, retry=node.attempts <= max_retries)
        return node.state is State.READY

    def block(
        self,
        node: DagNode,
        dependencies: Sequence[Spec],
        *,
        unseen_grace_s: float,
        unseen_hint: str = "",
    ) -> str | None:
        """``node`` found ``dependencies`` missing; rerun it once they're built.

        Ones this run sees built may not be visible to ``node`` yet, so if it
        names only those it is deferred to rerun; once it has found one missing
        for over ``unseen_grace_s`` without getting anywhere in between, it
        fails and this returns why (with ``unseen_hint`` appended).
        """
        if not dependencies:
            why = "reported itself blocked without naming a missing dependency"
            self._failed(node, why, retry=False)
            return why
        unbuilt = [dep for dep in dependencies if not self.built(dep)]
        if unbuilt:
            node.unseen_since.clear()  # it got somewhere: start its clocks over
        else:
            now = time.monotonic()
            for dep in dependencies:
                waited = now - node.unseen_since.setdefault(dep.object_id, now)
                if waited > unseen_grace_s:
                    span = (
                        f"for {format_duration(waited)}"
                        if unseen_grace_s
                        else "after a rerun"
                    )
                    why = (
                        f"still finds {dep._log_label} missing {span}, "
                        f"though this run sees it built{unseen_hint}"
                    )
                    self._failed(node, why, retry=False)
                    return why
        self.add(unbuilt)
        for dep in unbuilt:
            if dep_node := self.nodes_by_id.get(dep.object_id):
                self._link(node, dep_node)
        # Rerunning at once would likely miss the same specs again.
        self._move(node, State.BLOCKED if node.dependencies else State.DEFERRED)
        return None

    def _failed(self, node: DagNode, error: str, *, retry: bool) -> None:
        node.error = error
        if node.first_failure is None:
            node.first_failure = next(_failure_order)
        self._move(node, State.READY if retry else State.FAILED)

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
