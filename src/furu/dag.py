"""What is left to run and what each part waits for, by id.

The manager keeps one for everything it was sent, and a launch that runs in
its own process keeps one for itself.
"""

from __future__ import annotations

import time
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from furu.utils import format_duration, spec_label


@dataclass(eq=False, slots=True)
class _Node:
    key: str
    # A value that holds specs has nothing to run: it is finished when they are.
    runs: bool
    # How many unfinished nodes it waits for.
    waits: int = 0


@dataclass(slots=True)
class Dag:
    """The specs still to run and the values that hold them, each under its id.

    Anything that is not a node is finished. The runner takes specs out of
    ``ready``, then hands each back through ``complete`` or ``block``.
    """

    nodes: dict[str, _Node] = field(default_factory=dict)
    # The nodes that wait for each node. One that was taken out since is no
    # longer the node under its key.
    dependents: dict[str, list[_Node]] = field(default_factory=dict)
    # The specs that wait for nothing and that the runner has not taken.
    ready: dict[str, None] = field(default_factory=dict)

    def add(
        self, holds: Mapping[str, Iterable[str]], values: Collection[str] = ()
    ) -> None:
        """Add each key of ``holds`` that is not a node yet.

        ``holds`` names what each holds, and a new node waits for what it
        holds among the other keys: only those are still to run. ``values``
        are the keys that have nothing to run themselves.
        """
        new = [
            self.nodes.setdefault(key, _Node(key, key not in values))
            for key in holds
            if key not in self.nodes
        ]
        for node in new:
            for held in dict.fromkeys(holds[node.key]):
                if held in holds and held != node.key:
                    self._wait(node, held)
        for node in new:
            self._settle(node)

    def waits(self, key: str) -> bool:
        return self.nodes[key].waits > 0

    def block(self, key: str, on: Iterable[str]) -> None:
        """The spec ``key`` ran and found ``on`` missing: ready once those finish."""
        node = self.nodes[key]
        for held in dict.fromkeys(on):
            if held not in self.nodes:
                continue
            if (path := self._path(key, held)) is not None:
                raise RuntimeError(
                    "dependency cycle: "
                    + " → ".join(spec_label(*step.split(":")) for step in (*path, held))
                )
            self._wait(node, held)
        self._settle(node)

    def complete(self, key: str) -> None:
        """``key`` is finished, or nobody wants it: let go of what waits for it."""
        finished = [key]
        while finished:
            key = finished.pop()
            del self.nodes[key]
            self.ready.pop(key, None)
            for node in self.dependents.pop(key, ()):
                node.waits -= 1
                if self.nodes.get(node.key) is node and not node.waits:
                    if node.runs:
                        self.ready[node.key] = None
                    else:
                        finished.append(node.key)

    def waiting_for(self, keys: Iterable[str]) -> list[str]:
        """The specs that cannot run before ``keys`` finish, however far up."""
        seen = set(keys)
        found = []
        pending = list(seen)
        while pending:
            for node in self.dependents.get(pending.pop(), ()):
                if self.nodes.get(node.key) is node and node.key not in seen:
                    seen.add(node.key)
                    pending.append(node.key)
                    if node.runs:
                        found.append(node.key)
        return found

    def _wait(self, node: _Node, held: str) -> None:
        node.waits += 1
        self.dependents.setdefault(held, []).append(node)

    def _settle(self, node: _Node) -> None:
        if self.nodes.get(node.key) is node and not node.waits:
            if node.runs:
                self.ready[node.key] = None
            else:
                self.complete(node.key)

    def _path(self, start: str, goal: str) -> list[str] | None:
        """The specs from ``goal`` down to ``start``, when ``goal`` waits for ``start``."""
        parents: dict[str, str | None] = {start: None}
        pending = [start]
        while pending:
            key = pending.pop()
            if key == goal:
                path = []
                step: str | None = key
                while step is not None:
                    if self.nodes[step].runs:
                        path.append(step)
                    step = parents[step]
                return path
            for node in self.dependents.get(key, ()):
                if self.nodes.get(node.key) is node and node.key not in parents:
                    parents[node.key] = key
                    pending.append(node.key)
        return None


def why_stuck(
    since: dict[str, float],
    dependencies: Sequence[str],
    unbuilt: bool,
    grace_s: float,
) -> str | None:
    """Why a spec whose create() found ``dependencies`` missing cannot run again, if so.

    With ``unbuilt``, its run does not see all of them built either, and it
    reruns once they are. Otherwise the machine it ran on may not show yet
    what another just published, so it reruns as it is. ``since`` holds when
    it first found each one missing since it last got anywhere. Past
    ``grace_s`` for one of them, this says so.
    """
    if not dependencies:
        return "reported itself blocked without naming a missing dependency"
    if unbuilt:
        since.clear()
        return None
    now = time.monotonic()
    for dependency in dependencies:
        waited = now - since.setdefault(dependency, now)
        if waited > grace_s:
            span = f"for {format_duration(waited)}" if grace_s else "after a rerun"
            return (
                f"still finds {spec_label(*dependency.split(':'))} missing {span}, "
                "though this run sees it built"
            )
    return None
