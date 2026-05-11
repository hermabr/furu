from __future__ import annotations

import pytest

import furu
from furu import Furu
from furu.graph import (
    NodeKey,
    discover_missing_closure,
    node_key_for,
)


class Leaf(Furu[str]):
    name: str

    def create(self) -> str:
        return f"Leaf({self.name})"


class Mid(Furu[str]):
    leaf: Leaf

    def create(self) -> str:
        return f"Mid({self.leaf.load_or_create()})"


class Top(Furu[str]):
    mid: Mid
    suffix: str

    def create(self) -> str:
        return f"Top({self.mid.load_or_create()}/{self.suffix})"


class Dynamic(Furu[str]):
    name: str

    def create(self) -> str:
        child = Leaf(name=self.name)
        return f"Dyn({child.load_or_create()})"


class TryLoadParent(Furu[str]):
    name: str

    def create(self) -> str:
        child = Leaf(name=self.name)
        return f"Try({child.try_load()})"


class BrokenLeaf(Furu[str]):
    name: str

    def create(self) -> str:
        raise RuntimeError("boom")


def test_submit_single_returns_result():
    obj = Leaf(name="a")
    submission = furu.submit(obj, executor=furu.LocalExecutor(num_workers=1))
    result = submission.result(timeout_s=30)
    assert result == "Leaf(a)"


def test_submit_list_returns_list_of_results():
    objs = [Leaf(name="a"), Leaf(name="b")]
    submission = furu.submit(objs, executor=furu.LocalExecutor(num_workers=2))
    result = submission.result(timeout_s=30)
    assert result == ["Leaf(a)", "Leaf(b)"]


def test_submit_with_declared_dependencies():
    obj = Top(mid=Mid(leaf=Leaf(name="x")), suffix="end")
    submission = furu.submit(obj, executor=furu.LocalExecutor(num_workers=2))
    result = submission.result(timeout_s=30)
    assert result == "Top(Mid(Leaf(x))/end)"


def test_submit_with_dynamic_dependency():
    obj = Dynamic(name="dyn")
    submission = furu.submit(obj, executor=furu.LocalExecutor(num_workers=2))
    result = submission.result(timeout_s=30)
    assert result == "Dyn(Leaf(dyn))"


def test_submit_try_load_dynamic_dependency():
    obj = TryLoadParent(name="try")
    submission = furu.submit(obj, executor=furu.LocalExecutor(num_workers=2))
    result = submission.result(timeout_s=30)
    assert result == "Try(Leaf(try))"


def test_furu_submit_convenience_method():
    obj = Leaf(name="conv")
    submission = obj.submit(executor=furu.LocalExecutor(num_workers=1))
    result = submission.result(timeout_s=30)
    assert result == "Leaf(conv)"


def test_submission_status_done_after_result():
    obj = Leaf(name="status")
    submission = furu.submit(obj, executor=furu.LocalExecutor(num_workers=1))
    submission.result(timeout_s=30)
    status = submission.status()
    assert status.status == "done"
    assert status.done_nodes == status.total_nodes


def test_discover_missing_closure_includes_transitive_declared_refs():
    obj = Top(mid=Mid(leaf=Leaf(name="t")), suffix="end")
    graph = discover_missing_closure([obj])

    keys = {node.key for node in graph.nodes}
    assert node_key_for(obj) in keys
    assert node_key_for(obj.mid) in keys
    assert node_key_for(obj.mid.leaf) in keys

    edges_by_dependent: dict[NodeKey, set[NodeKey]] = {}
    for dep, dependent in graph.edges:
        edges_by_dependent.setdefault(dependent, set()).add(dep)
    assert node_key_for(obj.mid) in edges_by_dependent[node_key_for(obj)]
    assert node_key_for(obj.mid.leaf) in edges_by_dependent[node_key_for(obj.mid)]


def test_discover_missing_closure_stops_at_cached_results():
    leaf = Leaf(name="cached")
    leaf.load_or_create()

    graph = discover_missing_closure([Mid(leaf=leaf)])

    keys = {node.key for node in graph.nodes}
    assert node_key_for(leaf) in keys

    # Even though leaf is in the graph, it should have no outgoing or
    # incoming edges discovered because we stop expanding cached nodes.
    edges_from_leaf = [
        (dep, dependent)
        for dep, dependent in graph.edges
        if dep == node_key_for(leaf) or dependent == node_key_for(leaf)
    ]
    assert len(edges_from_leaf) == 1
    # The edge from leaf to mid should still be present (it was discovered
    # while expanding mid), but leaf itself has no expansion.
    dep, dependent = edges_from_leaf[0]
    assert dep == node_key_for(leaf)


def test_submit_failure_raises_submission_failed():
    from furu.submission import SubmissionFailed

    obj = BrokenLeaf(name="b")
    submission = furu.submit(obj, executor=furu.LocalExecutor(num_workers=1))

    with pytest.raises(SubmissionFailed, match="boom"):
        submission.result(timeout_s=30)
