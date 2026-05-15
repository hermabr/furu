from dataclasses import dataclass
from typing import ClassVar

import pytest

import furu
from furu import Furu
from furu.dag import DagNode
from furu.execution.manager import Manager
from furu._storage_layout import (
    compute_lock_path_in,
    run_log_path_in,
)
from furu.worker.backends.local import LocalThreadWorkerBackend


class Leaf(Furu[str]):
    name: str

    def create(self) -> str:
        return f"leaf:{self.name}"


class Mid(Furu[str]):
    label: str
    child: Leaf

    def create(self) -> str:
        return f"mid:{self.label}:{self.child.load_or_create()}"


class Top(Furu[str]):
    name: str
    left: Mid
    right: Mid

    def create(self) -> str:
        return f"top:{self.name}"


@dataclass(frozen=True)
class LeafBundle:
    a: Leaf
    b: Leaf


class NestedParent(Furu[str]):
    bundle: LeafBundle

    def create(self) -> str:
        return self.bundle.a.load_or_create()


class ComputedParent(Furu[str]):
    name: str

    @furu.dependency
    def computed_child(self) -> Leaf:
        return Leaf(name=f"computed-{self.name}")

    def create(self) -> str:
        return self.computed_child.load_or_create()


def mark_running(obj: Furu) -> None:
    obj._base_dir.mkdir(parents=True, exist_ok=True)
    compute_lock_path_in(obj._base_dir).touch()


def test_add_to_dag_single_object_no_dependencies():
    leaf = Leaf(name="x")
    manager = Manager([leaf])

    assert len(manager.ready) == 1
    (root,) = manager.ready.values()
    assert isinstance(root, DagNode)
    assert root.obj is leaf
    assert root.dependencies == []
    assert root.dependents == []

    assert manager.nodes_by_id == {leaf.object_id: root}
    assert manager.blocked == {}


def test_add_to_dag_traverses_declared_refs_recursively():
    leaf_a = Leaf(name="a")
    leaf_b = Leaf(name="b")
    mid_left = Mid(label="L", child=leaf_a)
    mid_right = Mid(label="R", child=leaf_b)
    top = Top(name="t", left=mid_left, right=mid_right)

    manager = Manager([top])

    assert set(manager.ready) == {leaf_a.object_id, leaf_b.object_id}
    assert set(manager.blocked) == {
        mid_left.object_id,
        mid_right.object_id,
        top.object_id,
    }

    assert set(manager.nodes_by_id) == {
        leaf_a.object_id,
        leaf_b.object_id,
        mid_left.object_id,
        mid_right.object_id,
        top.object_id,
    }

    leaf_a_node = manager.nodes_by_id[leaf_a.object_id]
    mid_left_node = manager.nodes_by_id[mid_left.object_id]
    top_node = manager.nodes_by_id[top.object_id]

    assert leaf_a_node.dependencies == []
    assert leaf_a_node.dependents == [mid_left_node]
    assert mid_left_node.dependencies == [leaf_a_node]
    assert mid_left_node.dependents == [top_node]
    assert {dep.obj.object_id for dep in top_node.dependencies} == {
        mid_left.object_id,
        mid_right.object_id,
    }
    assert top_node.dependents == []


def test_add_to_dag_shared_dependency_has_multiple_dependents():
    shared = Leaf(name="shared")
    mid_left = Mid(label="L", child=shared)
    mid_right = Mid(label="R", child=shared)
    top = Top(name="t", left=mid_left, right=mid_right)

    manager = Manager([top])

    assert len(manager.ready) == 1
    (shared_root,) = manager.ready.values()
    assert shared_root.obj is shared
    assert shared_root.dependencies == []
    assert {n.obj.object_id for n in shared_root.dependents} == {
        mid_left.object_id,
        mid_right.object_id,
    }

    assert set(manager.nodes_by_id) == {
        shared.object_id,
        mid_left.object_id,
        mid_right.object_id,
        top.object_id,
    }
    assert manager.nodes_by_id[shared.object_id] is shared_root


def test_add_to_dag_stops_recursion_at_completed_objects():
    leaf = Leaf(name="cached")
    mid = Mid(label="m", child=leaf)
    leaf.load_or_create()
    assert leaf.status() == "completed"

    manager = Manager([mid])

    assert len(manager.ready) == 1
    (mid_root,) = manager.ready.values()
    assert mid_root.obj is mid
    assert mid_root.dependencies == []

    assert set(manager.nodes_by_id) == {mid.object_id}
    assert mid_root.dependents == []


def test_add_to_dag_completed_root_has_no_dependencies():
    leaf = Leaf(name="root-cached")
    mid = Mid(label="m", child=leaf)
    mid.load_or_create()
    assert mid.status() == "completed"

    manager = Manager([mid])

    assert manager.ready == {}
    assert manager.nodes_by_id == {}
    assert manager.blocked == {}


def test_add_to_dag_rejects_running_root():
    leaf = Leaf(name="running-root")
    mark_running(leaf)
    assert leaf.status() == "running"

    with pytest.raises(RuntimeError, match="cannot add running object to DAG"):
        Manager([leaf])


def test_add_to_dag_rejects_running_dependency():
    leaf = Leaf(name="running-dependency")
    mid = Mid(label="m", child=leaf)
    mark_running(leaf)
    assert leaf.status() == "running"

    with pytest.raises(RuntimeError, match="cannot add running object to DAG"):
        Manager([mid])


def test_add_to_dag_accepts_a_list_of_inputs():
    leaf_a = Leaf(name="a")
    leaf_b = Leaf(name="b")
    mid = Mid(label="m", child=leaf_a)

    manager = Manager([mid, leaf_b])

    assert set(manager.ready) == {leaf_a.object_id, leaf_b.object_id}

    assert set(manager.nodes_by_id) == {
        leaf_a.object_id,
        leaf_b.object_id,
        mid.object_id,
    }

    leaf_b_node = manager.nodes_by_id[leaf_b.object_id]
    assert leaf_b_node.dependents == []


def test_add_to_dag_handles_nested_dataclass_refs():
    leaf_a = Leaf(name="a")
    leaf_b = Leaf(name="b")
    parent = NestedParent(bundle=LeafBundle(a=leaf_a, b=leaf_b))

    manager = Manager([parent])

    assert set(manager.ready) == {leaf_a.object_id, leaf_b.object_id}

    assert set(manager.nodes_by_id) == {
        leaf_a.object_id,
        leaf_b.object_id,
        parent.object_id,
    }


def test_add_to_dag_walks_computed_dependencies():
    parent = ComputedParent(name="p")

    manager = Manager([parent])

    assert len(manager.ready) == 1
    (child_root,) = manager.ready.values()
    assert child_root.obj.object_id == parent.computed_child.object_id
    assert {n.obj.object_id for n in child_root.dependents} == {parent.object_id}
    assert set(manager.nodes_by_id) == {
        parent.object_id,
        parent.computed_child.object_id,
    }


def test_add_to_dag_rejects_non_furu_values():
    with pytest.raises(TypeError, match="expected Furu objects"):
        Manager([Leaf(name="ok"), "not-a-furu"])  # ty: ignore[invalid-argument-type]


class TrackingLeaf(Furu[int]):
    n: int
    create_calls: ClassVar[list[int]] = []

    def create(self) -> int:
        type(self).create_calls.append(self.n)
        return self.n * 2


class TrackingMid(Furu[int]):
    label: str
    child: TrackingLeaf
    create_calls: ClassVar[list[str]] = []

    def create(self) -> int:
        type(self).create_calls.append(self.label)
        return self.child.load_or_create() + 1


class LazyChildLoader(Furu[int]):
    base: int
    create_calls: ClassVar[list[int]] = []

    def create(self) -> int:
        type(self).create_calls.append(self.base)
        return self.base + TrackingLeaf(n=self.base).load_or_create()


class AlwaysFails(Furu[int]):
    name: str

    def create(self) -> int:
        raise RuntimeError(f"intentional failure: {self.name}")


class DependsOnFailing(Furu[int]):
    label: str
    child: AlwaysFails

    def create(self) -> int:
        return self.child.load_or_create() + 1


@pytest.fixture(autouse=True)
def _reset_tracking() -> None:
    TrackingLeaf.create_calls.clear()
    TrackingMid.create_calls.clear()
    LazyChildLoader.create_calls.clear()


def test_manager_run_runs_single_zero_dependency_node():
    leaf = TrackingLeaf(n=3)

    Manager([leaf]).run(worker_backend=LocalThreadWorkerBackend())

    assert TrackingLeaf.create_calls == [3]
    assert leaf.status() == "completed"
    assert leaf.load_or_create() == 6


def test_manager_run_runs_static_dependencies_in_order():
    leaf = TrackingLeaf(n=4)
    mid = TrackingMid(label="m", child=leaf)

    Manager([mid]).run(worker_backend=LocalThreadWorkerBackend())

    assert TrackingLeaf.create_calls == [4]
    assert TrackingMid.create_calls == ["m"]
    assert mid.load_or_create() == 9


def test_manager_run_handles_shared_dependency_only_once():
    shared = TrackingLeaf(n=5)
    left = TrackingMid(label="L", child=shared)
    right = TrackingMid(label="R", child=shared)

    Manager([left, right]).run(worker_backend=LocalThreadWorkerBackend())

    assert TrackingLeaf.create_calls == [5]
    assert sorted(TrackingMid.create_calls) == ["L", "R"]


def test_manager_run_with_multiple_workers_runs_independent_nodes():
    leaves = [TrackingLeaf(n=i) for i in range(8)]

    Manager(leaves).run(worker_backend=LocalThreadWorkerBackend(n_workers=4))

    assert sorted(TrackingLeaf.create_calls) == list(range(8))
    for leaf in leaves:
        assert leaf.status() == "completed"


def test_manager_run_discovers_lazy_dependencies_and_reruns_parent():
    parent = LazyChildLoader(base=7)

    Manager([parent]).run(worker_backend=LocalThreadWorkerBackend())

    assert TrackingLeaf.create_calls == [7]
    # Parent's create() is called once to discover the lazy dep (raising
    # _DependencyNotReady), then once more after the dep completes.
    assert LazyChildLoader.create_calls == [7, 7]
    assert parent.load_or_create() == 21
    parent_log = run_log_path_in(parent._base_dir).read_text(encoding="utf-8")
    assert (
        "load_or_create deferred: load_or_create discovered 1 missing dependency/dependencies"
        in parent_log
    )
    assert "load_or_create failed" not in parent_log
    assert "=== Debug Details (with locals) ===" not in parent_log


def test_manager_run_skips_already_completed_objects():
    leaf = TrackingLeaf(n=8)
    leaf.load_or_create()
    TrackingLeaf.create_calls.clear()
    mid = TrackingMid(label="cached-child", child=leaf)

    Manager([mid]).run(worker_backend=LocalThreadWorkerBackend())

    assert TrackingLeaf.create_calls == []
    assert TrackingMid.create_calls == ["cached-child"]


def test_manager_run_records_worker_failures_and_blocked_dependents():
    failing = AlwaysFails(name="boom")
    parent = DependsOnFailing(label="p", child=failing)
    manager = Manager([parent])

    with pytest.raises(RuntimeError, match="failed jobs"):
        manager.run(worker_backend=LocalThreadWorkerBackend())

    assert failing.object_id in manager.failed
    assert "intentional failure: boom" in manager.failed[failing.object_id].error
    assert parent.object_id in manager.blocked
    assert manager.done.is_set()
