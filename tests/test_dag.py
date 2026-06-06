import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, ClassVar, Iterator

import pytest

import furu
from furu import Furu
from furu.config import get_config
from furu.dag import DagNode, _add_to_dag
from furu.execution.coordinator import ExecutionCoordinator
from furu.locking import lock_many
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
        return f"mid:{self.label}:{self.child.create()}"


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
        return self.bundle.a.create()


class ComputedParent(Furu[str]):
    name: str

    @furu.dependency
    def computed_child(self) -> Leaf:
        return Leaf(name=f"computed-{self.name}")

    def create(self) -> str:
        return self.computed_child.create()


@contextmanager
def mark_running(obj: Furu) -> Iterator[None]:
    obj._base_dir.mkdir(parents=True, exist_ok=True)
    with lock_many([compute_lock_path_in(obj._base_dir)]):
        yield


def _new_coordinator(
    objs: Sequence[Furu[Any]],
    *,
    max_retries_per_object: int | None = None,
) -> ExecutionCoordinator:
    if max_retries_per_object is None:
        max_retries_per_object = get_config().worker.max_retries_per_object
    coordinator = ExecutionCoordinator(max_retries_per_object=max_retries_per_object)
    _add_to_dag(coordinator, objs)
    digest = hashlib.blake2s(digest_size=16)
    for obj in objs:
        digest.update(obj.object_id.encode("utf-8"))
        digest.update(b"\0")
    coordinator.executor_id = digest.hexdigest()
    return coordinator


def test_add_to_dag_single_object_no_dependencies():
    leaf = Leaf(name="x")
    coordinator = _new_coordinator([leaf])

    assert len(coordinator.ready) == 1
    (root,) = coordinator.ready.values()
    assert isinstance(root, DagNode)
    assert root.obj is leaf
    assert root.dependencies == []
    assert root.dependents == []

    assert coordinator.nodes_by_id == {leaf.object_id: root}
    assert coordinator.blocked == {}


def test_add_to_dag_traverses_declared_refs_recursively():
    leaf_a = Leaf(name="a")
    leaf_b = Leaf(name="b")
    mid_left = Mid(label="L", child=leaf_a)
    mid_right = Mid(label="R", child=leaf_b)
    top = Top(name="t", left=mid_left, right=mid_right)

    coordinator = _new_coordinator([top])

    assert set(coordinator.ready) == {leaf_a.object_id, leaf_b.object_id}
    assert set(coordinator.blocked) == {
        mid_left.object_id,
        mid_right.object_id,
        top.object_id,
    }

    assert set(coordinator.nodes_by_id) == {
        leaf_a.object_id,
        leaf_b.object_id,
        mid_left.object_id,
        mid_right.object_id,
        top.object_id,
    }

    leaf_a_node = coordinator.nodes_by_id[leaf_a.object_id]
    mid_left_node = coordinator.nodes_by_id[mid_left.object_id]
    top_node = coordinator.nodes_by_id[top.object_id]

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

    coordinator = _new_coordinator([top])

    assert len(coordinator.ready) == 1
    (shared_root,) = coordinator.ready.values()
    assert shared_root.obj is shared
    assert shared_root.dependencies == []
    assert {n.obj.object_id for n in shared_root.dependents} == {
        mid_left.object_id,
        mid_right.object_id,
    }

    assert set(coordinator.nodes_by_id) == {
        shared.object_id,
        mid_left.object_id,
        mid_right.object_id,
        top.object_id,
    }
    assert coordinator.nodes_by_id[shared.object_id] is shared_root


def test_add_to_dag_stops_recursion_at_completed_objects():
    leaf = Leaf(name="cached")
    mid = Mid(label="m", child=leaf)
    leaf.create()
    assert leaf.status() == "completed"

    coordinator = _new_coordinator([mid])

    assert len(coordinator.ready) == 1
    (mid_root,) = coordinator.ready.values()
    assert mid_root.obj is mid
    assert mid_root.dependencies == []

    assert set(coordinator.nodes_by_id) == {mid.object_id}
    assert mid_root.dependents == []


def test_add_to_dag_completed_root_has_no_dependencies():
    leaf = Leaf(name="root-cached")
    mid = Mid(label="m", child=leaf)
    mid.create()
    assert mid.status() == "completed"

    coordinator = _new_coordinator([mid])

    assert coordinator.ready == {}
    assert coordinator.nodes_by_id == {}
    assert coordinator.blocked == {}


def test_add_to_dag_rejects_running_root():
    leaf = Leaf(name="running-root")

    with mark_running(leaf):
        assert leaf.status() == "running"

        with pytest.raises(RuntimeError, match="cannot add running object to DAG"):
            _new_coordinator([leaf])


def test_add_to_dag_rejects_running_dependency():
    leaf = Leaf(name="running-dependency")
    mid = Mid(label="m", child=leaf)

    with mark_running(leaf):
        assert leaf.status() == "running"

        with pytest.raises(RuntimeError, match="cannot add running object to DAG"):
            _new_coordinator([mid])


def test_add_to_dag_does_not_reject_inactive_compute_lock():
    leaf = Leaf(name="inactive-lock")
    leaf._base_dir.mkdir(parents=True, exist_ok=True)
    lock_path = compute_lock_path_in(leaf._base_dir)
    lock_path.touch()

    assert leaf.status() == "failed"
    coordinator = _new_coordinator([leaf])

    assert set(coordinator.ready) == {leaf.object_id}


def test_add_to_dag_accepts_a_list_of_inputs():
    leaf_a = Leaf(name="a")
    leaf_b = Leaf(name="b")
    mid = Mid(label="m", child=leaf_a)

    coordinator = _new_coordinator([mid, leaf_b])

    assert set(coordinator.ready) == {leaf_a.object_id, leaf_b.object_id}

    assert set(coordinator.nodes_by_id) == {
        leaf_a.object_id,
        leaf_b.object_id,
        mid.object_id,
    }

    leaf_b_node = coordinator.nodes_by_id[leaf_b.object_id]
    assert leaf_b_node.dependents == []


def test_add_to_dag_handles_nested_dataclass_refs():
    leaf_a = Leaf(name="a")
    leaf_b = Leaf(name="b")
    parent = NestedParent(bundle=LeafBundle(a=leaf_a, b=leaf_b))

    coordinator = _new_coordinator([parent])

    assert set(coordinator.ready) == {leaf_a.object_id, leaf_b.object_id}

    assert set(coordinator.nodes_by_id) == {
        leaf_a.object_id,
        leaf_b.object_id,
        parent.object_id,
    }


def test_add_to_dag_walks_computed_dependencies():
    parent = ComputedParent(name="p")

    coordinator = _new_coordinator([parent])

    assert len(coordinator.ready) == 1
    (child_root,) = coordinator.ready.values()
    assert child_root.obj.object_id == parent.computed_child.object_id
    assert {n.obj.object_id for n in child_root.dependents} == {parent.object_id}
    assert set(coordinator.nodes_by_id) == {
        parent.object_id,
        parent.computed_child.object_id,
    }


def test_add_to_dag_rejects_non_furu_values():
    with pytest.raises(TypeError, match="expected Furu objects"):
        _new_coordinator(
            [Leaf(name="ok"), "not-a-furu"]  # ty: ignore[invalid-argument-type]
        )


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
        return self.child.create() + 1


class LazyChildLoader(Furu[int]):
    base: int
    create_calls: ClassVar[list[int]] = []

    def create(self) -> int:
        type(self).create_calls.append(self.base)
        return self.base + TrackingLeaf(n=self.base).create()


class AlwaysFails(Furu[int]):
    name: str

    def create(self) -> int:
        raise RuntimeError(f"intentional failure: {self.name}")


class DependsOnFailing(Furu[int]):
    label: str
    child: AlwaysFails

    def create(self) -> int:
        return self.child.create() + 1


@pytest.fixture(autouse=True)
def _reset_tracking() -> None:
    TrackingLeaf.create_calls.clear()
    TrackingMid.create_calls.clear()
    LazyChildLoader.create_calls.clear()


def test_execution_coordinator_run_runs_single_zero_dependency_node():
    leaf = TrackingLeaf(n=3)

    ExecutionCoordinator.run([leaf], worker_backends=(LocalThreadWorkerBackend(),))

    assert TrackingLeaf.create_calls == [3]
    assert leaf.status() == "completed"
    assert leaf.create() == 6


def test_execution_coordinator_run_runs_static_dependencies_in_order():
    leaf = TrackingLeaf(n=4)
    mid = TrackingMid(label="m", child=leaf)

    ExecutionCoordinator.run([mid], worker_backends=(LocalThreadWorkerBackend(),))

    assert TrackingLeaf.create_calls == [4]
    assert TrackingMid.create_calls == ["m"]
    assert mid.create() == 9


def test_execution_coordinator_run_handles_shared_dependency_only_once():
    shared = TrackingLeaf(n=5)
    left = TrackingMid(label="L", child=shared)
    right = TrackingMid(label="R", child=shared)

    ExecutionCoordinator.run(
        [left, right], worker_backends=(LocalThreadWorkerBackend(),)
    )

    assert TrackingLeaf.create_calls == [5]
    assert sorted(TrackingMid.create_calls) == ["L", "R"]


def test_execution_coordinator_run_with_multiple_workers_runs_independent_nodes():
    leaves = [TrackingLeaf(n=i) for i in range(8)]

    ExecutionCoordinator.run(
        leaves, worker_backends=(LocalThreadWorkerBackend(max_workers=4),)
    )

    assert sorted(TrackingLeaf.create_calls) == list(range(8))
    for leaf in leaves:
        assert leaf.status() == "completed"


def test_execution_coordinator_run_discovers_lazy_dependencies_and_reruns_parent():
    parent = LazyChildLoader(base=7)

    ExecutionCoordinator.run([parent], worker_backends=(LocalThreadWorkerBackend(),))

    assert TrackingLeaf.create_calls == [7]
    # Parent's create() is called once to discover the lazy dep (raising
    # _DependencyNotReady), then once more after the dep completes.
    assert LazyChildLoader.create_calls == [7, 7]
    assert parent.create() == 21
    parent_log = run_log_path_in(parent._base_dir).read_text(encoding="utf-8")
    assert (
        "create deferred: create discovered 1 missing dependency/dependencies"
        in parent_log
    )
    assert "create failed" not in parent_log
    assert "=== Debug Details (with locals) ===" not in parent_log


def test_execution_coordinator_run_skips_already_completed_objects():
    leaf = TrackingLeaf(n=8)
    leaf.create()
    TrackingLeaf.create_calls.clear()
    mid = TrackingMid(label="cached-child", child=leaf)

    ExecutionCoordinator.run([mid], worker_backends=(LocalThreadWorkerBackend(),))

    assert TrackingLeaf.create_calls == []
    assert TrackingMid.create_calls == ["cached-child"]


def test_execution_coordinator_run_reports_worker_failures():
    failing = AlwaysFails(name="boom")
    parent = DependsOnFailing(label="p", child=failing)

    with pytest.raises(RuntimeError, match="failed jobs"):
        ExecutionCoordinator.run(
            [parent],
            max_retries_per_object=0,
            worker_backends=(LocalThreadWorkerBackend(),),
        )

    assert failing.status() == "failed"
    assert parent.status() == "missing"
