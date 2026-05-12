from dataclasses import dataclass
from typing import ClassVar

import pytest

import furu
from furu import Furu
from furu.execution.manager import DagNode, Manager
from furu.storage_layout import run_log_path_in


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


def test_add_to_dag_single_object_no_dependencies():
    leaf = Leaf(name="x")
    manager = Manager([])
    manager._add_to_dag([leaf])

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

    manager = Manager([])
    manager._add_to_dag([top])

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

    manager = Manager([])
    manager._add_to_dag([top])

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

    manager = Manager([])
    manager._add_to_dag([mid])

    assert len(manager.ready) == 1
    (leaf_root,) = manager.ready.values()
    assert leaf_root.obj is leaf
    assert leaf_root.dependencies == []

    assert set(manager.nodes_by_id) == {leaf.object_id, mid.object_id}
    assert {n.obj.object_id for n in leaf_root.dependents} == {mid.object_id}


def test_add_to_dag_completed_root_has_no_dependencies():
    leaf = Leaf(name="root-cached")
    mid = Mid(label="m", child=leaf)
    mid.load_or_create()
    assert mid.status() == "completed"

    manager = Manager([])
    manager._add_to_dag([mid])

    assert len(manager.ready) == 1
    (root,) = manager.ready.values()
    assert root.obj is mid
    assert root.dependencies == []
    assert root.dependents == []
    assert manager.nodes_by_id == {mid.object_id: root}
    assert manager.blocked == {}


def test_add_to_dag_accepts_a_list_of_inputs():
    leaf_a = Leaf(name="a")
    leaf_b = Leaf(name="b")
    mid = Mid(label="m", child=leaf_a)

    manager = Manager([])
    manager._add_to_dag([mid, leaf_b])

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

    manager = Manager([])
    manager._add_to_dag([parent])

    assert set(manager.ready) == {leaf_a.object_id, leaf_b.object_id}

    assert set(manager.nodes_by_id) == {
        leaf_a.object_id,
        leaf_b.object_id,
        parent.object_id,
    }


def test_add_to_dag_walks_computed_dependencies():
    parent = ComputedParent(name="p")

    manager = Manager([])
    manager._add_to_dag([parent])

    assert len(manager.ready) == 1
    (child_root,) = manager.ready.values()
    assert child_root.obj.object_id == parent.computed_child.object_id
    assert {n.obj.object_id for n in child_root.dependents} == {parent.object_id}
    assert set(manager.nodes_by_id) == {
        parent.object_id,
        parent.computed_child.object_id,
    }


def test_add_to_dag_empty_list_is_noop():
    manager = Manager([])
    manager._add_to_dag([])

    assert manager.ready == {}
    assert manager.blocked == {}
    assert manager.nodes_by_id == {}


def test_add_to_dag_rejects_non_furu_values():
    manager = Manager([])
    with pytest.raises(TypeError, match="expected Furu objects"):
        manager._add_to_dag([Leaf(name="ok"), "not-a-furu"])  # ty: ignore[invalid-argument-type]


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


@pytest.fixture(autouse=True)
def _reset_tracking() -> None:
    TrackingLeaf.create_calls.clear()
    TrackingMid.create_calls.clear()
    LazyChildLoader.create_calls.clear()


def test_manager_run_runs_single_zero_dependency_node():
    leaf = TrackingLeaf(n=3)

    Manager([leaf]).run()

    assert TrackingLeaf.create_calls == [3]
    assert leaf.status() == "completed"
    assert leaf.load_or_create() == 6


def test_manager_run_runs_static_dependencies_in_order():
    leaf = TrackingLeaf(n=4)
    mid = TrackingMid(label="m", child=leaf)

    Manager([mid]).run()

    assert TrackingLeaf.create_calls == [4]
    assert TrackingMid.create_calls == ["m"]
    assert mid.load_or_create() == 9


def test_manager_run_handles_shared_dependency_only_once():
    shared = TrackingLeaf(n=5)
    left = TrackingMid(label="L", child=shared)
    right = TrackingMid(label="R", child=shared)

    Manager([left, right]).run()

    assert TrackingLeaf.create_calls == [5]
    assert sorted(TrackingMid.create_calls) == ["L", "R"]


def test_manager_run_discovers_lazy_dependencies_and_reruns_parent():
    parent = LazyChildLoader(base=7)

    Manager([parent]).run()

    assert TrackingLeaf.create_calls == [7]
    # Parent's create() is called once to discover the lazy dep (raising
    # _DependencyNotReady), then once more after the dep completes.
    assert LazyChildLoader.create_calls == [7, 7]
    assert parent.load_or_create() == 21
    parent_log = run_log_path_in(parent.data_dir).read_text(encoding="utf-8")
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

    Manager([mid]).run()

    assert TrackingLeaf.create_calls == []
    assert TrackingMid.create_calls == ["cached-child"]


def test_manager_run_empty_list_is_noop():
    Manager([]).run()
