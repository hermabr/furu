from dataclasses import dataclass

import pytest

import furu
from furu import Furu
from furu.dag import FuruDagNode, make_execution_dag


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


def test_make_execution_dag_single_object_no_dependencies():
    leaf = Leaf(name="x")
    nodes_by_id: dict[str, FuruDagNode[Furu]] = {}
    zero_dep = make_execution_dag([leaf], nodes_by_id)

    assert len(zero_dep) == 1
    (root,) = zero_dep
    assert isinstance(root, FuruDagNode)
    assert root.obj is leaf
    assert root.dependencies == []
    assert root.dependents == []

    assert nodes_by_id == {leaf.object_id: root}


def test_make_execution_dag_traverses_declared_refs_recursively():
    leaf_a = Leaf(name="a")
    leaf_b = Leaf(name="b")
    mid_left = Mid(label="L", child=leaf_a)
    mid_right = Mid(label="R", child=leaf_b)
    top = Top(name="t", left=mid_left, right=mid_right)

    nodes_by_id: dict[str, FuruDagNode[Furu]] = {}
    zero_dep = make_execution_dag([top], nodes_by_id)

    zero_dep_ids = {node.obj.object_id for node in zero_dep}
    assert zero_dep_ids == {leaf_a.object_id, leaf_b.object_id}

    assert set(nodes_by_id) == {
        leaf_a.object_id,
        leaf_b.object_id,
        mid_left.object_id,
        mid_right.object_id,
        top.object_id,
    }

    leaf_a_node = nodes_by_id[leaf_a.object_id]
    mid_left_node = nodes_by_id[mid_left.object_id]
    top_node = nodes_by_id[top.object_id]

    assert leaf_a_node.dependencies == []
    assert leaf_a_node.dependents == [mid_left_node]
    assert mid_left_node.dependencies == [leaf_a_node]
    assert mid_left_node.dependents == [top_node]
    assert {dep.obj.object_id for dep in top_node.dependencies} == {
        mid_left.object_id,
        mid_right.object_id,
    }
    assert top_node.dependents == []


def test_make_execution_dag_shared_dependency_has_multiple_dependents():
    shared = Leaf(name="shared")
    mid_left = Mid(label="L", child=shared)
    mid_right = Mid(label="R", child=shared)
    top = Top(name="t", left=mid_left, right=mid_right)

    nodes_by_id: dict[str, FuruDagNode[Furu]] = {}
    zero_dep = make_execution_dag([top], nodes_by_id)

    assert len(zero_dep) == 1
    (shared_root,) = zero_dep
    assert shared_root.obj is shared
    assert shared_root.dependencies == []
    assert {n.obj.object_id for n in shared_root.dependents} == {
        mid_left.object_id,
        mid_right.object_id,
    }

    assert set(nodes_by_id) == {
        shared.object_id,
        mid_left.object_id,
        mid_right.object_id,
        top.object_id,
    }
    assert nodes_by_id[shared.object_id] is shared_root


def test_make_execution_dag_stops_recursion_at_completed_objects():
    leaf = Leaf(name="cached")
    mid = Mid(label="m", child=leaf)
    leaf.load_or_create()
    assert leaf.status() == "completed"

    nodes_by_id: dict[str, FuruDagNode[Furu]] = {}
    zero_dep = make_execution_dag([mid], nodes_by_id)

    assert len(zero_dep) == 1
    (leaf_root,) = zero_dep
    assert leaf_root.obj is leaf
    assert leaf_root.dependencies == []

    assert set(nodes_by_id) == {leaf.object_id, mid.object_id}
    assert {n.obj.object_id for n in leaf_root.dependents} == {mid.object_id}


def test_make_execution_dag_completed_root_has_no_dependencies():
    leaf = Leaf(name="root-cached")
    mid = Mid(label="m", child=leaf)
    mid.load_or_create()
    assert mid.status() == "completed"

    nodes_by_id: dict[str, FuruDagNode[Furu]] = {}
    zero_dep = make_execution_dag([mid], nodes_by_id)

    assert len(zero_dep) == 1
    (root,) = zero_dep
    assert root.obj is mid
    assert root.dependencies == []
    assert root.dependents == []
    assert nodes_by_id == {mid.object_id: root}


def test_make_execution_dag_accepts_a_list_of_inputs():
    leaf_a = Leaf(name="a")
    leaf_b = Leaf(name="b")
    mid = Mid(label="m", child=leaf_a)

    nodes_by_id: dict[str, FuruDagNode[Furu]] = {}
    zero_dep = make_execution_dag([mid, leaf_b], nodes_by_id)

    zero_dep_ids = {node.obj.object_id for node in zero_dep}
    assert zero_dep_ids == {leaf_a.object_id, leaf_b.object_id}

    assert set(nodes_by_id) == {leaf_a.object_id, leaf_b.object_id, mid.object_id}

    leaf_b_node = nodes_by_id[leaf_b.object_id]
    assert leaf_b_node.dependents == []


def test_make_execution_dag_handles_nested_dataclass_refs():
    leaf_a = Leaf(name="a")
    leaf_b = Leaf(name="b")
    parent = NestedParent(bundle=LeafBundle(a=leaf_a, b=leaf_b))

    nodes_by_id: dict[str, FuruDagNode[Furu]] = {}
    zero_dep = make_execution_dag([parent], nodes_by_id)

    zero_dep_ids = {node.obj.object_id for node in zero_dep}
    assert zero_dep_ids == {leaf_a.object_id, leaf_b.object_id}

    assert set(nodes_by_id) == {leaf_a.object_id, leaf_b.object_id, parent.object_id}


def test_make_execution_dag_walks_computed_dependencies():
    parent = ComputedParent(name="p")

    nodes_by_id: dict[str, FuruDagNode[Furu]] = {}
    zero_dep = make_execution_dag([parent], nodes_by_id)

    assert len(zero_dep) == 1
    (child_root,) = zero_dep
    assert child_root.obj.object_id == parent.computed_child.object_id
    assert {n.obj.object_id for n in child_root.dependents} == {parent.object_id}
    assert set(nodes_by_id) == {parent.object_id, parent.computed_child.object_id}


def test_make_execution_dag_empty_list_returns_empty_results():
    nodes_by_id: dict[str, FuruDagNode[Furu]] = {}
    zero_dep = make_execution_dag([], nodes_by_id)

    assert zero_dep == []
    assert nodes_by_id == {}


def test_make_execution_dag_rejects_non_furu_values():
    with pytest.raises(TypeError, match="expected Furu objects"):
        make_execution_dag([Leaf(name="ok"), "not-a-furu"], {})  # ty: ignore[invalid-argument-type]
