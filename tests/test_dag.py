from dataclasses import dataclass

import pytest

import furu
from furu import Furu, FuruDag, make_dag


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


def test_make_dag_single_object_no_dependencies():
    leaf = Leaf(name="x")
    dag = make_dag(leaf)

    assert isinstance(dag, FuruDag)
    assert dag.nodes == {leaf.object_id: leaf}
    assert dag.dependencies == {leaf.object_id: ()}


def test_make_dag_traverses_declared_refs_recursively():
    leaf_a = Leaf(name="a")
    leaf_b = Leaf(name="b")
    mid_left = Mid(label="L", child=leaf_a)
    mid_right = Mid(label="R", child=leaf_b)
    top = Top(name="t", left=mid_left, right=mid_right)

    dag = make_dag(top)

    assert set(dag.nodes) == {
        top.object_id,
        mid_left.object_id,
        mid_right.object_id,
        leaf_a.object_id,
        leaf_b.object_id,
    }
    assert set(dag.dependencies[top.object_id]) == {
        mid_left.object_id,
        mid_right.object_id,
    }
    assert dag.dependencies[mid_left.object_id] == (leaf_a.object_id,)
    assert dag.dependencies[mid_right.object_id] == (leaf_b.object_id,)
    assert dag.dependencies[leaf_a.object_id] == ()
    assert dag.dependencies[leaf_b.object_id] == ()


def test_make_dag_deduplicates_shared_refs_by_object_id():
    shared = Leaf(name="shared")
    mid_left = Mid(label="L", child=shared)
    mid_right = Mid(label="R", child=shared)
    top = Top(name="t", left=mid_left, right=mid_right)

    dag = make_dag(top)

    assert set(dag.nodes) == {
        top.object_id,
        mid_left.object_id,
        mid_right.object_id,
        shared.object_id,
    }
    assert dag.dependencies[mid_left.object_id] == (shared.object_id,)
    assert dag.dependencies[mid_right.object_id] == (shared.object_id,)
    assert dag.dependencies[shared.object_id] == ()


def test_make_dag_stops_recursion_at_completed_objects():
    leaf = Leaf(name="cached")
    mid = Mid(label="m", child=leaf)
    leaf.load_or_create()
    assert leaf.status() == "completed"

    dag = make_dag(mid)

    assert set(dag.nodes) == {mid.object_id, leaf.object_id}
    assert dag.dependencies[mid.object_id] == (leaf.object_id,)
    assert dag.dependencies[leaf.object_id] == ()


def test_make_dag_includes_completed_root_but_does_not_expand_it():
    leaf = Leaf(name="root-cached")
    mid = Mid(label="m", child=leaf)
    mid.load_or_create()
    assert mid.status() == "completed"

    dag = make_dag(mid)

    assert dag.nodes == {mid.object_id: mid}
    assert dag.dependencies == {mid.object_id: ()}


def test_make_dag_accepts_a_list_of_roots():
    leaf_a = Leaf(name="a")
    leaf_b = Leaf(name="b")
    mid = Mid(label="m", child=leaf_a)

    dag = make_dag([mid, leaf_b])

    assert set(dag.nodes) == {mid.object_id, leaf_a.object_id, leaf_b.object_id}
    assert dag.dependencies[mid.object_id] == (leaf_a.object_id,)
    assert dag.dependencies[leaf_b.object_id] == ()


def test_make_dag_handles_nested_dataclass_refs():
    leaf_a = Leaf(name="a")
    leaf_b = Leaf(name="b")
    parent = NestedParent(bundle=LeafBundle(a=leaf_a, b=leaf_b))

    dag = make_dag(parent)

    assert set(dag.nodes) == {parent.object_id, leaf_a.object_id, leaf_b.object_id}
    assert set(dag.dependencies[parent.object_id]) == {
        leaf_a.object_id,
        leaf_b.object_id,
    }


def test_make_dag_walks_computed_dependencies():
    parent = ComputedParent(name="p")

    dag = make_dag(parent)

    assert set(dag.nodes) == {parent.object_id, parent.computed_child.object_id}
    assert dag.dependencies[parent.object_id] == (parent.computed_child.object_id,)


def test_make_dag_empty_list_returns_empty_dag():
    dag = make_dag([])

    assert dag.nodes == {}
    assert dag.dependencies == {}


def test_make_dag_rejects_non_furu_values():
    with pytest.raises(TypeError, match="expected Furu objects"):
        make_dag([Leaf(name="ok"), "not-a-furu"])  # ty: ignore[invalid-argument-type]

    with pytest.raises(
        TypeError, match="expected a Furu object or a sequence of Furu objects"
    ):
        make_dag(42)  # ty: ignore[invalid-argument-type]
