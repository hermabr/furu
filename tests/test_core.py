from dataclasses import FrozenInstanceError, is_dataclass

import pytest

from furu import Furu


class Node(Furu[int]):
    name: str

    def _create(self):
        pass

    def _load(self):
        pass


class WeightedNode(Node):
    weight: float


class NodePair(Furu[int]):
    node1: Node
    node2: WeightedNode
    name: str | int

    def _create(self):
        pass

    def _load(self):
        pass


def test_frozen_dataclass_inheritance():
    for cls in [Node, WeightedNode]:
        if cls == Node:
            obj = cls(name="x")
        else:
            obj = cls(name="x", weight=1.5)
        assert obj.name == "x"
        if isinstance(obj, WeightedNode):
            assert obj.weight == 1.5

        assert is_dataclass(cls)

        with pytest.raises(TypeError):
            type.__call__(cls, 1, 2)
        with pytest.raises(FrozenInstanceError):
            setattr(obj, "a", 3)
        with pytest.raises(TypeError):
            cls(1, 2)  # ty: ignore[missing-argument,too-many-positional-arguments]
        with pytest.raises(FrozenInstanceError):
            obj.a = 3  # ty: ignore[invalid-assignment]


def test_furu_hash_and_dir():
    assert (
        NodePair(
            name="x", node1=Node(name="y"), node2=WeightedNode(name="z", weight=1)
        ).furu_hash
        == "997d6e006e7621cff809"
    )

    assert (
        NodePair(
            name="x", node1=Node(name="y"), node2=WeightedNode(name="z", weight=1)
        ).furu_schema_hash
        == "9b169f9e51cfa7bbac828bdad6684b09b41d63c1a76f99ca7b9cb56ecc4952f5"
    )

    # TODO: check what happens if a furu object has itself as a field type, such as Fib
