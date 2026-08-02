from dataclasses import dataclass
from typing import cast

import pytest
from pydantic import BaseModel

import furu
from furu._pytree import tree_leaves, tree_map
from furu.worker.backends.local import LocalThreadWorkerBackend


class Number(furu.Spec[int]):
    value: int

    def create(self) -> int:
        return self.value * 10


@dataclass(frozen=True)
class Pair:
    left: object
    right: object


class Box(BaseModel):
    item: object


def test_tree_leaves_and_identity_map_preserve_the_tree() -> None:
    tree = {"list": [1, Pair(2, 3)], "tuple": (4,), "set": {6, 5}}

    assert tree_leaves(tree) == [1, 2, 3, 4, 5, 6]
    assert tree_map(lambda leaf: leaf, tree) == tree


def test_tree_map_is_composed_from_flatten_and_unflatten() -> None:
    tree = Pair(left=[1, 2], right=Box(item=(3, 4)))

    mapped = tree_map(lambda value: cast(int, value) * 10, tree)

    assert mapped == Pair(left=[10, 20], right=Box(item=(30, 40)))


def test_create_preserves_an_arbitrary_pytree_shape() -> None:
    tree = {
        "pair": Pair(Number(value=1), Number(value=2)),
        "box": Box(item=Number(value=3)),
        "tuple": (Number(value=4),),
    }

    created = furu.create(tree)

    assert created == {
        "pair": Pair(10, 20),
        "box": Box(item=30),
        "tuple": (40,),
    }
    assert furu.load_existing(tree) == created


def test_create_treats_a_spec_as_a_leaf_not_as_its_dataclass_fields() -> None:
    assert furu.create(Number(value=7)) == 70


def test_create_on_worker_backend_accepts_pytree_roots() -> None:
    tree = {"left": Number(value=8), "right": (Number(value=9),)}

    assert furu.create(tree, on=(LocalThreadWorkerBackend(),)) == {
        "left": 80,
        "right": (90,),
    }


@pytest.mark.parametrize("tree", [{"bad": 1}, [Number(value=1), "bad"]])
def test_create_rejects_non_spec_leaves(tree: object) -> None:
    with pytest.raises(TypeError, match="PyTree of Spec objects"):
        furu.create(tree)


@pytest.mark.parametrize("tree", [[], (), {}, Pair([], {})])
def test_create_preserves_empty_pytrees(tree: object) -> None:
    assert furu.create(tree) == tree
