import types
from dataclasses import FrozenInstanceError, is_dataclass, replace
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Generic, Literal, TypeVar

import pytest

from furu import Furu
from furu.config import config
from furu.serialize import to_json

T = TypeVar("T")


class Node(Furu[str]):
    name: str

    def _create(self) -> str:
        return f"Node({self.name})"


class WeightedNode(Node):
    weight: float

    def _create(self) -> str:
        return f"WNode({self.name}:{self.weight})"


class NodePair(Furu[dict]):
    node1: Node
    node2: WeightedNode
    name: str | int

    def _create(self) -> dict:
        return {
            "node1": self.node1.load_or_create(),
            "node2": self.node2.load_or_create(),
            "name": self.name,
        }


class RandomObj(Furu[float]):
    id: int

    def _create(self) -> float:
        import random

        return random.random()


class Fruit(Enum):
    apple = "apple"
    banana = "banana"


class A(Furu, Generic[T]):
    x: int | str | list
    z: T
    w: list[int | float]
    fruit: Fruit = Fruit("banana")

    def _create(self):
        pass


class B(Furu, Generic[T]):
    a: A[T] | int
    y: dict[Literal["ney", "hey"] | bool, int]
    t: tuple[int | str, float]

    def _create(self):
        pass

    @classmethod
    def with_hidden_field(cls):
        cached = getattr(cls, "__hidden_variant__", None)
        if cached is not None:
            return cached

        ann = dict(getattr(cls, "__annotations__", {}))
        ann["_hidden"] = int

        Hidden = types.new_class(
            cls.__name__,
            (cls,),
            exec_body=lambda ns: ns.update(
                {
                    "__annotations__": ann,
                    "_hidden": 0,
                }
            ),
        )
        Hidden.__module__ = cls.__module__
        Hidden.__qualname__ = cls.__qualname__  # <- the “same qualname” lie

        cls.__hidden_variant__ = Hidden
        return Hidden


class B_priv(B):
    _h: int


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


def test_hashes_and_data_dir():
    assert (
        NodePair(
            name="x", node1=Node(name="y"), node2=WeightedNode(name="z", weight=1)
        ).artifact_hash
        == "997d6e006e7621cff809"
    )

    assert (
        NodePair(
            name="x", node1=Node(name="y"), node2=WeightedNode(name="z", weight=1)
        ).artifact_hash
        != NodePair(
            name="z", node1=Node(name="y"), node2=WeightedNode(name="z", weight=1)
        ).artifact_hash
    )

    assert (
        NodePair(
            name="y", node1=Node(name="y"), node2=WeightedNode(name="z", weight=1)
        ).schema_hash
        == "50a9b8624ed259ec38df"
    )

    assert NodePair(
        name="y", node1=Node(name="y"), node2=WeightedNode(name="z", weight=1)
    ) != NodePair(name=67, node1=Node(name="y"), node2=WeightedNode(name="z", weight=1))

    assert NodePair(
        node1=Node(name="y"),
        node2=WeightedNode(
            weight=1,
            name="z",
        ),
        name="y",
    ) != NodePair(
        node1=Node(name="y"),
        node2=WeightedNode(name="z", weight=1),
        name=67,
    )

    assert (
        B(
            a=A(x=1, z="123", w=[6, 7]), y={"hey": 123, True: 1}, t=("123", 12)
        ).schema_hash
        != B_priv(
            a=A(x=1, z="123", w=[6, 7]),
            y={"hey": 123, True: 1},
            t=("123", 12),
            _h=1,
        ).schema_hash
    )

    def qualname_alias[T](cls: T, *, ret_typ: type) -> T:
        alias = type(ret_typ.__qualname__, (cls,), {"__module__": cls.__module__})  # ty: ignore[invalid-base]
        alias.__qualname__ = ret_typ.__qualname__
        return alias  # ty: ignore[invalid-return-type]

    B_priv_as_B = qualname_alias(B_priv, ret_typ=B)
    assert (
        B(
            a=A(x=1, z="123", w=[6, 7]), y={"ney": 123, True: 1}, t=("123", 12)
        ).schema_hash
        == B_priv_as_B(
            a=A(x=1, z="123", w=[6, 7]),
            y={"hey": 123, "ney": 1},
            t=("123", 12),
            _h=1,
        ).schema_hash
    )


def expected_schema_for_B_like(cls_name: str) -> dict:
    return {
        "|class": f"test_core.{cls_name}",
        "fields": {
            "a": [
                "builtins.int",
                {
                    "|class": "test_core.A",
                    "fields": {
                        "fruit": "test_core.Fruit",
                        "w": {
                            "|origin": "builtins.list",
                            "|args": [["builtins.float", "builtins.int"]],
                        },
                        "x": ["builtins.int", "builtins.list", "builtins.str"],
                        "z": "~T",
                    },
                },
            ],
            "t": {
                "|origin": "builtins.tuple",
                "|args": ["builtins.float", ["builtins.int", "builtins.str"]],
            },
            "y": {
                "|origin": "builtins.dict",
                "|args": [
                    "builtins.int",
                    [
                        "builtins.bool",
                        {"|origin": "typing.Literal", "|args": ["hey", "ney"]},
                    ],
                ],
            },
        },
    }


@pytest.mark.parametrize(
    "make, expected",
    [
        pytest.param(
            lambda: B(
                a=A(x=1, z="123", w=[6, 7]), y={"hey": 123, True: 1}, t=("123", 12)
            ),
            expected_schema_for_B_like("B"),
            id="B",
        ),
        pytest.param(
            lambda: partial(B_priv, _h=1)(
                a=A(x=1, z="123", w=[6, 7]), y=["123", True], t=("123", 12)
            ),
            expected_schema_for_B_like("B_priv"),
            id="B_priv",
        ),
        pytest.param(
            lambda: NodePair(
                node1=Node(name="x"),
                node2=WeightedNode(name="z", weight=2),
                name="name",
            ),
            {
                "|class": "test_core.NodePair",
                "fields": {
                    "name": ["builtins.int", "builtins.str"],
                    "node1": {
                        "|class": "test_core.Node",
                        "fields": {"name": "builtins.str"},
                    },
                    "node2": {
                        "|class": "test_core.WeightedNode",
                        "fields": {"name": "builtins.str", "weight": "builtins.float"},
                    },
                },
            },
            id="NodePair",
        ),
    ],
)
def test_schema(make: type[Furu], expected):
    assert make().schema == expected


def test_to_json():
    node_pair = NodePair(
        name="x", node1=Node(name="y"), node2=WeightedNode(name="z", weight=1)
    )
    expected = {
        "|class": "test_core.NodePair",
        "node1": {"|class": "test_core.Node", "name": "y"},
        "node2": {"|class": "test_core.WeightedNode", "name": "z", "weight": 1},
        "name": "x",
    }
    assert to_json(node_pair) == expected
    assert to_json(node_pair) == node_pair.to_json()
    assert node_pair.to_json() == expected


def test_data_dir():
    node_pair = NodePair(
        name="x", node1=Node(name="y"), node2=WeightedNode(name="z", weight=1)
    )
    assert node_pair.data_dir == (
        config.directories.data
        / "test_core"
        / "NodePair"
        / "50a9b8624ed259ec38df"
        / "997d6e006e7621cff809"
    )
    assert node_pair.data_dir == Path(
        config.directories.data
        / "test_core"
        / "NodePair"
        / node_pair.schema_hash
        / node_pair.artifact_hash
    )


def test_create_object_and_exists():
    node_pair = NodePair(
        name="x", node1=Node(name="y"), node2=WeightedNode(name="z", weight=1)
    )
    assert not node_pair.exists()
    for i in range(3):
        assert node_pair.load_or_create() == {
            "node1": "Node(y)",
            "node2": "WNode(z:1)",
            "name": "x",
        }
    assert node_pair.exists()
    assert not replace(node_pair, name="y").exists()


def test_creating_and_loading_random_result_furu_obj():
    n_ids = 5
    results = {
        obj_id: [RandomObj(id=obj_id).load_or_create() for _ in range(3)]
        for obj_id in range(n_ids)
    }
    assert all(len(set(values)) == 1 for values in results.values())
    assert len({values[0] for values in results.values()}) == n_ids
