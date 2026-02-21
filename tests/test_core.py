import types
from ast import excepthandler
from copy import deepcopy
from dataclasses import FrozenInstanceError, dataclass, is_dataclass
from enum import Enum
from functools import partial
from typing import Generic, Literal, TypeVar, get_args, get_origin

import pytest

from furu import Furu
from furu.core import CLASSMARKER

T = TypeVar("T")


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

    def _load(self):
        pass


class B(Furu, Generic[T]):
    a: A[T] | int
    y: dict[Literal["ney", "hey"] | bool, int]
    t: tuple[int | str, float]

    def _create(self):
        pass

    def _load(self):
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
        ).furu_hash
        != NodePair(
            name="z", node1=Node(name="y"), node2=WeightedNode(name="z", weight=1)
        ).furu_hash
    )

    assert (
        NodePair(
            name="y", node1=Node(name="y"), node2=WeightedNode(name="z", weight=1)
        ).furu_schema_hash
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
        ).furu_schema_hash
        != B_priv(
            a=A(x=1, z="123", w=[6, 7]), y={"hey": 123, True: 1}, t=("123", 12), _h=1
        ).furu_schema_hash
    )

    def qualname_alias(cls: type, *, qualname: str) -> type:
        alias = type(qualname, (cls,), {"__module__": cls.__module__})
        alias.__qualname__ = qualname
        return alias

    B_priv_as_B = qualname_alias(B_priv, qualname="B")
    assert (
        B(
            a=A(x=1, z="123", w=[6, 7]), y={"ney": 123, True: 1}, t=("123", 12)
        ).furu_schema_hash
        == B_priv_as_B(
            a=A(x=1, z="123", w=[6, 7]), y={"123": 123, True: 1}, t=("123", 12), _h=1
        ).furu_schema_hash
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
def test_furu_schema(make, expected):
    print(make().furu_schema)
    assert make().furu_schema == expected
