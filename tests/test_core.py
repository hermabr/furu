import types
from collections.abc import Callable
from dataclasses import FrozenInstanceError, is_dataclass, replace
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Literal, TypeVar
from unittest.mock import patch

import pytest

from furu import Furu
from furu.config import config
from furu.serialize import to_json
from furu.utils import fully_qualified_name

T = TypeVar("T")

type SOME_TYPE = Literal["a", "b"] | int


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
        # Intentionally non-deterministic to test that load_or_create caches results
        import random

        return random.random()


class Fruit(Enum):  # TODO: test enums at some point
    apple = "apple"
    banana = "banana"


class A[T](Furu):
    x: int | str | list
    z: T
    w: list[int | float]
    some_obj: SOME_TYPE = "a"

    def _create(self) -> None:
        pass


class B[T](Furu):
    a: A[T] | int
    y: dict[Literal["ney", "hey"] | bool, int]
    t: tuple[int | str, float]
    maybe_val: int | None = None

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


class VariadicTuple(Furu[None]):
    t: tuple[int, ...]

    def _create(self):
        pass


class UsesPath(Furu[str]):
    path: Path

    def _create(self) -> str:
        return str(self.path)


class UsesClassValue(Furu[None]):
    node_cls: type[Node]

    def _create(self) -> None:
        return None


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
        == "c4ff0c2ad0f653af7ce2"
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
                        "some_obj": [
                            "builtins.int",
                            {"|origin": "typing.Literal", "|args": ["a", "b"]},
                        ],
                        "w": {
                            "|origin": "builtins.list",
                            "|args": [["builtins.float", "builtins.int"]],
                        },
                        "x": ["builtins.int", "builtins.list", "builtins.str"],
                        "z": "T",
                    },
                },
            ],
            "maybe_val": ["builtins.NoneType", "builtins.int"],
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
        pytest.param(
            lambda: UsesPath(path=Path("/tmp/out")),
            {
                "|class": "test_core.UsesPath",
                "fields": {"path": fully_qualified_name(Path)},
            },
            id="UsesPath",
        ),
    ],
)
def test_schema(make: Callable[[], Furu], expected):
    assert make().schema == expected


def test_to_json():
    node_pair = NodePair(
        name="x", node1=Node(name="y"), node2=WeightedNode(name="z", weight=1)
    )
    expected = {
        "|kind": "instance",
        "|class": "test_core.NodePair",
        "fields": {
            "node1": {
                "|kind": "instance",
                "|class": "test_core.Node",
                "fields": {"name": "y"},
            },
            "node2": {
                "|kind": "instance",
                "|class": "test_core.WeightedNode",
                "fields": {"name": "z", "weight": 1},
            },
            "name": "x",
        },
    }
    assert to_json(node_pair) == expected
    assert to_json(node_pair) == node_pair.to_json()
    assert node_pair.to_json() == expected


def test_to_json_with_none_field():
    obj = B(
        a=A(x=1, z="123", w=[6, 7]),
        y={"hey": 123, "ney": 1},
        t=("123", 12),
        maybe_val=None,
    )

    expected = {
        "|kind": "instance",
        "|class": "test_core.B",
        "fields": {
            "a": {
                "|kind": "instance",
                "|class": "test_core.A",
                "fields": {"x": 1, "z": "123", "w": [6, 7], "some_obj": "a"},
            },
            "y": {"hey": 123, "ney": 1},
            "t": ["123", 12],
            "maybe_val": None,
        },
    }

    assert to_json(obj) == expected


def test_to_json_with_class_field_value():
    obj = UsesClassValue(node_cls=Node)

    assert to_json(Node) == {"|kind": "type_ref", "|class": "test_core.Node"}
    assert obj.to_json() == {
        "|kind": "instance",
        "|class": "test_core.UsesClassValue",
        "fields": {"node_cls": {"|kind": "type_ref", "|class": "test_core.Node"}},
    }
    assert isinstance(obj.artifact_hash, str)


def test_schema_with_ellipsis_type_arg():
    assert VariadicTuple(t=(1, 2, 3)).schema == {
        "|class": "test_core.VariadicTuple",
        "fields": {
            "t": {
                "|origin": "builtins.tuple",
                "|args": ["builtins.ellipsis", "builtins.int"],
            }
        },
    }


def test_data_dir():
    node_pair = NodePair(
        name="x", node1=Node(name="y"), node2=WeightedNode(name="z", weight=1)
    )
    assert node_pair.data_dir == (
        config.directories.data
        / "test_core"
        / "NodePair"
        / "50a9b8624ed259ec38df"
        / "c4ff0c2ad0f653af7ce2"
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
    assert not node_pair.is_completed()
    for _i in range(3):
        assert node_pair.load_or_create() == {
            "node1": "Node(y)",
            "node2": "WNode(z:1)",
            "name": "x",
        }
    assert node_pair.is_completed()
    assert not replace(node_pair, name="y").is_completed()


def test_creating_and_loading_random_result_furu_obj():
    n_ids = 5
    results = {
        obj_id: [RandomObj(id=obj_id).load_or_create() for _ in range(3)]
        for obj_id in range(n_ids)
    }
    assert all(len(set(values)) == 1 for values in results.values())
    assert len({values[0] for values in results.values()}) == n_ids


def test_delete_force() -> None:
    node = Node(name="x")

    assert node.load_or_create() == "Node(x)"
    assert node.data_dir.exists()
    assert node.delete(mode="force")
    assert not node.data_dir.exists()
    assert node.load_or_create() == "Node(x)"


def test_delete_prompt_cancel() -> None:
    node = Node(name="x")

    assert node.load_or_create() == "Node(x)"
    with patch("builtins.input", return_value="n"):
        assert not node.delete()
    assert node.data_dir.exists()


def test_delete_returns_false_when_missing() -> None:
    assert not Node(name="x").delete(mode="force")
