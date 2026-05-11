import json
import types
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, dataclass, is_dataclass, replace
from enum import Enum
from functools import cached_property, partial
from pathlib import Path
from typing import Any, ClassVar, Literal, cast
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ConfigDict

import furu
import furu.execution as execution_module
from furu import Furu, load_or_create, validate
from furu.config import config
from furu.locking import lock_many
from furu.metadata import ArtifactMetadata
from furu.result import load_result_bundle, save_result_bundle
from furu.result.codec import _default_result_registry
from furu.serialize import _from_json, to_json
from furu.utils import fully_qualified_name

type SOME_TYPE = Literal["a", "b"] | int


class Node(Furu[str]):
    name: str

    def create(self) -> str:
        return f"Node({self.name})"


class WeightedNode(Node):
    weight: float

    def create(self) -> str:
        return f"WNode({self.name}:{self.weight})"


class CustomStorageRootNode(Node):
    @cached_property
    def storage_root(self) -> Path:
        return Path("custom/data/location")


class NodePair(Furu[dict]):
    node1: Node
    node2: WeightedNode
    name: str | int

    def create(self) -> dict:
        return {
            "node1": self.node1.load_or_create(),
            "node2": self.node2.load_or_create(),
            "name": self.name,
        }


class RandomObj(Furu[float]):
    id: int

    def create(self) -> float:
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

    def create(self) -> None:
        pass


class B[T](Furu):
    a: A[T] | int
    y: dict[Literal["ney", "hey"] | bool, int]
    t: tuple[int | str, float]
    maybe_val: int | None = None

    def create(self):
        pass

    @classmethod
    def with_hidden_field(cls):
        cached = getattr(cls, "__hidden_variant__", None)
        if cached is not None:
            return cached

        ann = dict(getattr(cls, "__annotations__", {}))
        ann["_hidden"] = int
        namespace: dict[str, object] = {
            "__annotations__": ann,
            "_hidden": 0,
        }
        if cls._furu_create_mode == "single":
            namespace["create"] = lambda self: cls.create(self)
        else:
            namespace["_create_batched"] = classmethod(
                lambda hidden_cls, objs: cls._create_batched(objs)
            )

        Hidden = types.new_class(
            cls.__name__,
            (cls,),
            exec_body=lambda ns: ns.update(namespace),
        )
        Hidden.__module__ = cls.__module__
        Hidden.__qualname__ = cls.__qualname__  # <- the “same qualname” lie

        cls.__hidden_variant__ = Hidden
        return Hidden


class B_priv(B):
    _h: int

    def create(self):
        return super().create()


class VariadicTuple(Furu[None]):
    t: tuple[int, ...]

    def create(self):
        pass


class UsesPath(Furu[str]):
    path: Path

    def create(self) -> str:
        return str(self.path)


class UsesClassValue(Furu[None]):
    node_cls: type[Node]

    def create(self) -> None:
        return None


class UsesFalseLiteral(Furu[None]):
    tie_word_embeddings: Literal[False]

    def create(self) -> None:
        return None


class LoggedLeaf(Furu[str]):
    name: str

    def create(self) -> str:
        self.logger.info("leaf detail for %s", self.name)
        return f"leaf:{self.name}"


class LoggedParent(Furu[dict[str, str]]):
    child: LoggedLeaf

    def create(self) -> dict[str, str]:
        self.logger.info("parent before child")
        child_result = self.child.load_or_create()
        self.logger.info("parent after child")
        return {"child": child_result}


class PositiveValue(Furu[int]):
    value: int

    @validate
    def _validate_positive(self) -> None:
        if self.value <= 0:
            raise ValueError("value must be positive")

    def create(self) -> int:
        return self.value


class InheritedPositiveValue(PositiveValue):
    extra: str

    def create(self) -> int:
        return super().create()


class ParentAndChildValidated(PositiveValue):
    child_value: int

    @validate
    def _validate_child_value(self) -> None:
        if self.child_value <= 0:
            raise ValueError("child_value must be positive")

    def create(self) -> int:
        return super().create()


class PydanticSubclass(BaseModel):
    model_config = ConfigDict(frozen=True)
    field1: int


class PydanticFields(Furu[None]):
    pydantic_obj: PydanticSubclass

    def create(self) -> None:
        return None


GROUP_EXECUTION_EVENTS: list[tuple[str, tuple[int, ...]]] = []


class CountedSingleValue(Furu[str]):
    key: int
    create_calls: ClassVar[list[int]] = []

    def create(self) -> str:
        type(self).create_calls.append(self.key)
        GROUP_EXECUTION_EVENTS.append(("single", (self.key,)))
        return f"single:{self.key}"


class BatchOnlyValue(Furu[str]):
    key: int
    batch_calls: ClassVar[list[tuple[int, ...]]] = []

    @classmethod
    def _create_batched(cls, objs) -> list[str]:
        keys = tuple(obj.key for obj in objs)
        cls.batch_calls.append(keys)
        return [f"batch:{obj.key}" for obj in objs]


class GroupBatchA(Furu[str]):
    key: int
    batch_calls: ClassVar[list[tuple[int, ...]]] = []

    @classmethod
    def _create_batched(cls, objs) -> list[str]:
        keys = tuple(obj.key for obj in objs)
        cls.batch_calls.append(keys)
        GROUP_EXECUTION_EVENTS.append(("batch_a", keys))
        return [f"group-a:{obj.key}" for obj in objs]


class GroupBatchB(Furu[str]):
    key: int
    batch_calls: ClassVar[list[tuple[int, ...]]] = []

    @classmethod
    def _create_batched(cls, objs) -> list[str]:
        keys = tuple(obj.key for obj in objs)
        cls.batch_calls.append(keys)
        GROUP_EXECUTION_EVENTS.append(("batch_b", keys))
        return [f"group-b:{obj.key}" for obj in objs]


class LoggedBatchValue(Furu[str]):
    key: int

    @classmethod
    def _create_batched(cls, objs) -> list[str]:
        keys = ",".join(str(obj.key) for obj in objs)
        objs[0].logger.info("batched detail for %s", keys)
        return [f"logged-batch:{obj.key}" for obj in objs]


class LoggedSingleValue(Furu[str]):
    key: int

    def create(self) -> str:
        self.logger.info("single detail for %s", self.key)
        return f"logged-single:{self.key}"


class FailingBatchValue(Furu[str]):
    key: int

    @classmethod
    def _create_batched(cls, objs) -> list[str]:
        raise RuntimeError(f"failed batch for {[obj.key for obj in objs]}")


class FailingSingleValue(Furu[str]):
    key: int

    def create(self) -> str:
        raise RuntimeError(f"failed single for {self.key}")


class PartialBatchValue(Furu[str]):
    key: int

    @classmethod
    def _create_batched(cls, objs) -> list[str]:
        return [f"partial:{obj.key}" for obj in objs]


class MetadataTimingValue(Furu[str]):
    key: int
    create_events: ClassVar[list[tuple[int, bool, bool]]] = []
    siblings_by_key: ClassVar[dict[int, "MetadataTimingValue"]] = {}

    def create(self) -> str:
        sibling_key = 2 if self.key == 1 else 1
        sibling = type(self).siblings_by_key[sibling_key]
        type(self).create_events.append(
            (
                self.key,
                self._metadata_path.exists(),
                sibling._metadata_path.exists(),
            )
        )
        return f"timed:{self.key}"


@dataclass(frozen=True)
class DependencyBundle:
    first: Node
    second: WeightedNode


class NestedDependencyParent(Furu[str]):
    bundle: DependencyBundle

    def create(self) -> str:
        return self.bundle.first.load_or_create()


class ComputedDependencyParent(Furu[str]):
    name: str

    @furu.dependency
    def child(self) -> Node:
        return Node(name=self.name)

    def create(self) -> str:
        return self.child.load_or_create()


class LazyDependencyParent(Furu[str]):
    name: str

    def create(self) -> str:
        return Node(name=self.name).load_or_create()


class TryLoadDependencyParent(Furu[str]):
    name: str

    def create(self) -> str:
        try:
            Node(name=self.name).try_load()
        except NotImplementedError:
            return "missing"
        return "loaded"


class FuruBoundaryParent(Furu[str]):
    child: NodePair

    def create(self) -> str:
        return self.child.node1.load_or_create()


class BatchDependencyParent(Furu[str]):
    key: int
    eager: Node

    @classmethod
    def _create_batched(cls, objs) -> list[str]:
        eager_values = [obj.eager.load_or_create() for obj in objs]
        lazy_value = Node(name="shared-lazy").load_or_create()
        return [f"{value}:{lazy_value}" for value in eager_values]


@pytest.fixture(autouse=True)
def _reset_batch_trackers() -> None:
    CountedSingleValue.create_calls.clear()
    BatchOnlyValue.batch_calls.clear()
    GroupBatchA.batch_calls.clear()
    GroupBatchB.batch_calls.clear()
    GROUP_EXECUTION_EVENTS.clear()
    MetadataTimingValue.create_events.clear()
    MetadataTimingValue.siblings_by_key.clear()


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


def test_class_level_validation():
    assert PositiveValue(value=2).load_or_create() == 2

    with pytest.raises(ValueError, match="value must be positive"):
        PositiveValue(value=0)


def test_validators_are_inherited():
    InheritedPositiveValue(value=1, extra="ok")

    with pytest.raises(ValueError, match="value must be positive"):
        InheritedPositiveValue(value=-1, extra="oops")


def test_parent_and_child_validators_both_run():
    ParentAndChildValidated(value=1, child_value=1)

    with pytest.raises(ValueError, match="value must be positive"):
        ParentAndChildValidated(value=0, child_value=1)

    with pytest.raises(ValueError, match="child_value must be positive"):
        ParentAndChildValidated(value=1, child_value=0)


def test_unannotated_public_attribute_raises_clear_error():
    with pytest.raises(
        TypeError, match="UnannotatedParameter.a must have a type annotation"
    ):

        class UnannotatedParameter(Furu[int]):
            a = 1

            def create(self) -> int:
                return self.a


def test_unannotated_private_attribute_raises_clear_error():
    with pytest.raises(
        TypeError, match="UnannotatedPrivate._a must have a type annotation"
    ):

        class UnannotatedPrivate(Furu[int]):
            _a = 1

            def create(self) -> int:
                return self._a


def test_validate_decorator_supports_call_syntax():
    class CallSyntaxValidated(Furu[int]):
        value: int

        @validate()
        def _validate_positive(self) -> None:
            if self.value <= 0:
                raise ValueError("value must be positive")

        def create(self) -> int:
            return self.value

    assert CallSyntaxValidated(value=2).value == 2

    with pytest.raises(ValueError, match="value must be positive"):
        CallSyntaxValidated(value=0)


def test_post_init_can_transform_values_before_validation():
    class PostInitValidated(Furu[int]):
        raw_value: int | str
        value: int = 0

        def __post_init__(self) -> None:
            object.__setattr__(self, "value", int(self.raw_value))

        @validate
        def _validate_positive(self) -> None:
            if self.value <= 0:
                raise ValueError("value must be positive")

        def create(self) -> int:
            assert isinstance(self.value, int)
            return self.value

    assert PostInitValidated(raw_value="2").value == 2

    with pytest.raises(ValueError, match="value must be positive"):
        PostInitValidated(raw_value="0")


def test_post_init_and_inherited_validators_both_run():
    class PostInitInheritedPositiveValue(PositiveValue):
        raw_value: int | str

        def __post_init__(self) -> None:
            object.__setattr__(self, "value", int(self.raw_value))

        def create(self) -> int:
            return super().create()

    assert PostInitInheritedPositiveValue(value=1, raw_value="2").value == 2

    with pytest.raises(ValueError, match="value must be positive"):
        PostInitInheritedPositiveValue(value=1, raw_value="0")


def test_inherited_post_init_and_inherited_validators_both_run():
    class BasePostInitValue(Furu[int]):
        raw_value: int | str
        value: int = 0

        def __post_init__(self) -> None:
            object.__setattr__(self, "value", int(self.raw_value))

        @validate
        def _validate_positive(self) -> None:
            if self.value <= 0:
                raise ValueError("value must be positive")

        def create(self) -> int:
            return self.value

    class ChildPostInitValue(BasePostInitValue):
        label: str

        def create(self) -> int:
            return super().create()

    assert ChildPostInitValue(raw_value="2", label="ok").value == 2

    with pytest.raises(ValueError, match="value must be positive"):
        ChildPostInitValue(raw_value="0", label="nope")


def test_post_init_chain_runs_before_validators_without_duplicate_calls():
    calls: list[str] = []

    class BaseOrderedValue(Furu[int]):
        value: int

        def __post_init__(self) -> None:
            calls.append("base_post_init")

        @validate
        def _validate_base(self) -> None:
            calls.append("base_validate")

        def create(self) -> int:
            return self.value

    class ChildOrderedValue(BaseOrderedValue):
        label: str

        def __post_init__(self) -> None:
            calls.append("child_post_init")

        @validate
        def _validate_child(self) -> None:
            calls.append("child_validate")

        def create(self) -> int:
            return super().create()

    ChildOrderedValue(value=1, label="ok")

    assert calls == [
        "base_post_init",
        "child_post_init",
        "base_validate",
        "child_validate",
    ]


def test_hashes_and_data_dir():
    assert (
        NodePair(
            name="x", node1=Node(name="y"), node2=WeightedNode(name="z", weight=1)
        ).object_id
        == "test_core.NodePair:21733b1febfab88b565c:685af925669262434640"
    )

    assert (
        NodePair(
            name="x", node1=Node(name="y"), node2=WeightedNode(name="z", weight=1)
        ).artifact_hash
        == "685af925669262434640"
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
        ).artifact_schema_hash
        == "21733b1febfab88b565c"
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
        ).artifact_schema_hash
        != B_priv(
            a=A(x=1, z="123", w=[6, 7]),
            y={"hey": 123, True: 1},
            t=("123", 12),
            _h=1,
        ).artifact_schema_hash
    )

    def qualname_alias(cls: type[Furu[object]], *, ret_typ: type) -> type[Furu[object]]:
        namespace: dict[str, object] = {"__module__": cls.__module__}
        if cls._furu_create_mode == "single":
            namespace["create"] = lambda self: cls.create(self)
        else:
            namespace["_create_batched"] = classmethod(
                lambda alias_cls, objs: cls._create_batched(objs)
            )
        alias = type(ret_typ.__qualname__, (cls,), namespace)
        alias.__qualname__ = ret_typ.__qualname__
        return alias

    B_priv_as_B = cast(type[B_priv], qualname_alias(B_priv, ret_typ=B))
    assert (
        B(
            a=A(x=1, z="123", w=[6, 7]), y={"ney": 123, True: 1}, t=("123", 12)
        ).artifact_schema_hash
        != B_priv_as_B(
            a=A(x=1, z="123", w=[6, 7]),
            y={"hey": 123, "ney": 1},
            t=("123", 12),
            _h=1,
        ).artifact_schema_hash
    )


def expected_schema_for_B_like(
    cls_name: str, *, include_private_h: bool = False
) -> dict:
    fields: dict[str, Any] = {
        "a": [
            "builtins.int",
            {
                "|class": "test_core.A",
                "|fields": {
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
    }
    if include_private_h:
        fields["_h"] = "builtins.int"

    return {
        "|class": f"test_core.{cls_name}",
        "|fields": fields,
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
                a=A(x=1, z="123", w=[6, 7]), y={"hey": 123, True: 1}, t=("123", 12)
            ),
            expected_schema_for_B_like("B_priv", include_private_h=True),
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
                "|fields": {
                    "name": ["builtins.int", "builtins.str"],
                    "node1": {
                        "|class": "test_core.Node",
                        "|fields": {"name": "builtins.str"},
                    },
                    "node2": {
                        "|class": "test_core.WeightedNode",
                        "|fields": {"name": "builtins.str", "weight": "builtins.float"},
                    },
                },
            },
            id="NodePair",
        ),
        pytest.param(
            lambda: UsesPath(path=Path("/tmp/out")),
            {
                "|class": "test_core.UsesPath",
                "|fields": {"path": fully_qualified_name(Path)},
            },
            id="UsesPath",
        ),
        pytest.param(
            lambda: PydanticFields(pydantic_obj=PydanticSubclass(field1=1)),
            {
                "|class": "test_core.PydanticFields",
                "|fields": {
                    "pydantic_obj": {
                        "|class": "test_core.PydanticSubclass",
                        "|fields": {"field1": "builtins.int"},
                    }
                },
            },
            id="PydanticFields",
        ),
        pytest.param(
            lambda: UsesFalseLiteral(tie_word_embeddings=False),
            {
                "|class": "test_core.UsesFalseLiteral",
                "|fields": {
                    "tie_word_embeddings": {
                        "|origin": "typing.Literal",
                        "|args": [False],
                    }
                },
            },
            id="UsesFalseLiteral",
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
        "|fields": {
            "node1": {
                "|kind": "instance",
                "|class": "test_core.Node",
                "|fields": {"name": "y"},
            },
            "node2": {
                "|kind": "instance",
                "|class": "test_core.WeightedNode",
                "|fields": {"name": "z", "weight": 1},
            },
            "name": "x",
        },
    }
    assert to_json(node_pair) == expected
    assert to_json(node_pair) == node_pair.artifact_data
    assert node_pair.artifact_data == expected


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
        "|fields": {
            "a": {
                "|kind": "instance",
                "|class": "test_core.A",
                "|fields": {"x": 1, "z": "123", "w": [6, 7], "some_obj": "a"},
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
    assert obj.artifact_data == {
        "|kind": "instance",
        "|class": "test_core.UsesClassValue",
        "|fields": {"node_cls": {"|kind": "type_ref", "|class": "test_core.Node"}},
    }
    assert isinstance(obj.artifact_hash, str)


def test_to_json_with_pydantic_field_value():
    obj = PydanticFields(pydantic_obj=PydanticSubclass(field1=1))

    expected = {
        "|kind": "instance",
        "|class": "test_core.PydanticFields",
        "|fields": {
            "pydantic_obj": {
                "|kind": "instance",
                "|class": "test_core.PydanticSubclass",
                "|fields": {"field1": 1},
            }
        },
    }

    assert to_json(obj) == expected
    assert obj.artifact_data == expected
    assert isinstance(obj.artifact_hash, str)


def test_furu_object_round_trips_from_json_artifact():
    obj = NodePair(
        name="x",
        node1=Node(name="y"),
        node2=WeightedNode(name="z", weight=1),
    )

    loaded = _from_json(obj.artifact_data)

    assert loaded == obj
    assert isinstance(loaded, NodePair)
    assert loaded.object_id == obj.object_id


def test_furu_object_with_typed_fields_round_trips_from_json_artifact():
    path_obj = UsesPath(path=Path("/tmp/out"))
    class_obj = UsesClassValue(node_cls=Node)

    assert _from_json(path_obj.artifact_data) == path_obj
    assert _from_json(class_obj.artifact_data) == class_obj
    assert isinstance(cast(UsesPath, _from_json(path_obj.artifact_data)).path, Path)


def test_furu_from_artifact_returns_furu_object():
    obj = NodePair(
        name="x",
        node1=Node(name="y"),
        node2=WeightedNode(name="z", weight=1),
    )
    obj.load_or_create()
    artifact = ArtifactMetadata(
        data=obj.artifact_data,
        hash=obj.artifact_hash,
        schema=obj.schema,
        schema_hash=obj.artifact_schema_hash,
    )

    loaded = NodePair.from_artifact(artifact)
    raw_metadata = json.loads(obj._metadata_path.read_text())

    assert loaded == obj
    assert isinstance(loaded, NodePair)
    assert loaded.data_dir == obj.data_dir
    assert raw_metadata["kind"] == "completed"
    assert raw_metadata["artifact"] == {
        "data": obj.artifact_data,
        "hash": obj.artifact_hash,
        "schema": obj.schema,
        "schema_hash": obj.artifact_schema_hash,
    }
    assert "artifact_hash" not in raw_metadata
    assert "artifact_schema" not in raw_metadata
    assert "artifact_schema_hash" not in raw_metadata


def _dependency_object_ids(obj: Furu[Any]) -> list[str]:
    metadata = json.loads(obj._metadata_path.read_text())
    return metadata["observed_dependencies"]


def test_field_dependencies_are_eager_but_metadata_stores_only_loaded_objects() -> None:
    first = Node(name="nested")
    second = WeightedNode(name="weighted", weight=2)
    parent = NestedDependencyParent(bundle=DependencyBundle(first=first, second=second))

    assert parent._declared_refs() == (first, second)
    assert parent.load_or_create() == "Node(nested)"
    assert _dependency_object_ids(parent) == [first.object_id]


def test_computed_dependency_is_cached_property_and_eager_loaded_dependency() -> None:
    parent = ComputedDependencyParent(name="computed")

    assert parent.child is parent.child
    assert parent._declared_refs() == (parent.child,)
    assert parent.load_or_create() == "Node(computed)"

    assert _dependency_object_ids(parent) == [parent.child.object_id]


def test_load_or_create_inside_create_is_recorded_and_deduped() -> None:
    parent = LazyDependencyParent(name="lazy")

    assert parent.load_or_create() == "Node(lazy)"

    assert _dependency_object_ids(parent) == [Node(name="lazy").object_id]


def test_try_load_inside_create_is_recorded_even_on_missing_result() -> None:
    parent = TryLoadDependencyParent(name="optional")

    assert parent.load_or_create() == "missing"

    metadata = json.loads(parent._metadata_path.read_text())
    assert metadata["observed_dependencies"] == [Node(name="optional").object_id]


def test_furu_objects_block_nested_eager_traversal_but_direct_runtime_loads_are_recorded() -> (
    None
):
    node1 = Node(name="inner")
    node2 = WeightedNode(name="other", weight=3)
    child = NodePair(node1=node1, node2=node2, name="pair")
    parent = FuruBoundaryParent(child=child)

    assert parent.load_or_create() == "Node(inner)"

    assert parent._declared_refs() == (child,)
    assert _dependency_object_ids(parent) == [node1.object_id]


def test_batched_dependencies_record_all_observed_loads() -> None:
    objs = [
        BatchDependencyParent(key=1, eager=Node(name="eager-1")),
        BatchDependencyParent(key=2, eager=Node(name="eager-2")),
    ]

    assert load_or_create(objs) == [
        "Node(eager-1):Node(shared-lazy)",
        "Node(eager-2):Node(shared-lazy)",
    ]

    for obj in objs:
        assert obj._declared_refs() == (obj.eager,)
        assert _dependency_object_ids(obj) == sorted(
            [
                objs[0].eager.object_id,
                objs[1].eager.object_id,
                Node(name="shared-lazy").object_id,
            ]
        )


def test_furu_from_artifact_infers_furu_object_type():
    obj = NodePair(
        name="x",
        node1=Node(name="y"),
        node2=WeightedNode(name="z", weight=1),
    )
    obj.load_or_create()
    artifact = ArtifactMetadata(
        data=obj.artifact_data,
        hash=obj.artifact_hash,
        schema=obj.schema,
        schema_hash=obj.artifact_schema_hash,
    )

    loaded = Furu.from_artifact(artifact)

    assert loaded == obj
    assert isinstance(loaded, NodePair)


def test_furu_from_artifact_accepts_loaded_metadata_artifact():
    obj = Node(name="x")
    obj.load_or_create()
    metadata = json.loads(obj._metadata_path.read_text())
    artifact = ArtifactMetadata(**metadata["artifact"])

    loaded = Node.from_artifact(artifact)

    assert loaded == obj
    assert isinstance(loaded, Node)


def test_furu_from_artifact_accepts_artifact_metadata():
    obj = Node(name="x")
    artifact = ArtifactMetadata(
        data=obj.artifact_data,
        hash=obj.artifact_hash,
        schema=obj.schema,
        schema_hash=obj.artifact_schema_hash,
    )

    loaded = Node.from_artifact(artifact)

    assert loaded == obj
    assert isinstance(loaded, Node)


def test_furu_from_artifact_type_mismatch_names_expected_and_loaded_type():
    obj = WeightedNode(name="x", weight=1)
    artifact = ArtifactMetadata(
        data=obj.artifact_data,
        hash=obj.artifact_hash,
        schema=obj.schema,
        schema_hash=obj.artifact_schema_hash,
    )

    with pytest.raises(
        TypeError,
        match=(
            r"Artifact described test_core\.WeightedNode, "
            r"expected test_core\.NodePair"
        ),
    ):
        NodePair.from_artifact(artifact)


def test_furu_from_artifact_rejects_artifact_metadata_hash_mismatch():
    obj = Node(name="x")
    bad_hash = "wrong-artifact-hash"
    artifact = ArtifactMetadata(
        data=obj.artifact_data,
        hash=bad_hash,
        schema=obj.schema,
        schema_hash=obj.artifact_schema_hash,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Artifact hash did not match loaded object: "
            + f"artifact={bad_hash[:5]}, loaded={obj.artifact_hash[:5]}"
        ),
    ):
        Node.from_artifact(artifact)


def test_furu_from_artifact_rejects_artifact_metadata_schema_hash_mismatch():
    obj = Node(name="x")
    bad_schema_hash = "wrong-schema-hash"
    artifact = ArtifactMetadata(
        data=obj.artifact_data,
        hash=obj.artifact_hash,
        schema=obj.schema,
        schema_hash=bad_schema_hash,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Artifact schema hash did not match loaded object: "
            + f"artifact={bad_schema_hash[:5]}, "
            + f"loaded={obj.artifact_schema_hash[:5]}"
        ),
    ):
        Node.from_artifact(artifact)


def test_schema_with_ellipsis_type_arg():
    assert VariadicTuple(t=(1, 2, 3)).schema == {
        "|class": "test_core.VariadicTuple",
        "|fields": {
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
        / "21733b1febfab88b565c"
        / "685af925669262434640"
    )
    assert node_pair.data_dir == Path(
        config.directories.data
        / "test_core"
        / "NodePair"
        / node_pair.artifact_schema_hash
        / node_pair.artifact_hash
    )


def test_storage_root_can_be_overridden_with_cached_property():
    node = CustomStorageRootNode(name="x")

    assert node.storage_root == Path("custom/data/location")
    assert node.storage_root is node.storage_root
    assert node.data_dir == (
        Path("custom/data/location")
        / "test_core"
        / "CustomStorageRootNode"
        / node.artifact_schema_hash
        / node.artifact_hash
    )


def test_data_dir_override_is_rejected():
    with pytest.raises(TypeError, match="must not override data_dir"):

        class CustomDataDirNode(Node):
            @cached_property
            def data_dir(self) -> Path:
                return Path("custom/data/location")


def test_create_object_and_exists():
    node_pair = NodePair(
        name="x", node1=Node(name="y"), node2=WeightedNode(name="z", weight=1)
    )
    assert node_pair.status() == "missing"
    for _i in range(3):
        assert node_pair.load_or_create() == {
            "node1": "Node(y)",
            "node2": "WNode(z:1)",
            "name": "x",
        }
    assert node_pair.status() == "completed"
    assert replace(node_pair, name="y").status() == "missing"


def test_status_is_running_while_compute_lock_is_held() -> None:
    node = Node(name="x")
    node._internal_furu_dir.mkdir(parents=True, exist_ok=True)

    with lock_many([node._lock_path]):
        assert node.status() == "running"


def test_status_is_failed_after_create_error() -> None:
    obj = FailingSingleValue(key=1)

    with pytest.raises(RuntimeError, match="failed single for 1"):
        obj.load_or_create()

    assert obj.status() == "failed"


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


def test_log_file_is_written_to_internal_dir() -> None:
    node = LoggedLeaf(name="x")

    assert node.load_or_create() == "leaf:x"

    assert node._log_path == node._internal_furu_dir / "run.log"
    log_text = node._log_path.read_text(encoding="utf-8")
    assert "leaf detail for x" in log_text


def test_nested_load_or_create_scopes_logs_to_child_file() -> None:
    child = LoggedLeaf(name="child")
    parent = LoggedParent(child=child)

    assert parent.load_or_create() == {"child": "leaf:child"}

    parent_log = parent._log_path.read_text(encoding="utf-8")
    child_log = child._log_path.read_text(encoding="utf-8")

    assert "parent before child" in parent_log
    assert f"calling {child._log_label}.load_or_create()" in parent_log
    assert ".load_or_create() returned" in parent_log
    assert "parent after child" in parent_log
    assert "leaf detail for child" not in parent_log

    assert "leaf detail for child" in child_log


def test_method_load_or_create_delegates_to_shared_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = Node(name="delegated")
    calls: list[tuple[Furu[str], bool]] = []

    def fake_load_or_create(obj: Furu[str], *, use_lock: bool = True) -> str:
        calls.append((obj, use_lock))
        return "delegated"

    monkeypatch.setattr(execution_module, "load_or_create", fake_load_or_create)

    assert node.load_or_create(use_lock=False) == "delegated"
    assert calls == [(node, False)]


def test_create_cannot_be_called_directly() -> None:
    node = Node(name="direct")

    with pytest.raises(RuntimeError, match=r"create\(\) cannot be called directly"):
        node.create()


def test_resolved_create_mode_validation() -> None:
    class ExplicitSingle(Node):
        label: str

        def create(self) -> str:
            return f"single:{self.label}"

    class ExplicitBatch(Furu[str]):
        label: str

        @classmethod
        def _create_batched(cls, objs) -> list[str]:
            return [f"batch:{obj.label}" for obj in objs]

    assert Node._furu_create_mode == "single"
    assert BatchOnlyValue._furu_create_mode == "batched"
    assert ExplicitSingle._furu_create_mode == "single"
    assert ExplicitBatch._furu_create_mode == "batched"

    class InheritedSingle(Node):
        label: str

    class InheritedBatch(BatchOnlyValue):
        label: str

    assert InheritedSingle._furu_create_mode == "single"
    assert InheritedBatch._furu_create_mode == "batched"

    with pytest.raises(
        TypeError, match="must define exactly one of create or _create_batched"
    ):

        class InvalidBoth(Furu[int]):
            def create(self) -> int:
                return 1

            @classmethod
            def _create_batched(cls, objs) -> list[int]:
                return [1 for _ in objs]

    with pytest.raises(TypeError, match=r"_create_batched must be a @classmethod"):

        class InvalidBatchMethod(Furu[int]):
            def _create_batched(self, objs) -> list[int]:
                return [1 for _ in objs]

    with pytest.raises(
        TypeError, match="must define exactly one of create or _create_batched"
    ):

        class InvalidInherited(Node):
            label: str

            @classmethod
            def _create_batched(cls, objs) -> list[str]:
                return [obj.label for obj in objs]

    with pytest.raises(
        TypeError,
        match="must define exactly one create hook in its inheritance chain",
    ):

        class InvalidNone(Furu[int]):
            pass


def test_single_object_on_batch_only_class_uses_create_batched() -> None:
    assert load_or_create(BatchOnlyValue(key=1)) == "batch:1"
    assert BatchOnlyValue.batch_calls == [(1,)]


def test_list_input_on_single_only_class_uses_sequentialcreate() -> None:
    objs = [
        CountedSingleValue(key=1),
        CountedSingleValue(key=2),
        CountedSingleValue(key=3),
    ]

    assert load_or_create(objs) == ["single:1", "single:2", "single:3"]
    assert CountedSingleValue.create_calls == [1, 2, 3]


def test_sequential_fallback_writes_running_metadata_per_object() -> None:
    first = MetadataTimingValue(key=1)
    second = MetadataTimingValue(key=2)
    MetadataTimingValue.siblings_by_key.update({1: first, 2: second})

    assert load_or_create([first, second], use_lock=False) == ["timed:1", "timed:2"]
    assert MetadataTimingValue.create_events == [
        (1, True, True),
        (2, True, True),
    ]


def test_list_input_on_batch_only_class_calls_create_batched_once_per_concrete_group() -> (
    None
):
    objs = [
        GroupBatchA(key=1),
        GroupBatchB(key=1),
        GroupBatchA(key=2),
        GroupBatchB(key=2),
    ]

    assert load_or_create(objs) == ["group-a:1", "group-b:1", "group-a:2", "group-b:2"]
    assert GroupBatchA.batch_calls == [(1, 2)]
    assert GroupBatchB.batch_calls == [(1, 2)]


def test_duplicate_cache_identities_compute_once_and_preserve_input_order() -> None:
    objs = [
        CountedSingleValue(key=1),
        CountedSingleValue(key=1),
        CountedSingleValue(key=2),
        CountedSingleValue(key=1),
    ]

    assert load_or_create(objs) == ["single:1", "single:1", "single:2", "single:1"]
    assert CountedSingleValue.create_calls == [1, 2]


def test_existing_items_are_skipped_before_locking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = CountedSingleValue(key=1)
    missing = CountedSingleValue(key=2)

    assert existing.load_or_create() == "single:1"

    lock_calls: list[list[Path]] = []

    @contextmanager
    def fake_lock_many(lock_paths: list[Path], **_: object):
        lock_calls.append(lock_paths)
        yield lambda: True

    monkeypatch.setattr(execution_module, "lock_many", fake_lock_many)

    assert load_or_create([existing, missing]) == ["single:1", "single:2"]
    assert lock_calls == [[missing._lock_path]]


def test_pending_items_are_rechecked_after_lock_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pending = CountedSingleValue(key=5)

    @contextmanager
    def fake_lock_many(lock_paths: list[Path], **_: object):
        assert lock_paths == [pending._lock_path]
        save_result_bundle(
            "single:5", pending._result_dir, registry=_default_result_registry()
        )
        yield lambda: True

    monkeypatch.setattr(execution_module, "lock_many", fake_lock_many)

    assert load_or_create([pending]) == ["single:5"]
    assert CountedSingleValue.create_calls == []


def test_empty_list_returns_empty_list() -> None:
    assert load_or_create([]) == []


def test_mixed_type_list_follows_documented_grouping_policy() -> None:
    objs = [
        GroupBatchA(key=1),
        CountedSingleValue(key=10),
        GroupBatchA(key=2),
        GroupBatchB(key=20),
        CountedSingleValue(key=11),
        GroupBatchB(key=21),
    ]

    assert load_or_create(objs) == [
        "group-a:1",
        "single:10",
        "group-a:2",
        "group-b:20",
        "single:11",
        "group-b:21",
    ]
    assert GROUP_EXECUTION_EVENTS == [
        ("batch_a", (1, 2)),
        ("single", (10,)),
        ("single", (11,)),
        ("batch_b", (20, 21)),
    ]


def test_batched_compute_writes_result_layout_per_object() -> None:
    objs = [BatchOnlyValue(key=1), BatchOnlyValue(key=2)]

    assert load_or_create(objs) == ["batch:1", "batch:2"]

    for obj, expected in zip(objs, ["batch:1", "batch:2"], strict=True):
        assert obj._result_manifest_path.exists()
        assert obj._metadata_path.exists()
        assert obj._log_path.exists()
        assert load_result_bundle(obj._result_dir) == expected


def test_batched_compute_writes_shared_logs_to_every_participant() -> None:
    objs = [LoggedBatchValue(key=1), LoggedBatchValue(key=2)]

    assert load_or_create(objs) == ["logged-batch:1", "logged-batch:2"]

    for obj in objs:
        log_text = obj._log_path.read_text(encoding="utf-8")
        assert "batched detail for 1,2" in log_text
        for persisted_obj in objs:
            assert f"stored result bundle at {persisted_obj._result_dir}" in log_text


def test_sequential_group_compute_writes_shared_logs_to_every_participant() -> None:
    objs = [LoggedSingleValue(key=1), LoggedSingleValue(key=2)]

    assert load_or_create(objs) == ["logged-single:1", "logged-single:2"]

    for obj in objs:
        log_text = obj._log_path.read_text(encoding="utf-8")
        assert "single detail for 1" in log_text
        assert "single detail for 2" in log_text
        for persisted_obj in objs:
            assert f"stored result bundle at {persisted_obj._result_dir}" in log_text


def test_batched_failure_writes_error_details_to_run_log_for_every_participant() -> (
    None
):
    objs = [FailingBatchValue(key=1), FailingBatchValue(key=2)]

    with pytest.raises(RuntimeError, match="failed batch"):
        load_or_create(objs)

    for obj in objs:
        log_text = obj._log_path.read_text(encoding="utf-8")
        assert "load_or_create failed" in log_text
        assert "failed batch for [1, 2]" in log_text
        assert "=== Debug Details (with locals) ===" in log_text
        assert list(obj._internal_furu_dir.glob("error-*.log")) == []


def test_partial_persistence_leaves_already_written_objects_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objs = [PartialBatchValue(key=1), PartialBatchValue(key=2)]
    real_store_result = execution_module._store_result
    call_count = 0

    def flaky_store_result(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("stop after first store")
        return real_store_result(*args, **kwargs)

    monkeypatch.setattr(execution_module, "_store_result", flaky_store_result)

    with pytest.raises(RuntimeError, match="stop after first store"):
        load_or_create(objs)

    assert objs[0]._result_manifest_path.exists()
    assert not objs[1]._result_manifest_path.exists()
