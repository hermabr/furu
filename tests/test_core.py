import types
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, is_dataclass, replace
from enum import Enum
from functools import partial
from pathlib import Path
import threading
import pickle
from typing import ClassVar, Literal, TypeVar, cast
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ConfigDict

from furu import Furu, load_or_create, validate
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


class UsesFalseLiteral(Furu[None]):
    tie_word_embeddings: Literal[False]

    def _create(self) -> None:
        return None

class LoggedLeaf(Furu[str]):
    name: str

    def _create(self) -> str:
        self.logger.info("leaf detail for %s", self.name)
        return f"leaf:{self.name}"


class LoggedParent(Furu[dict[str, str]]):
    child: LoggedLeaf

    def _create(self) -> dict[str, str]:
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

    def _create(self) -> int:
        return self.value


class InheritedPositiveValue(PositiveValue):
    extra: str


class ParentAndChildValidated(PositiveValue):
    child_value: int

    @validate
    def _validate_child_value(self) -> None:
        if self.child_value <= 0:
            raise ValueError("child_value must be positive")


class PydanticSubclass(BaseModel):
    model_config = ConfigDict(frozen=True)
    field1: int


class PydanticFields(Furu[None]):
    pydantic_obj: PydanticSubclass

    def _create(self) -> None:
        return None


class CountingValue(Furu[int]):
    value: int

    create_calls: ClassVar[list[int]] = []

    def _create(self) -> int:
        type(self).create_calls.append(self.value)
        return self.value


class BatchOnlyValue(Furu[int]):
    value: int

    batch_calls: ClassVar[list[list[int]]] = []

    @classmethod
    def _create_batched(cls, objs: Sequence[Furu[int]]) -> list[int]:
        typed_objs = [cast(BatchOnlyValue, obj) for obj in objs]
        cls.batch_calls.append([obj.value for obj in typed_objs])
        return [obj.value * 10 for obj in typed_objs]


class InheritedBatchOnlyValue(BatchOnlyValue):
    label: str


class OtherBatchValue(Furu[str]):
    value: str

    batch_calls: ClassVar[list[list[str]]] = []

    @classmethod
    def _create_batched(cls, objs: Sequence[Furu[str]]) -> list[str]:
        typed_objs = [cast(OtherBatchValue, obj) for obj in objs]
        cls.batch_calls.append([obj.value for obj in typed_objs])
        return [obj.value.upper() for obj in typed_objs]


class ReentrantSelfLoad(Furu[int]):
    value: int

    def _create(self) -> int:
        return self.load_or_create(use_lock=False)


class BatchedLoggedValue(Furu[str]):
    name: str

    @classmethod
    def _create_batched(cls, objs: Sequence[Furu[str]]) -> list[str]:
        typed_objs = [cast(BatchedLoggedValue, obj) for obj in objs]
        typed_objs[0].logger.info(
            "batch detail for %s",
            ",".join(obj.name for obj in typed_objs),
        )
        return [f"batched:{obj.name}" for obj in typed_objs]


class PartialPersistBatch(Furu[object]):
    key: str

    @classmethod
    def _create_batched(cls, objs: Sequence[Furu[object]]) -> list[object]:
        typed_objs = [cast(PartialPersistBatch, obj) for obj in objs]
        results: list[object] = []
        for obj in typed_objs:
            if obj.key == "bad":
                results.append(threading.Lock())
            else:
                results.append(obj.key)
        return results


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

        def _create(self) -> int:
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

    assert PostInitInheritedPositiveValue(value=1, raw_value="2").value == 2

    with pytest.raises(ValueError, match="value must be positive"):
        PostInitInheritedPositiveValue(value=1, raw_value="0")


def test_class_cannot_define_both_create_hooks():
    with pytest.raises(TypeError, match="at most one"):

        class InvalidBothHooks(Furu[int]):
            value: int

            def _create(self) -> int:
                return self.value

            @classmethod
            def _create_batched(cls, objs: Sequence[Furu[int]]) -> list[int]:
                return [1 for _ in objs]


def test_class_must_define_or_inherit_a_create_hook():
    with pytest.raises(TypeError, match="must define or inherit exactly one"):

        class MissingCreateHooks(Furu[int]):
            value: int


def test_instance_load_or_create_delegates_to_top_level_function():
    node = Node(name="x")

    with patch("furu.core.load_or_create", return_value="delegated") as mocked:
        assert node.load_or_create() == "delegated"
        mocked.assert_called_once_with(node, use_lock=True)


def test_top_level_load_or_create_returns_empty_list_for_empty_input():
    assert load_or_create([], use_lock=False) == []


def test_batch_only_class_works_for_single_object_and_singleton_list():
    BatchOnlyValue.batch_calls.clear()

    assert BatchOnlyValue(value=2).load_or_create(use_lock=False) == 20
    assert load_or_create([BatchOnlyValue(value=3)], use_lock=False) == [30]
    assert BatchOnlyValue.batch_calls == [[2], [3]]


def test_batch_mode_is_inherited():
    BatchOnlyValue.batch_calls.clear()

    assert InheritedBatchOnlyValue(value=4, label="x").load_or_create(use_lock=False) == 40
    assert BatchOnlyValue.batch_calls == [[4]]


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

        def _create(self) -> int:
            return self.value

    class ChildPostInitValue(BasePostInitValue):
        label: str

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

        def _create(self) -> int:
            return self.value

    class ChildOrderedValue(BaseOrderedValue):
        label: str

        def __post_init__(self) -> None:
            calls.append("child_post_init")

        @validate
        def _validate_child(self) -> None:
            calls.append("child_validate")

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
        != B_priv_as_B(
            a=A(x=1, z="123", w=[6, 7]),
            y={"hey": 123, "ney": 1},
            t=("123", 12),
            _h=1,
        ).schema_hash
    )


def expected_schema_for_B_like(cls_name: str, *, include_private_h: bool = False) -> dict:
    fields = {
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
    }
    if include_private_h:
        fields["_h"] = "builtins.int"

    return {
        "|class": f"test_core.{cls_name}",
        "fields": fields,
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
        pytest.param(
            lambda: PydanticFields(pydantic_obj=PydanticSubclass(field1=1)),
            {
                "|class": "test_core.PydanticFields",
                "fields": {
                    "pydantic_obj": {
                        "|class": "test_core.PydanticSubclass",
                        "fields": {"field1": "builtins.int"},
                    }
                },
            },
            id="PydanticFields",
        ),
        pytest.param(
            lambda: UsesFalseLiteral(tie_word_embeddings=False),
            {
                "|class": "test_core.UsesFalseLiteral",
                "fields": {
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
    assert to_json(node_pair) == node_pair.artifact
    assert node_pair.artifact == expected


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
    assert obj.artifact == {
        "|kind": "instance",
        "|class": "test_core.UsesClassValue",
        "fields": {"node_cls": {"|kind": "type_ref", "|class": "test_core.Node"}},
    }
    assert isinstance(obj.artifact_hash, str)


def test_to_json_with_pydantic_field_value():
    obj = PydanticFields(pydantic_obj=PydanticSubclass(field1=1))

    expected = {
        "|kind": "instance",
        "|class": "test_core.PydanticFields",
        "fields": {
            "pydantic_obj": {
                "|kind": "instance",
                "|class": "test_core.PydanticSubclass",
                "fields": {"field1": 1},
            }
        },
    }

    assert to_json(obj) == expected
    assert obj.artifact == expected
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


def test_list_load_or_create_deduplicates_by_cache_identity_and_preserves_order() -> None:
    CountingValue.create_calls.clear()

    result = load_or_create(
        [CountingValue(value=1), CountingValue(value=2), CountingValue(value=1)],
        use_lock=False,
    )

    assert result == [1, 2, 1]
    assert CountingValue.create_calls == [1, 2]


def test_completed_items_are_skipped_before_locking() -> None:
    BatchOnlyValue.batch_calls.clear()
    completed = BatchOnlyValue(value=1)
    pending = BatchOnlyValue(value=2)

    assert completed.load_or_create(use_lock=False) == 10

    observed_lock_paths: list[tuple[Path, ...]] = []

    @contextmanager
    def fake_lock_many(lock_paths, **_kwargs):
        observed_lock_paths.append(tuple(lock_paths))
        yield lambda: True

    with patch("furu.core.lock_many", new=fake_lock_many):
        assert load_or_create([completed, pending], use_lock=True) == [10, 20]

    assert observed_lock_paths == [((pending._internal_furu_dir / "compute.lock"),)]
    assert BatchOnlyValue.batch_calls == [[1], [2]]


def test_pending_items_are_rechecked_after_lock_acquisition() -> None:
    BatchOnlyValue.batch_calls.clear()
    pending = BatchOnlyValue(value=5)

    @contextmanager
    def fake_lock_many(lock_paths, **_kwargs):
        assert list(lock_paths) == [pending._internal_furu_dir / "compute.lock"]
        with pending._result_path.open("wb") as f:
            pickle.dump(50, f)
        yield lambda: True

    with patch("furu.core.lock_many", new=fake_lock_many):
        assert load_or_create([pending], use_lock=True) == [50]

    assert BatchOnlyValue.batch_calls == []


def test_mixed_concrete_classes_are_partitioned_inside_one_public_call() -> None:
    BatchOnlyValue.batch_calls.clear()
    OtherBatchValue.batch_calls.clear()

    result = load_or_create(
        [
            BatchOnlyValue(value=1),
            OtherBatchValue(value="x"),
            BatchOnlyValue(value=2),
        ],
        use_lock=False,
    )

    assert result == [10, "X", 20]
    assert BatchOnlyValue.batch_calls == [[1, 2]]
    assert OtherBatchValue.batch_calls == [["x"]]


def test_batch_persistence_is_non_transactional() -> None:
    ok = PartialPersistBatch(key="ok")
    bad = PartialPersistBatch(key="bad")
    tail = PartialPersistBatch(key="tail")

    with pytest.raises(TypeError, match="cannot pickle"):
        load_or_create([ok, bad, tail], use_lock=False)

    assert ok.is_completed()
    assert ok.try_load() == "ok"
    assert not bad.is_completed()
    assert not tail.is_completed()


def test_same_thread_self_reentry_is_rejected() -> None:
    obj = ReentrantSelfLoad(value=1)

    with pytest.raises(RuntimeError, match="same-thread load_or_create reentry"):
        obj.load_or_create(use_lock=False)


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


def test_batch_log_scope_fans_out_compute_logs_but_not_per_object_persistence_logs() -> None:
    left = BatchedLoggedValue(name="left")
    right = BatchedLoggedValue(name="right")

    assert load_or_create([left, right], use_lock=False) == [
        "batched:left",
        "batched:right",
    ]

    left_log = left._log_path.read_text(encoding="utf-8")
    right_log = right._log_path.read_text(encoding="utf-8")

    assert "batch detail for left,right" in left_log
    assert "batch detail for left,right" in right_log

    assert f"stored result at {left._result_path}" in left_log
    assert f"stored result at {right._result_path}" not in left_log
    assert f"stored result at {right._result_path}" in right_log
    assert f"stored result at {left._result_path}" not in right_log
