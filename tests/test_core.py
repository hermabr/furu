import json
import os
import types
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import (
    FrozenInstanceError,
    InitVar,
    dataclass,
    fields,
    is_dataclass,
    replace,
)
from enum import Enum
from functools import cached_property, partial
from pathlib import Path
from typing import Any, ClassVar, Literal, cast
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ConfigDict

import furu
import furu.execution.load_or_create as execution_module
from furu import (
    ResourceRequirements,
    Spec,
    validate,
)
from furu.storage._layout import (
    compute_lock_path_in,
    data_dir_in,
    metadata_path_in,
    result_dir_in,
    result_manifest_path_in,
    run_log_path_in,
)
from furu.config import _FuruConfig, _FuruDirectories, get_config
from furu.dependencies import collect_declared_refs
from furu.execution.load_or_create import _load_or_create
from furu.locking import LockManifest, lock
from furu.logging import _scoped_log_files
from furu.metadata import ArtifactSpec
from furu.result.bundle import _save_result_bundle, load_result_bundle
from furu.serializer.artifact import _from_json, to_json
from furu.testing import override_config
from furu.utils import fully_qualified_name
from furu.worker.context import (
    _DependencyNotReady,
    _worker_execution_lease_id,
    worker_execution_context,
)

type SOME_TYPE = Literal["a", "b"] | int


def write_stale_lock(lock_path: Path) -> None:
    claim_path = lock_path.with_name(f"{lock_path.name}.claim")
    manifest = LockManifest(
        claim_path=claim_path.resolve(strict=False),
        lock_paths=(lock_path.resolve(strict=False),),
    )
    claim_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    os.link(claim_path, lock_path)
    os.utime(claim_path, times=(1, 1))


def _to_json(obj: Any, declared_type: object) -> Any:
    return to_json(
        obj,
        declared_type=declared_type,
        artifact_serializers=(),
    )


class Node(Spec[str]):
    name: str

    def create(self) -> str:
        return f"Node({self.name})"


class MaxWorkersIdentityNode(Spec[str]):
    name: str
    max_workers = 1

    def create(self) -> str:
        return self.name


class WeightedNode(Node):
    weight: float

    def create(self) -> str:
        return f"WNode({self.name}:{self.weight})"


class CustomStorageRootNode(Node):
    @cached_property
    def storage_root(self) -> Path:
        return Path("custom/data/location")


class NodePair(Spec[dict]):
    node1: Node
    node2: WeightedNode
    name: str | int

    def create(self) -> dict:
        return {
            "node1": self.node1.create(),
            "node2": self.node2.create(),
            "name": self.name,
        }


class RandomObj(Spec[float]):
    id: int

    def create(self) -> float:
        # Intentionally non-deterministic to test that create caches results
        import random

        return random.random()


class UserDataWritingValue(Spec[str]):
    name: str

    def create(self) -> str:
        payload_path = self.directory.data / "payload.txt"
        payload_path.write_text(self.name, encoding="utf-8")
        return payload_path.read_text(encoding="utf-8")


class Fruit(Enum):  # TODO: test enums at some point
    apple = "apple"
    banana = "banana"


class A[T](Spec):
    x: int | str | list
    z: T
    w: list[int | float]
    some_obj: SOME_TYPE = "a"

    def create(self) -> None:
        pass


class B[T](Spec):
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
            namespace["create_batched"] = classmethod(
                lambda hidden_cls, objs: cls.create_batched(objs)
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


class VariadicTuple(Spec[None]):
    t: tuple[int, ...]

    def create(self):
        pass


class UsesPath(Spec[str]):
    path: Path

    def create(self) -> str:
        return str(self.path)


class UsesClassValue(Spec[None]):
    node_cls: type[Node]

    def create(self) -> None:
        return None


class UsesContainers(Spec[None]):
    ints: set[int]
    frozen: frozenset[str]
    pair: tuple[Path, int]
    maybe_path: Path | None
    untyped: dict[str, Any]

    def create(self) -> None:
        return None


class UsesFalseLiteral(Spec[None]):
    tie_word_embeddings: Literal[False]

    def create(self) -> None:
        return None


class LoggedLeaf(Spec[str]):
    name: str

    def create(self) -> str:
        self.logger.info("leaf detail for %s", self.name)
        return f"leaf:{self.name}"


class LoggedParent(Spec[dict[str, str]]):
    child: LoggedLeaf

    def create(self) -> dict[str, str]:
        self.logger.info("parent before child")
        child_result = self.child.create()
        self.logger.info("parent after child")
        return {"child": child_result}


class PositiveValue(Spec[int]):
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


class PydanticFields(Spec[None]):
    pydantic_obj: PydanticSubclass

    def create(self) -> None:
        return None


@furu.spec
def letter_count(source: str, letter: str) -> int:
    return source.count(letter)


@furu.spec
def letter_count_with_default(source: str, letter: str = "a") -> int:
    return source.count(letter)


@furu.spec
def letter_count_without_return_annotation(source: str, letter: str):
    return source.count(letter)


@furu.spec
def letter_count_with_untyped_source(source, letter: str) -> int:
    return source.count(letter)


@furu.spec()
def letter_count_with_parentheses(source: str, letter: str) -> int:
    return source.count(letter)


GROUP_EXECUTION_EVENTS: list[tuple[str, tuple[int, ...]]] = []


class CountedSingleValue(Spec[str]):
    key: int
    create_calls: ClassVar[list[int]] = []

    def create(self) -> str:
        type(self).create_calls.append(self.key)
        GROUP_EXECUTION_EVENTS.append(("single", (self.key,)))
        return f"single:{self.key}"


class ObjectIdStorageRootValue(Spec[str]):
    key: int
    storage_root_override: ClassVar[Path] = Path("object-id-root")
    create_calls: ClassVar[list[int]] = []

    @cached_property
    def storage_root(self) -> Path:
        return type(self).storage_root_override

    def create(self) -> str:
        type(self).create_calls.append(self.key)
        return f"object-id:{self.key}"


class NoCreateHookValue(Spec[str]):
    key: int


class BatchOnlyValue(Spec[str]):
    key: int
    batch_calls: ClassVar[list[tuple[int, ...]]] = []

    @classmethod
    def create_batched(cls, objs) -> list[str]:
        keys = tuple(obj.key for obj in objs)
        cls.batch_calls.append(keys)
        return [f"batch:{obj.key}" for obj in objs]


class DelegatingBatchValue(BatchOnlyValue):
    @classmethod
    def create_batched(cls, objs) -> list[str]:
        return BatchOnlyValue.create_batched(objs)


class GroupBatchA(Spec[str]):
    key: int
    batch_calls: ClassVar[list[tuple[int, ...]]] = []

    @classmethod
    def create_batched(cls, objs) -> list[str]:
        keys = tuple(obj.key for obj in objs)
        cls.batch_calls.append(keys)
        GROUP_EXECUTION_EVENTS.append(("batch_a", keys))
        return [f"group-a:{obj.key}" for obj in objs]


class GroupBatchB(Spec[str]):
    key: int
    batch_calls: ClassVar[list[tuple[int, ...]]] = []

    @classmethod
    def create_batched(cls, objs) -> list[str]:
        keys = tuple(obj.key for obj in objs)
        cls.batch_calls.append(keys)
        GROUP_EXECUTION_EVENTS.append(("batch_b", keys))
        return [f"group-b:{obj.key}" for obj in objs]


class LoggedBatchValue(Spec[str]):
    key: int

    @classmethod
    def create_batched(cls, objs) -> list[str]:
        keys = ",".join(str(obj.key) for obj in objs)
        objs[0].logger.info("batched detail for %s", keys)
        return [f"logged-batch:{obj.key}" for obj in objs]


class LoggedSingleValue(Spec[str]):
    key: int

    def create(self) -> str:
        self.logger.info("single detail for %s", self.key)
        return f"logged-single:{self.key}"


class FailingBatchValue(Spec[str]):
    key: int

    @classmethod
    def create_batched(cls, objs) -> list[str]:
        _local_debug_value = "furu-local-debug-value-should-not-leak"
        raise RuntimeError(f"failed batch for {[obj.key for obj in objs]}")


class FailingSingleValue(Spec[str]):
    key: int

    def create(self) -> str:
        raise RuntimeError(f"failed single for {self.key}")


class InterruptingValue(Spec[str]):
    key: int

    def create(self) -> str:
        raise KeyboardInterrupt


class PartialBatchValue(Spec[str]):
    key: int

    @classmethod
    def create_batched(cls, objs) -> list[str]:
        return [f"partial:{obj.key}" for obj in objs]


class MetadataTimingValue(Spec[str]):
    key: int
    create_events: ClassVar[list[tuple[int, bool, bool]]] = []
    siblings_by_key: ClassVar[dict[int, "MetadataTimingValue"]] = {}

    def create(self) -> str:
        sibling_key = 2 if self.key == 1 else 1
        sibling = type(self).siblings_by_key[sibling_key]
        type(self).create_events.append(
            (
                self.key,
                metadata_path_in(self._base_dir).exists(),
                metadata_path_in(sibling._base_dir).exists(),
            )
        )
        return f"timed:{self.key}"


@dataclass(frozen=True)
class DependencyBundle:
    first: Node
    second: WeightedNode


class NestedDependencyParent(Spec[str]):
    bundle: DependencyBundle

    def create(self) -> str:
        return self.bundle.first.create()


class ComputedDependencyParent(Spec[str]):
    name: str

    @furu.dependency
    def child(self) -> Node:
        return Node(name=self.name)

    def create(self) -> str:
        return self.child.create()


class CountingDependencyParent(Spec[str]):
    name: str
    calls: ClassVar[int] = 0

    @furu.dependency
    def child(self) -> Node:
        type(self).calls += 1
        return Node(name=f"{self.name}-{type(self).calls}")

    def create(self) -> str:
        return self.child.create()


class LazyDependencyParent(Spec[str]):
    name: str

    def create(self) -> str:
        return Node(name=self.name).create()


class LoadExistingDependencyParent(Spec[str]):
    name: str

    def create(self) -> str:
        try:
            Node(name=self.name).load_existing()
        except furu.Missing:
            return "missing"
        return "loaded"


class FunctionDependencyParent(Spec[int]):
    child: letter_count  # ty: ignore[invalid-type-form]

    def create(self) -> int:
        return self.child.create()


class FuruBoundaryParent(Spec[str]):
    child: NodePair

    def create(self) -> str:
        return self.child.node1.create()


class BatchDependencyParent(Spec[str]):
    key: int
    eager: Node

    @classmethod
    def create_batched(cls, objs) -> list[str]:
        eager_values = [obj.eager.create() for obj in objs]
        lazy_value = Node(name="shared-lazy").create()
        return [f"{value}:{lazy_value}" for value in eager_values]


@pytest.fixture(autouse=True)
def _reset_batch_trackers() -> None:
    CountedSingleValue.create_calls.clear()
    ObjectIdStorageRootValue.storage_root_override = Path("object-id-root")
    ObjectIdStorageRootValue.create_calls.clear()
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


def test_spec_function_creates_spec_from_function_signature():
    obj = letter_count("banana", "a")

    assert isinstance(obj, Spec)
    assert is_dataclass(type(obj))
    assert type(obj).__name__ == "letter_count"
    assert getattr(obj, "source") == "banana"
    assert getattr(obj, "letter") == "a"
    assert obj.create() == 3
    assert letter_count(source="banana", letter="a") == obj

    with pytest.raises(FrozenInstanceError):
        obj.source = "orange"  # ty: ignore[invalid-assignment]


def test_spec_function_supports_defaults_and_artifact_round_trip():
    obj = letter_count_with_default("banana")

    assert getattr(obj, "letter") == "a"
    assert obj.create() == 3
    assert obj._fully_qualified_name == "test_core.letter_count_with_default"
    assert _from_json(obj._artifact_data) == obj


def test_spec_function_allows_missing_return_annotation():
    assert letter_count_without_return_annotation("banana", "n").create() == 2


def test_spec_function_defaults_unannotated_parameters_to_any():
    obj = letter_count_with_untyped_source("banana", "a")

    assert obj.create() == 3
    assert obj._schema_data == {
        "|class": "test_core.letter_count_with_untyped_source",
        "|fields": {"letter": "builtins.str", "source": "typing.Any"},
    }
    assert _from_json(obj._artifact_data) == obj


def test_spec_function_supports_parenthesized_decorator():
    obj = letter_count_with_parentheses("banana", "n")

    assert isinstance(obj, Spec)
    assert getattr(obj, "source") == "banana"
    assert getattr(obj, "letter") == "n"
    assert obj.create() == 2
    assert obj._fully_qualified_name == "test_core.letter_count_with_parentheses"
    assert _from_json(obj._artifact_data) == obj


def test_spec_function_returns_the_spec_type():
    obj = letter_count("banana", "a")

    assert isinstance(letter_count, type)
    assert issubclass(cast(type, letter_count), Spec)
    assert type(obj) is letter_count
    assert obj.create() == 3
    assert obj._fully_qualified_name == "test_core.letter_count"
    assert _from_json(obj._artifact_data) == obj


def test_spec_function_rejects_variadic_parameters():
    with pytest.raises(
        TypeError,
        match=r"variadic_letter_count\.sources",
    ):

        @furu.spec
        def variadic_letter_count(*sources: str) -> int:
            return sum(source.count("a") for source in sources)


def test_class_level_validation():
    assert PositiveValue(value=2).create() == 2

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


def test_reserved_field_name_raises_at_class_creation():
    with pytest.raises(TypeError) as excinfo:

        class StatusField(Spec[int]):
            status: str  # ty: ignore[override-of-final-method]

            def create(self) -> int:
                return 0

    message = str(excinfo.value)
    assert "StatusField" in message
    assert "['status']" in message
    for name in (
        "create",
        "create_batched",
        "metadata",
        "status",
        "directory",
        "explain",
        "load_existing",
        "delete",
        "migrate",
        "migrations",
        "provenance",
    ):
        assert f"'{name}'" in message


def test_unannotated_public_attribute_raises_clear_error():
    with pytest.raises(
        TypeError, match="UnannotatedParameter.a must have a type annotation"
    ):

        class UnannotatedParameter(Spec[int]):
            a = 1

            def create(self) -> int:
                return self.a


def test_unannotated_private_attribute_raises_clear_error():
    with pytest.raises(
        TypeError, match="UnannotatedPrivate._a must have a type annotation"
    ):

        class UnannotatedPrivate(Spec[int]):
            _a = 1

            def create(self) -> int:
                return self._a


def test_validate_decorator_supports_call_syntax():
    class CallSyntaxValidated(Spec[int]):
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
    class PostInitValidated(Spec[int]):
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


def test_post_init_with_init_var_and_validators_both_run():
    class InitVarValidated(Spec[int]):
        scale: InitVar[int]
        value: int = 0

        def __post_init__(self, scale: int) -> None:
            object.__setattr__(self, "value", self.value * scale)

        @validate
        def _validate_positive(self) -> None:
            if self.value <= 0:
                raise ValueError("value must be positive")

        def create(self) -> int:
            return self.value

    assert InitVarValidated(scale=3, value=2).value == 6

    with pytest.raises(ValueError, match="value must be positive"):
        InitVarValidated(scale=0, value=2)


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
    class BasePostInitValue(Spec[int]):
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

    class BaseOrderedValue(Spec[int]):
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
        )._artifact_hash
        == "685af925669262434640"
    )

    assert (
        NodePair(
            name="x", node1=Node(name="y"), node2=WeightedNode(name="z", weight=1)
        )._artifact_hash
        != NodePair(
            name="z", node1=Node(name="y"), node2=WeightedNode(name="z", weight=1)
        )._artifact_hash
    )

    assert (
        NodePair(
            name="y", node1=Node(name="y"), node2=WeightedNode(name="z", weight=1)
        )._artifact_schema_hash
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
        )._artifact_schema_hash
        != B_priv(
            a=A(x=1, z="123", w=[6, 7]),
            y={"hey": 123, True: 1},
            t=("123", 12),
            _h=1,
        )._artifact_schema_hash
    )

    def qualname_alias(cls: type[Spec[object]], *, ret_typ: type) -> type[Spec[object]]:
        namespace: dict[str, object] = {"__module__": cls.__module__}
        if cls._furu_create_mode == "single":
            namespace["create"] = lambda self: cls.create(self)
        else:
            namespace["create_batched"] = classmethod(
                lambda alias_cls, objs: cls.create_batched(objs)
            )
        alias = type(ret_typ.__qualname__, (cls,), namespace)
        alias.__qualname__ = ret_typ.__qualname__
        return alias

    B_priv_as_B = cast(type[B_priv], qualname_alias(B_priv, ret_typ=B))
    assert (
        B(
            a=A(x=1, z="123", w=[6, 7]), y={"ney": 123, True: 1}, t=("123", 12)
        )._artifact_schema_hash
        != B_priv_as_B(
            a=A(x=1, z="123", w=[6, 7]),
            y={"hey": 123, "ney": 1},
            t=("123", 12),
            _h=1,
        )._artifact_schema_hash
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
def test_schema(make: Callable[[], Spec], expected):
    assert make()._schema_data == expected


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
    assert _to_json(node_pair, NodePair) == expected
    assert _to_json(node_pair, NodePair) == node_pair._artifact_data
    assert node_pair._artifact_data == expected


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
            "t": {"|kind": "tuple", "|value": ["123", 12]},
            "maybe_val": None,
        },
    }

    assert _to_json(obj, B) == expected


def test_to_json_with_class_field_value():
    obj = UsesClassValue(node_cls=Node)

    assert _to_json(Node, type) == {"|kind": "type_ref", "|class": "test_core.Node"}
    assert obj._artifact_data == {
        "|kind": "instance",
        "|class": "test_core.UsesClassValue",
        "|fields": {"node_cls": {"|kind": "type_ref", "|class": "test_core.Node"}},
    }
    assert isinstance(obj._artifact_hash, str)


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

    assert _to_json(obj, PydanticFields) == expected
    assert obj._artifact_data == expected
    assert isinstance(obj._artifact_hash, str)


def test_furu_object_round_trips_from_json_artifact():
    obj = NodePair(
        name="x",
        node1=Node(name="y"),
        node2=WeightedNode(name="z", weight=1),
    )

    loaded = _from_json(obj._artifact_data)

    assert loaded == obj
    assert isinstance(loaded, NodePair)
    assert loaded.object_id == obj.object_id


def test_furu_object_with_typed_fields_round_trips_from_json_artifact():
    path_obj = UsesPath(path=Path("/tmp/out"))
    class_obj = UsesClassValue(node_cls=Node)

    assert _from_json(path_obj._artifact_data) == path_obj
    assert _from_json(class_obj._artifact_data) == class_obj
    assert isinstance(cast(UsesPath, _from_json(path_obj._artifact_data)).path, Path)


def test_container_fields_round_trip_with_exact_types():
    obj = UsesContainers(
        ints={3, 1, 2},
        frozen=frozenset({"b", "a"}),
        pair=(Path("/tmp/out"), 7),
        maybe_path=Path("/tmp/maybe"),
        untyped={"tup": (1, 2), "nested": {4, 5}},
    )

    loaded = cast(UsesContainers, _from_json(obj._artifact_data))

    assert loaded == obj
    assert loaded.ints == {1, 2, 3}
    assert isinstance(loaded.ints, set)
    assert isinstance(loaded.frozen, frozenset)
    assert loaded.pair == (Path("/tmp/out"), 7)
    assert isinstance(loaded.pair, tuple)
    assert isinstance(loaded.pair[0], Path)
    assert isinstance(loaded.maybe_path, Path)
    assert loaded.untyped == {"tup": (1, 2), "nested": {4, 5}}
    assert loaded.object_id == obj.object_id


def test_set_artifact_data_is_deterministic():
    first = UsesContainers(
        ints={1, 2, 3},
        frozen=frozenset({"a", "b"}),
        pair=(Path("/tmp/out"), 7),
        maybe_path=None,
        untyped={},
    )
    second = UsesContainers(
        ints={3, 2, 1},
        frozen=frozenset({"b", "a"}),
        pair=(Path("/tmp/out"), 7),
        maybe_path=None,
        untyped={},
    )

    assert first._artifact_data == second._artifact_data
    assert first._artifact_hash == second._artifact_hash


def test_furu_from_artifact_returns_furu_object():
    obj = NodePair(
        name="x",
        node1=Node(name="y"),
        node2=WeightedNode(name="z", weight=1),
    )
    obj.create()
    artifact = ArtifactSpec(
        fully_qualified_name=obj._fully_qualified_name,
        artifact_data=obj._artifact_data,
        artifact_hash=obj._artifact_hash,
        schema_data=obj._schema_data,
        schema_hash=obj._artifact_schema_hash,
    )

    loaded = NodePair.from_artifact(artifact)
    raw_metadata = json.loads(metadata_path_in(obj._base_dir).read_text())

    assert artifact.object_id == obj.object_id
    assert artifact.schema_data == obj._schema_data
    assert "schema_data" in type(artifact).model_fields
    assert loaded == obj
    assert isinstance(loaded, NodePair)
    assert loaded.directory.data == obj.directory.data
    assert raw_metadata["kind"] == "completed"
    assert raw_metadata["base_path"] == str(obj._base_dir)
    assert "data_path" not in raw_metadata
    assert raw_metadata["artifact"] == {
        "fully_qualified_name": obj._fully_qualified_name,
        "artifact_data": obj._artifact_data,
        "artifact_hash": obj._artifact_hash,
        "schema_data": obj._schema_data,
        "schema_hash": obj._artifact_schema_hash,
    }
    assert "hash" not in raw_metadata["artifact"]
    assert "artifact_schema" not in raw_metadata
    assert "artifact_schema_hash" not in raw_metadata


def _dependency_object_ids(obj: Spec) -> list[str]:
    metadata = json.loads(metadata_path_in(obj._base_dir).read_text())
    return metadata["observed_dependencies"]


def test_field_dependencies_are_eager_but_metadata_stores_only_loaded_objects() -> None:
    first = Node(name="nested")
    second = WeightedNode(name="weighted", weight=2)
    parent = NestedDependencyParent(bundle=DependencyBundle(first=first, second=second))

    assert collect_declared_refs(parent) == (first, second)
    assert parent.create() == "Node(nested)"
    assert _dependency_object_ids(parent) == [first.object_id]


def test_computed_dependency_is_cached_property_and_eager_loaded_dependency() -> None:
    parent = ComputedDependencyParent(name="computed")

    assert parent.child is parent.child
    assert collect_declared_refs(parent) == (parent.child,)
    assert parent.create() == "Node(computed)"

    assert _dependency_object_ids(parent) == [parent.child.object_id]


def test_dependency_computes_once_per_instance() -> None:
    CountingDependencyParent.calls = 0
    parent = CountingDependencyParent(name="counting")

    assert parent.child is parent.child
    assert collect_declared_refs(parent) == (parent.child,)
    assert CountingDependencyParent.calls == 1


def test_create_inside_create_is_recorded_and_deduped() -> None:
    parent = LazyDependencyParent(name="lazy")

    assert parent.create() == "Node(lazy)"

    assert _dependency_object_ids(parent) == [Node(name="lazy").object_id]


def test_load_existing_inside_create_is_recorded_even_on_missing_result() -> None:
    parent = LoadExistingDependencyParent(name="optional")

    assert parent.create() == "missing"

    metadata = json.loads(metadata_path_in(parent._base_dir).read_text())
    assert metadata["observed_dependencies"] == [Node(name="optional").object_id]


def test_load_existing_missing_result_explains_how_to_compute() -> None:
    with pytest.raises(
        furu.Missing,
        match=(
            r"Node:[a-f0-9]{5}:[a-f0-9]{5}\.load_existing\(\) could not find a result\. "
            r"load_existing\(\) only loads existing results; use create\(\) to "
            r"compute missing results\."
        ),
    ):
        Node(name="missing").load_existing()


def test_top_level_create_accepts_single_spec_and_sequence() -> None:
    single = Node(name="top-create-single")
    assert furu.create(single) == "Node(top-create-single)"

    nodes = [Node(name="top-create-a"), Node(name="top-create-b")]
    assert furu.create(nodes) == ["Node(top-create-a)", "Node(top-create-b)"]


def test_top_level_load_existing_rejects_single_furu_object() -> None:
    node = Node(name="single-load")

    assert node.create() == "Node(single-load)"

    with pytest.raises(TypeError, match="expected a sequence of Spec objects"):
        furu.load_existing(node)  # ty:ignore[invalid-argument-type]


def test_top_level_load_existing_accepts_list_and_logs_once(tmp_path: Path) -> None:
    nodes = [Node(name="load-a"), Node(name="load-b")]

    assert [node.create() for node in nodes] == ["Node(load-a)", "Node(load-b)"]

    log_path = tmp_path / "load-existing.log"
    with _scoped_log_files((log_path,)):
        assert furu.load_existing(nodes) == ["Node(load-a)", "Node(load-b)"]

    info_lines = [
        line
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if "level=info" in line
    ]
    assert len(info_lines) == 1
    assert info_lines[0].endswith(
        f'msg="loaded 2 furu objects including {nodes[0]._log_label}"'
    )


def test_function_type_can_be_declared_field_dependency() -> None:
    child = letter_count("banana", "a")
    parent = FunctionDependencyParent(child=child)

    assert child._schema_data == {
        "|class": "test_core.letter_count",
        "|fields": {"letter": "builtins.str", "source": "builtins.str"},
    }
    assert child._artifact_data == {
        "|kind": "instance",
        "|class": "test_core.letter_count",
        "|fields": {"letter": "a", "source": "banana"},
    }
    assert parent._schema_data == {
        "|class": "test_core.FunctionDependencyParent",
        "|fields": {
            "child": {
                "|class": "test_core.letter_count",
                "|fields": {"letter": "builtins.str", "source": "builtins.str"},
            }
        },
    }
    assert parent._artifact_data == {
        "|kind": "instance",
        "|class": "test_core.FunctionDependencyParent",
        "|fields": {
            "child": {
                "|kind": "instance",
                "|class": "test_core.letter_count",
                "|fields": {"letter": "a", "source": "banana"},
            }
        },
    }
    assert collect_declared_refs(parent) == (child,)
    assert parent.create() == 3
    assert _dependency_object_ids(parent) == [child.object_id]


def test_furu_objects_block_nested_eager_traversal_but_direct_runtime_loads_are_recorded() -> (
    None
):
    node1 = Node(name="inner")
    node2 = WeightedNode(name="other", weight=3)
    child = NodePair(node1=node1, node2=node2, name="pair")
    parent = FuruBoundaryParent(child=child)

    assert parent.create() == "Node(inner)"

    assert collect_declared_refs(parent) == (child,)
    assert _dependency_object_ids(parent) == [node1.object_id]


def test_batched_dependencies_record_all_observed_loads() -> None:
    objs = [
        BatchDependencyParent(key=1, eager=Node(name="eager-1")),
        BatchDependencyParent(key=2, eager=Node(name="eager-2")),
    ]

    assert _load_or_create(objs) == [
        "Node(eager-1):Node(shared-lazy)",
        "Node(eager-2):Node(shared-lazy)",
    ]

    for obj in objs:
        assert collect_declared_refs(obj) == (obj.eager,)
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
    obj.create()
    artifact = ArtifactSpec(
        fully_qualified_name=obj._fully_qualified_name,
        artifact_data=obj._artifact_data,
        artifact_hash=obj._artifact_hash,
        schema_data=obj._schema_data,
        schema_hash=obj._artifact_schema_hash,
    )

    loaded = Spec.from_artifact(artifact)

    assert artifact.object_id == obj.object_id
    assert loaded == obj
    assert isinstance(loaded, NodePair)


def test_furu_from_artifact_accepts_loaded_metadata_artifact():
    obj = Node(name="x")
    obj.create()
    metadata = json.loads(metadata_path_in(obj._base_dir).read_text())
    artifact = ArtifactSpec(**metadata["artifact"])

    loaded = Node.from_artifact(artifact)

    assert artifact.object_id == obj.object_id
    assert loaded == obj
    assert isinstance(loaded, Node)


def test_furu_from_artifact_accepts_artifact_spec():
    obj = Node(name="x")
    artifact = ArtifactSpec(
        fully_qualified_name=obj._fully_qualified_name,
        artifact_data=obj._artifact_data,
        artifact_hash=obj._artifact_hash,
        schema_data=obj._schema_data,
        schema_hash=obj._artifact_schema_hash,
    )

    loaded = Node.from_artifact(artifact)

    assert artifact.object_id == obj.object_id
    assert loaded == obj
    assert isinstance(loaded, Node)


def test_furu_from_artifact_type_mismatch_names_expected_and_loaded_type():
    obj = WeightedNode(name="x", weight=1)
    artifact = ArtifactSpec(
        fully_qualified_name=obj._fully_qualified_name,
        artifact_data=obj._artifact_data,
        artifact_hash=obj._artifact_hash,
        schema_data=obj._schema_data,
        schema_hash=obj._artifact_schema_hash,
    )

    with pytest.raises(
        TypeError,
        match=(
            r"Artifact described test_core\.WeightedNode, "
            r"expected test_core\.NodePair"
        ),
    ):
        NodePair.from_artifact(artifact)


def test_furu_from_artifact_rejects_artifact_spec_hash_mismatch():
    obj = Node(name="x")
    bad_hash = "wrong-artifact-hash"
    artifact = ArtifactSpec(
        fully_qualified_name=obj._fully_qualified_name,
        artifact_data=obj._artifact_data,
        artifact_hash=bad_hash,
        schema_data=obj._schema_data,
        schema_hash=obj._artifact_schema_hash,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Artifact hash did not match loaded object: "
            + f"artifact={bad_hash[:5]}, loaded={obj._artifact_hash[:5]}"
        ),
    ):
        Node.from_artifact(artifact)


def test_furu_from_artifact_rejects_artifact_spec_schema_hash_mismatch():
    obj = Node(name="x")
    bad_schema_hash = "wrong-schema-hash"
    artifact = ArtifactSpec(
        fully_qualified_name=obj._fully_qualified_name,
        artifact_data=obj._artifact_data,
        artifact_hash=obj._artifact_hash,
        schema_data=obj._schema_data,
        schema_hash=bad_schema_hash,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Artifact schema hash did not match loaded object: "
            + f"artifact={bad_schema_hash[:5]}, "
            + f"loaded={obj._artifact_schema_hash[:5]}"
        ),
    ):
        Node.from_artifact(artifact)


def test_schema_with_ellipsis_type_arg():
    assert VariadicTuple(t=(1, 2, 3))._schema_data == {
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
    assert node_pair._base_dir == (
        get_config().run_directories.objects
        / "test_core"
        / "NodePair"
        / "21733b1febfab88b565c"
        / "685af925669262434640"
    )
    assert node_pair.directory.data == node_pair._base_dir / "data"
    assert node_pair.directory.data == Path(
        get_config().run_directories.objects
        / "test_core"
        / "NodePair"
        / node_pair._artifact_schema_hash
        / node_pair._artifact_hash
        / "data"
    )


def test_resource_requirements_defaults_to_none():
    assert Node(name="x").resource_requirements is None


def test_max_workers_defaults_to_none():
    assert Node(name="x").max_workers is None


def test_max_workers_can_be_overridden_as_class_option():
    class LimitedNode(Spec[str]):
        name: str
        max_workers = 5

        def create(self) -> str:
            return self.name

    assert LimitedNode(name="x").max_workers == 5
    assert "max_workers" not in {field.name for field in fields(LimitedNode)}


def test_max_workers_can_be_overridden_with_classvar():
    class LimitedNode(Spec[str]):
        name: str
        max_workers: ClassVar[int | None] = 5

        def create(self) -> str:
            return self.name

    assert LimitedNode(name="x").max_workers == 5


def test_max_workers_does_not_affect_schema_or_object_identity():
    original_max_workers = MaxWorkersIdentityNode.max_workers
    before = MaxWorkersIdentityNode(name="x")
    before_schema_hash = before._artifact_schema_hash
    before_artifact_hash = before._artifact_hash
    before_object_id = before.object_id

    try:
        MaxWorkersIdentityNode.max_workers = 5
        after = MaxWorkersIdentityNode(name="x")
    finally:
        MaxWorkersIdentityNode.max_workers = original_max_workers

    assert after._artifact_schema_hash == before_schema_hash
    assert after._artifact_hash == before_artifact_hash
    assert after.object_id == before_object_id


def test_resource_requirements_can_be_overridden_with_property():
    class HeavyNode(Spec[str]):
        name: str

        @property
        def resource_requirements(self) -> ResourceRequirements | None:
            return ResourceRequirements(
                cpus=(4, 8), gpus=(1, None), memory_gib=(16, None)
            )

        def create(self) -> str:
            return self.name

    rr = HeavyNode(name="x").resource_requirements
    assert rr == ResourceRequirements(
        cpus=(4, 8), gpus=(1, None), memory_gib=(16, None)
    )
    assert rr is not None


def test_storage_root_can_be_overridden_with_cached_property(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    node = CustomStorageRootNode(name="x")

    assert node.storage_root == Path("custom/data/location")
    assert node.storage_root is node.storage_root
    assert node._base_dir == (
        Path("custom/data/location")
        / "test_core"
        / "CustomStorageRootNode"
        / node._artifact_schema_hash
        / node._artifact_hash
    )
    assert node.directory.data == node._base_dir / "data"


def test_debug_mode_ignores_storage_root_override() -> None:
    with override_config(_FuruConfig(debug_mode=True)):
        node = CustomStorageRootNode(name="x")

        assert node.storage_root == Path("custom/data/location")
        assert node._base_dir == (
            Path("furu-data")
            / "debug"
            / "objects"
            / "test_core"
            / "CustomStorageRootNode"
            / node._artifact_schema_hash
            / node._artifact_hash
        )


def test_debug_mode_uses_configured_debug_directory() -> None:
    config = _FuruConfig(
        debug_mode=True,
        directories=_FuruDirectories(
            objects=Path("main/objects"),
            executions=Path("main/executions"),
            debug=Path("custom/debug"),
        ),
    )
    with override_config(config):
        node = CustomStorageRootNode(name="x")

        assert node.storage_root == Path("custom/data/location")
        assert node._base_dir == (
            Path("custom/debug")
            / "objects"
            / "test_core"
            / "CustomStorageRootNode"
            / node._artifact_schema_hash
            / node._artifact_hash
        )


def test_create_object_and_exists():
    node_pair = NodePair(
        name="x", node1=Node(name="y"), node2=WeightedNode(name="z", weight=1)
    )
    assert node_pair.status == "missing"
    for _i in range(3):
        assert node_pair.create() == {
            "node1": "Node(y)",
            "node2": "WNode(z:1)",
            "name": "x",
        }
    assert node_pair.status == "done"
    assert replace(node_pair, name="y").status == "missing"


def test_data_dir_is_user_data_subdirectory() -> None:
    obj = UserDataWritingValue(name="payload")

    assert obj.create() == "payload"

    assert (obj.directory.data / "payload.txt").read_text(encoding="utf-8") == "payload"
    assert result_manifest_path_in(obj._base_dir).exists()
    assert metadata_path_in(obj._base_dir).exists()
    assert not (obj._base_dir / ".furu").exists()


def test_unused_data_dir_is_not_created_by_create() -> None:
    node = Node(name="unused-data")
    user_data_path = node._base_dir / "data"

    assert node.create() == "Node(unused-data)"

    assert not user_data_path.exists()


def test_data_dir_property_creates_user_data_subdirectory() -> None:
    node = Node(name="manual-data")
    user_data_path = node._base_dir / "data"

    assert not user_data_path.exists()
    assert node.directory.data == user_data_path
    assert user_data_path.exists()
    assert node.status == "failed"


def test_status_is_running_while_compute_lock_is_held() -> None:
    node = Node(name="x")
    node._base_dir.mkdir(parents=True, exist_ok=True)

    with lock([compute_lock_path_in(node._base_dir)]):
        assert node.status == "running"


def test_status_is_failed_when_compute_lock_is_not_active() -> None:
    node = Node(name="inactive-lock")
    node._base_dir.mkdir(parents=True, exist_ok=True)

    compute_lock_path_in(node._base_dir).touch()

    assert node.status == "failed"


def test_status_is_failed_when_compute_lock_is_stale() -> None:
    node = Node(name="stale-lock")
    node._base_dir.mkdir(parents=True, exist_ok=True)
    lock_path = compute_lock_path_in(node._base_dir)

    write_stale_lock(lock_path)

    assert node.status == "failed"


def test_status_is_failed_after_create_error() -> None:
    obj = FailingSingleValue(key=1)

    with pytest.raises(RuntimeError, match="failed single for 1"):
        obj.create()

    assert obj.status == "failed"


def test_creating_and_loading_random_result_furu_obj():
    n_ids = 5
    results = {
        obj_id: [RandomObj(id=obj_id).create() for _ in range(3)]
        for obj_id in range(n_ids)
    }
    assert all(len(set(values)) == 1 for values in results.values())
    assert len({values[0] for values in results.values()}) == n_ids


def test_delete_force() -> None:
    node = Node(name="x")

    assert node.create() == "Node(x)"
    assert node.directory.data.exists()
    assert node.delete(mode="force")
    assert not node.directory.data.exists()
    assert node.create() == "Node(x)"


def test_delete_prompt_cancel() -> None:
    node = Node(name="x")

    assert node.create() == "Node(x)"
    with patch("builtins.input", return_value="n"):
        assert not node.delete()
    assert node.directory.data.exists()


def test_delete_returns_false_when_missing() -> None:
    assert not Node(name="x").delete(mode="force")


def test_log_file_is_written_to_base_dir() -> None:
    node = LoggedLeaf(name="x")

    assert node.create() == "leaf:x"

    assert run_log_path_in(node._base_dir).parent == node._base_dir
    log_text = run_log_path_in(node._base_dir).read_text(encoding="utf-8")
    assert "leaf detail for x" in log_text


def test_nested_create_scopes_logs_to_child_file() -> None:
    child = LoggedLeaf(name="child")
    parent = LoggedParent(child=child)

    assert parent.create() == {"child": "leaf:child"}

    parent_log = run_log_path_in(parent._base_dir).read_text(encoding="utf-8")
    child_log = run_log_path_in(child._base_dir).read_text(encoding="utf-8")

    assert "parent before child" in parent_log
    assert f"creating {child._log_label}" in parent_log
    assert f"(object_id={child.object_id})" not in parent_log
    assert f"finished {child._log_label} ok" in parent_log
    assert "parent after child" in parent_log
    assert "leaf detail for child" not in parent_log

    assert "leaf detail for child" in child_log


def test_cached_create_logs_debug_call_and_only_cache_hit_info(
    tmp_path: Path,
) -> None:
    ObjectIdStorageRootValue.storage_root_override = tmp_path / "objects"
    obj = ObjectIdStorageRootValue(key=1)

    assert obj.create() == "object-id:1"

    log_path = tmp_path / "cached-create.log"
    with _scoped_log_files((log_path,)):
        assert obj.create() == "object-id:1"

    log_text = log_path.read_text(encoding="utf-8")
    assert f".create called for {obj}" in log_text
    assert f"cached {obj._log_label}" in log_text
    assert "building" not in log_text
    assert "creating " not in log_text
    assert "finished " not in log_text

    info_lines = [line for line in log_text.splitlines() if "level=info" in line]
    assert len(info_lines) == 1
    assert info_lines[0].endswith(f'msg="cached {obj._log_label}"')


def test_small_cache_summary_logs_labels_for_cached_and_missing_items(
    tmp_path: Path,
) -> None:
    ObjectIdStorageRootValue.storage_root_override = tmp_path / "objects"
    cached = ObjectIdStorageRootValue(key=1)
    missing = ObjectIdStorageRootValue(key=2)

    assert cached.create() == "object-id:1"

    log_path = tmp_path / "mixed-create.log"
    with _scoped_log_files((log_path,)):
        assert _load_or_create([cached, missing]) == ["object-id:1", "object-id:2"]

    assert (
        f"building {missing._log_label}, cached {cached._log_label}"
        in log_path.read_text(encoding="utf-8")
    )


def test_resolved_create_mode_validation() -> None:
    class ExplicitSingle(Node):
        label: str

        def create(self) -> str:
            return f"single:{self.label}"

    class ExplicitBatch(Spec[str]):
        label: str

        @classmethod
        def create_batched(cls, objs) -> list[str]:
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
        TypeError, match="must define exactly one of create or create_batched"
    ):

        class InvalidBoth(Spec[int]):
            def create(self) -> int:
                return 1

            @classmethod
            def create_batched(cls, objs) -> list[int]:
                return [1 for _ in objs]

    with pytest.raises(TypeError, match=r"create_batched must be a @classmethod"):

        class InvalidBatchMethod(Spec[int]):
            def create_batched(self, objs) -> list[int]:
                return [1 for _ in objs]

    with pytest.raises(
        TypeError, match="must define exactly one of create or create_batched"
    ):

        class InvalidInherited(Node):
            label: str

            @classmethod
            def create_batched(cls, objs) -> list[str]:
                return [obj.label for obj in objs]

    class NoCreateHook(Spec[int]):
        value: int

    assert NoCreateHook._furu_create_mode is None


def test_no_create_hook_loads_cached_result() -> None:
    obj = NoCreateHookValue(key=1)
    _save_result_bundle(
        "cached:1",
        result_dir_in(obj._base_dir),
        declared_type=str,
        result_codecs=(),
        data_dir=obj.directory.data,
    )

    assert obj.create() == "cached:1"


def test_no_create_hook_raises_only_for_missing_result() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "NoCreateHookValue cannot create missing results because it does not define "
            r"create\(\) or create_batched\(\)"
        ),
    ):
        NoCreateHookValue(key=2).create()


def test_no_create_hook_uses_post_lock_cache_recheck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obj = NoCreateHookValue(key=3)

    @contextmanager
    def fake_lock(lock_paths: list[Path], **_: object):
        assert lock_paths == [compute_lock_path_in(obj._base_dir)]
        _save_result_bundle(
            "cached-after-lock:1",
            result_dir_in(obj._base_dir),
            declared_type=str,
            result_codecs=(),
            data_dir=obj.directory.data,
        )
        yield lambda: True

    monkeypatch.setattr(execution_module, "lock", fake_lock)

    assert obj.create() == "cached-after-lock:1"


def test_single_object_on_batch_only_class_uses_create_batched() -> None:
    assert _load_or_create(BatchOnlyValue(key=1)) == "batch:1"
    assert BatchOnlyValue.batch_calls == [(1,)]


def test_create_batched_can_delegate_to_base_implementation() -> None:
    objs = [DelegatingBatchValue(key=1), DelegatingBatchValue(key=2)]

    assert _load_or_create(objs) == ["batch:1", "batch:2"]
    assert BatchOnlyValue.batch_calls == [(1, 2)]


def test_list_input_on_single_only_class_uses_sequential_create() -> None:
    objs = [
        CountedSingleValue(key=1),
        CountedSingleValue(key=2),
        CountedSingleValue(key=3),
    ]

    assert _load_or_create(objs) == ["single:1", "single:2", "single:3"]
    assert CountedSingleValue.create_calls == [1, 2, 3]


def test_sequential_fallback_writes_running_metadata_per_object() -> None:
    first = MetadataTimingValue(key=1)
    second = MetadataTimingValue(key=2)
    MetadataTimingValue.siblings_by_key.update({1: first, 2: second})

    assert _load_or_create([first, second], use_lock=False) == ["timed:1", "timed:2"]
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

    assert _load_or_create(objs) == ["group-a:1", "group-b:1", "group-a:2", "group-b:2"]
    assert GroupBatchA.batch_calls == [(1, 2)]
    assert GroupBatchB.batch_calls == [(1, 2)]


def test_duplicate_cache_identities_compute_once_and_preserve_input_order() -> None:
    objs = [
        CountedSingleValue(key=1),
        CountedSingleValue(key=1),
        CountedSingleValue(key=2),
        CountedSingleValue(key=1),
    ]

    assert _load_or_create(objs) == ["single:1", "single:1", "single:2", "single:1"]
    assert CountedSingleValue.create_calls == [1, 2]


def test_executor_deduplicates_by_object_id_not_data_dir(tmp_path: Path) -> None:
    ObjectIdStorageRootValue.storage_root_override = tmp_path / "first"
    first = ObjectIdStorageRootValue(key=1)
    first_data_dir = first.directory.data

    ObjectIdStorageRootValue.storage_root_override = tmp_path / "second"
    second = ObjectIdStorageRootValue(key=1)
    second_data_dir = second.directory.data

    assert first.object_id == second.object_id
    assert first_data_dir != second_data_dir

    assert _load_or_create([first, second]) == ["object-id:1", "object-id:1"]
    assert ObjectIdStorageRootValue.create_calls == [1]


def test_existing_items_are_skipped_before_locking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = CountedSingleValue(key=1)
    missing = CountedSingleValue(key=2)

    assert existing.create() == "single:1"

    lock_calls: list[list[Path]] = []

    @contextmanager
    def fake_lock(lock_paths: list[Path], **_: object):
        lock_calls.append(lock_paths)
        yield lambda: True

    monkeypatch.setattr(execution_module, "lock", fake_lock)

    assert _load_or_create([existing, missing]) == ["single:1", "single:2"]
    assert lock_calls == [[compute_lock_path_in(missing._base_dir)]]


def test_pending_items_are_rechecked_after_lock_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pending = CountedSingleValue(key=5)

    @contextmanager
    def fake_lock(lock_paths: list[Path], **_: object):
        assert lock_paths == [compute_lock_path_in(pending._base_dir)]
        _save_result_bundle(
            "single:5",
            result_dir_in(pending._base_dir),
            result_codecs=(),
            data_dir=pending.directory.data,
        )
        yield lambda: True

    monkeypatch.setattr(execution_module, "lock", fake_lock)

    assert _load_or_create([pending]) == ["single:5"]
    assert CountedSingleValue.create_calls == []


def test_empty_list_returns_empty_list() -> None:
    assert _load_or_create([]) == []


def test_worker_execution_context_is_scoped() -> None:
    assert _worker_execution_lease_id.get() is None
    with worker_execution_context(
        lease_id="lease-1",
    ):
        assert _worker_execution_lease_id.get() == "lease-1"
    assert _worker_execution_lease_id.get() is None


def test_worker_create_loads_cached_result_without_recomputing(
    tmp_path: Path,
) -> None:
    ObjectIdStorageRootValue.storage_root_override = tmp_path / "data"
    cached = ObjectIdStorageRootValue(key=10)

    assert cached.create() == "object-id:10"
    ObjectIdStorageRootValue.create_calls.clear()

    with worker_execution_context(
        lease_id="lease-1",
    ):
        assert cached.create() == "object-id:10"

    assert ObjectIdStorageRootValue.create_calls == []


def test_worker_create_reports_all_missing_dependencies(
    tmp_path: Path,
) -> None:
    ObjectIdStorageRootValue.storage_root_override = tmp_path / "data"
    first = ObjectIdStorageRootValue(key=11)
    second = ObjectIdStorageRootValue(key=12)

    with (
        worker_execution_context(
            lease_id="lease-1",
        ),
        pytest.raises(_DependencyNotReady) as exc_info,
    ):
        _load_or_create([first, second])

    exc = exc_info.value
    assert exc.call_kind == "create"
    assert exc.dependencies == (first, second)
    assert ObjectIdStorageRootValue.create_calls == []
    assert not result_manifest_path_in(first._base_dir).exists()
    assert not result_manifest_path_in(second._base_dir).exists()


def test_worker_load_existing_reports_missing_dependency(tmp_path: Path) -> None:
    ObjectIdStorageRootValue.storage_root_override = tmp_path / "data"
    missing = ObjectIdStorageRootValue(key=13)

    with (
        worker_execution_context(
            lease_id="lease-1",
        ),
        pytest.raises(_DependencyNotReady) as exc_info,
    ):
        missing.load_existing()

    exc = exc_info.value
    assert exc.call_kind == "load_existing"
    assert exc.dependencies == (missing,)


def test_worker_top_level_load_existing_reports_all_missing_dependencies(
    tmp_path: Path,
) -> None:
    ObjectIdStorageRootValue.storage_root_override = tmp_path / "data"
    missing_first = ObjectIdStorageRootValue(key=21)
    ready = ObjectIdStorageRootValue(key=22)
    missing_second = ObjectIdStorageRootValue(key=23)

    assert ready.create() == "object-id:22"

    with (
        worker_execution_context(
            lease_id="lease-1",
        ),
        pytest.raises(_DependencyNotReady) as exc_info,
    ):
        furu.load_existing([missing_first, ready, missing_second])

    exc = exc_info.value
    assert exc.call_kind == "load_existing"
    assert exc.dependencies == (missing_first, missing_second)


def test_worker_dependency_not_ready_is_not_caught_as_exception(
    tmp_path: Path,
) -> None:
    ObjectIdStorageRootValue.storage_root_override = tmp_path / "data"
    missing = ObjectIdStorageRootValue(key=14)

    with pytest.raises(_DependencyNotReady):
        with worker_execution_context(
            lease_id="lease-1",
        ):
            try:
                missing.load_existing()
            except Exception as exc:  # pragma: no cover
                raise AssertionError(
                    "ordinary Exception handler caught signal"
                ) from exc


def test_mixed_type_list_follows_documented_grouping_policy() -> None:
    objs = [
        GroupBatchA(key=1),
        CountedSingleValue(key=10),
        GroupBatchA(key=2),
        GroupBatchB(key=20),
        CountedSingleValue(key=11),
        GroupBatchB(key=21),
    ]

    assert _load_or_create(objs) == [
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

    assert _load_or_create(objs) == ["batch:1", "batch:2"]

    for obj, expected in zip(objs, ["batch:1", "batch:2"], strict=True):
        assert result_manifest_path_in(obj._base_dir).exists()
        assert metadata_path_in(obj._base_dir).exists()
        assert run_log_path_in(obj._base_dir).exists()
        assert (
            load_result_bundle(
                result_dir_in(obj._base_dir),
                data_dir=data_dir_in(obj._base_dir),
            )
            == expected
        )


def test_batched_compute_writes_shared_logs_to_every_participant() -> None:
    objs = [LoggedBatchValue(key=1), LoggedBatchValue(key=2)]

    assert _load_or_create(objs) == ["logged-batch:1", "logged-batch:2"]

    for obj in objs:
        log_text = run_log_path_in(obj._base_dir).read_text(encoding="utf-8")
        assert "batched detail for 1,2" in log_text
        for persisted_obj in objs:
            assert (
                f"stored result bundle at {result_dir_in(persisted_obj._base_dir)}"
                in log_text
            )


def test_sequential_group_compute_writes_shared_logs_to_every_participant() -> None:
    objs = [LoggedSingleValue(key=1), LoggedSingleValue(key=2)]

    assert _load_or_create(objs) == ["logged-single:1", "logged-single:2"]

    for obj in objs:
        log_text = run_log_path_in(obj._base_dir).read_text(encoding="utf-8")
        assert "single detail for 1" in log_text
        assert "single detail for 2" in log_text
        for persisted_obj in objs:
            assert (
                f"stored result bundle at {result_dir_in(persisted_obj._base_dir)}"
                in log_text
            )


def test_batched_failure_writes_error_details_to_run_log_for_every_participant() -> (
    None
):
    objs = [FailingBatchValue(key=1), FailingBatchValue(key=2)]

    with pytest.raises(RuntimeError, match="failed batch"):
        _load_or_create(objs)

    for obj in objs:
        log_text = run_log_path_in(obj._base_dir).read_text(encoding="utf-8")
        assert "create failed" in log_text
        assert "failed batch for [1, 2]" in log_text
        assert "furu-local-debug-value-should-not-leak" not in log_text
        assert list(obj._base_dir.glob("error-*.log")) == []


def test_create_failure_run_log_includes_user_create_call_stack() -> None:
    obj = FailingSingleValue(key=1)

    def _main() -> None:
        obj.create()

    with pytest.raises(RuntimeError, match="failed single"):
        _main()

    log_text = run_log_path_in(obj._base_dir).read_text(encoding="utf-8")
    assert "Traceback (most recent call last):" in log_text
    assert "failed single for 1" in log_text
    assert "Stack (most recent call last):" in log_text
    assert "in _main" in log_text
    assert "obj.create()" in log_text


def test_base_exception_does_not_log_as_load_failure() -> None:
    obj = InterruptingValue(key=1)

    with pytest.raises(KeyboardInterrupt):
        obj.create()

    log_text = run_log_path_in(obj._base_dir).read_text(encoding="utf-8")
    assert "create failed" not in log_text


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
        _load_or_create(objs)

    assert result_manifest_path_in(objs[0]._base_dir).exists()
    assert not result_manifest_path_in(objs[1]._base_dir).exists()


def test_create_publicly_loads_or_computes_result() -> None:
    obj = CountedSingleValue(key=99)

    assert obj.create() == "single:99"
    assert obj.create() == "single:99"
    assert CountedSingleValue.create_calls == [99]


def test_create_batched_cannot_be_called_directly() -> None:
    objs = [BatchOnlyValue(key=1), BatchOnlyValue(key=2)]
    with pytest.raises(RuntimeError, match="must not be called directly"):
        BatchOnlyValue.create_batched(objs)
