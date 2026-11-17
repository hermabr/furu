import logging
import re
import types
from contextlib import contextmanager
from dataclasses import (
    FrozenInstanceError,
    dataclass,
    fields,
    is_dataclass,
    replace,
)
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Literal, cast
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ConfigDict
from runs import ends_of, fixed_run, log_of, run_of

import furu
import furu.execution.load_or_create as execution_module
from furu import Metadata, Spec, Throttle, Worker
from furu.config import get_config
from furu.dependencies import (
    _DependencyNotReady,
    collect_declared_refs,
    under_creation,
)
from furu.execution.load_or_create import _load_or_create
from furu.logging import _execution_log
from furu.metadata import ArtifactSpec
from furu.result.bundle import _save_result_bundle, load_result_bundle
from furu.serializer.artifact import _from_json, to_json
from furu.storage._layout import (
    data_dir_in,
    events_path_in,
    lock_path_in,
    log_path_in,
    result_dir_in,
    result_manifest_path_in,
)

type SOME_TYPE = Literal["a", "b"] | int


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


class ThrottleIdentityNode(Spec[str]):
    name: str
    throttle = Throttle(max_running=1)

    def create(self) -> str:
        return self.name


class WeightedNode(Node):
    weight: float

    def create(self) -> str:
        return f"WNode({self.name}:{self.weight})"


class CustomStorageNode(Node):
    def metadata(self) -> Metadata:
        return Metadata(storage=Path("custom/data/location"))


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


class DoneProbe(Spec[bool]):
    def create(self) -> bool:
        return self.done


class DirectoryPeekingValue(Spec[str]):
    def create(self) -> str:
        assert self.directory.data == self._attempt_dir / "data"
        with pytest.raises(RuntimeError, match=r"only available inside"):
            _ = Node(name="other").directory
        return "ok"


class BatchedDirectoryValue(Spec[str]):
    key: int

    def batch_key(self, worker: furu.Worker) -> tuple[None, int]:
        return (None, 1024)

    @furu.batched(batch_key)
    def create(objs: list["BatchedDirectoryValue"]) -> list[str]:
        for obj in objs:
            assert obj.directory.data == obj._attempt_dir / "data"
        return [f"dir:{obj.key}" for obj in objs]


class ScratchWritingValue(Spec[str]):
    name: str

    def create(self) -> str:
        payload_path = self.directory.scratch / "download.csv"
        if not payload_path.exists():
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
        namespace["create"] = lambda self: cls._furu_create_hook(self)  # ty: ignore[call-non-callable]

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


class UsesPath(Spec[str]):
    path: Path

    def create(self) -> str:
        return str(self.path)


class UsesDatetime(Spec[str]):
    dt: datetime

    def create(self) -> str:
        return self.dt.isoformat()


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


class LoggedLeaf(Spec[str]):
    name: str

    def create(self) -> str:
        self.logger.info("leaf detail for %s", self.name)
        return f"leaf:{self.name}"


class StdlibLoggingLeaf(Spec[str]):
    name: str

    def create(self) -> str:
        logging.getLogger("some_library").warning("stdlib warning for %s", self.name)
        return f"stdlib:{self.name}"


class LoggedParent(Spec[dict[str, str]]):
    child_name: str

    def create(self) -> dict[str, str]:
        self.logger.info("parent before child")
        child_result = LoggedLeaf(name=self.child_name).create()
        self.logger.info("parent after child")
        return {"child": child_result}


class PydanticSubclass(BaseModel):
    model_config = ConfigDict(frozen=True)
    field1: int


class PydanticFields(Spec[None]):
    pydantic_obj: PydanticSubclass

    def create(self) -> None:
        return None


GROUP_EXECUTION_EVENTS: list[tuple[str, tuple[int, ...]]] = []


class CountedSingleValue(Spec[str]):
    key: int
    create_calls: ClassVar[list[int]] = []

    def create(self) -> str:
        type(self).create_calls.append(self.key)
        GROUP_EXECUTION_EVENTS.append(("single", (self.key,)))
        return f"single:{self.key}"


class NoWorkerNode(Spec[str]):
    name: str
    storage_override: ClassVar[Path] = Path("unsatisfiable-root")

    def metadata(self) -> Metadata:
        return Metadata(storage=type(self).storage_override)

    def runs_on(self, worker: Worker) -> bool:
        return False

    def create(self) -> str:
        return self.name


class ObjectIdStorageValue(Spec[str]):
    key: int
    storage_override: ClassVar[Path] = Path("object-id-root")
    create_calls: ClassVar[list[int]] = []

    def metadata(self) -> Metadata:
        return Metadata(storage=type(self).storage_override)

    def create(self) -> str:
        type(self).create_calls.append(self.key)
        return f"object-id:{self.key}"


class NoCreateHookValue(Spec[str]):
    key: int


class BatchOnlyValue(furu.Spec[str]):
    key: int
    batch_calls: ClassVar[list[tuple[int, ...]]] = []

    def batch_key(self, worker: furu.Worker) -> tuple[None, int]:
        return (None, 1024)

    @furu.batched(batch_key)
    def create(objs: list["BatchOnlyValue"]) -> list[str]:
        keys = tuple(obj.key for obj in objs)
        BatchOnlyValue.batch_calls.append(keys)
        return [f"batch:{obj.key}" for obj in objs]


class InheritedHookBatchValue(BatchOnlyValue):
    pass


class KeyedBatchValue(furu.Spec[str]):
    key: int
    group: str
    cap: int = 1024
    batch_calls: ClassVar[list[tuple[str, tuple[int, ...]]]] = []

    def batch_key(self, worker: furu.Worker) -> tuple[str, int]:
        return (self.group, self.cap)

    @furu.batched(batch_key)
    def create(objs: list["KeyedBatchValue"]) -> list[str]:
        KeyedBatchValue.batch_calls.append(
            (objs[0].group, tuple(obj.key for obj in objs))
        )
        return [f"keyed:{obj.group}:{obj.key}" for obj in objs]


class GroupBatchA(furu.Spec[str]):
    key: int
    batch_calls: ClassVar[list[tuple[int, ...]]] = []

    def batch_key(self, worker: furu.Worker) -> tuple[None, int]:
        return (None, 1024)

    @furu.batched(batch_key)
    def create(objs: list["GroupBatchA"]) -> list[str]:
        keys = tuple(obj.key for obj in objs)
        GroupBatchA.batch_calls.append(keys)
        GROUP_EXECUTION_EVENTS.append(("batch_a", keys))
        return [f"group-a:{obj.key}" for obj in objs]


class GroupBatchB(furu.Spec[str]):
    key: int
    batch_calls: ClassVar[list[tuple[int, ...]]] = []

    def batch_key(self, worker: furu.Worker) -> tuple[None, int]:
        return (None, 1024)

    @furu.batched(batch_key)
    def create(objs: list["GroupBatchB"]) -> list[str]:
        keys = tuple(obj.key for obj in objs)
        GroupBatchB.batch_calls.append(keys)
        GROUP_EXECUTION_EVENTS.append(("batch_b", keys))
        return [f"group-b:{obj.key}" for obj in objs]


class LoggedBatchValue(furu.Spec[str]):
    key: int

    def batch_key(self, worker: furu.Worker) -> tuple[None, int]:
        return (None, 1024)

    @furu.batched(batch_key)
    def create(objs: list["LoggedBatchValue"]) -> list[str]:
        keys = ",".join(str(obj.key) for obj in objs)
        objs[0].logger.info("batched detail for %s", keys)
        return [f"logged-batch:{obj.key}" for obj in objs]


class LoggedSingleValue(Spec[str]):
    key: int

    def create(self) -> str:
        self.logger.info("single detail for %s", self.key)
        return f"logged-single:{self.key}"


class FailingBatchValue(furu.Spec[str]):
    key: int

    def batch_key(self, worker: furu.Worker) -> tuple[None, int]:
        return (None, 1024)

    @furu.batched(batch_key)
    def create(objs: list["FailingBatchValue"]) -> list[str]:
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


class PartialBatchValue(furu.Spec[str]):
    key: int

    def batch_key(self, worker: furu.Worker) -> tuple[None, int]:
        return (None, 1024)

    @furu.batched(batch_key)
    def create(objs: list["PartialBatchValue"]) -> list[str]:
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
                events_path_in(self._attempt_dir).exists(),
                sibling.done,
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


class TreeDependencyParent(Spec[str]):
    names: tuple[str, ...]

    @furu.dependency
    def by_name(self) -> dict[str, tuple[Node, ...] | None]:
        return {name: (Node(name=name), Node(name="shared")) for name in self.names} | {
            "none": None
        }

    def create(self) -> str:
        return ""


class CountingDependencyParent(Spec[str]):
    name: str
    calls: ClassVar[int] = 0

    @furu.dependency
    def child(self) -> Node:
        type(self).calls += 1
        return Node(name=f"{self.name}-{type(self).calls}")

    def create(self) -> str:
        return self.child.create()


class LoadsDeclaredDependencyParent(Spec[str]):
    child: Node

    def create(self) -> str:
        return self.child.load_existing()


class FuruBoundaryParent(Spec[str]):
    child: NodePair

    def create(self) -> str:
        return self.child.node1.create()


class BatchDependencyParent(furu.Spec[str]):
    key: int
    eager: Node

    def batch_key(self, worker: furu.Worker) -> tuple[None, int]:
        return (None, 1024)

    @furu.batched(batch_key)
    def create(objs: list["BatchDependencyParent"]) -> list[str]:
        eager_values = [obj.eager.create() for obj in objs]
        lazy_value = Node(name="shared-lazy").create()
        return [f"{value}:{lazy_value}" for value in eager_values]


@pytest.fixture(autouse=True)
def _reset_batch_trackers() -> None:
    CountedSingleValue.create_calls.clear()
    ObjectIdStorageValue.storage_override = Path("object-id-root")
    ObjectIdStorageValue.create_calls.clear()
    BatchOnlyValue.batch_calls.clear()
    KeyedBatchValue.batch_calls.clear()
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
            obj.a = 3  # ty: ignore[invalid-assignment]
        with pytest.raises(TypeError):
            cls(1, 2)  # ty: ignore[missing-argument,too-many-positional-arguments]
        with pytest.raises(FrozenInstanceError):
            obj.a = 3  # ty: ignore[invalid-assignment]


def test_reserved_field_name_raises_at_class_creation():
    with pytest.raises(TypeError) as excinfo:

        class DoneField(Spec[int]):
            done: str  # ty: ignore[override-of-final-method,invalid-property-type-override]

            def create(self) -> int:
                return 0

    message = str(excinfo.value)
    assert "DoneField" in message
    assert "['done']" in message
    for name in (
        "create",
        "metadata",
        "done",
        "directory",
        "load_existing",
        "delete",
        "migrations",
        "provenance",
        "throttle",
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


def test_hashes_and_data_dir():
    assert (
        NodePair(
            name="x", node1=Node(name="y"), node2=WeightedNode(name="z", weight=1)
        ).object_id
        == "test_core.NodePair:06b20558c682cb23a593:0213f99a818a40d63cf9"
    )

    assert (
        NodePair(
            name="x", node1=Node(name="y"), node2=WeightedNode(name="z", weight=1)
        )._artifact_hash
        == "0213f99a818a40d63cf9"
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
        )._class_part
        == "06b20558c682cb23a593"
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
        )._class_part
        != B_priv(
            a=A(x=1, z="123", w=[6, 7]),
            y={"hey": 123, True: 1},
            t=("123", 12),
            _h=1,
        )._class_part
    )

    def qualname_alias(cls: type[Spec[object]], *, ret_typ: type) -> type[Spec[object]]:
        namespace: dict[str, object] = {"__module__": cls.__module__}
        namespace["create"] = lambda self: cls._furu_create_hook(self)  # ty: ignore[call-non-callable]
        alias = type(ret_typ.__qualname__, (cls,), namespace)
        alias.__qualname__ = ret_typ.__qualname__
        return alias

    B_priv_as_B = cast(type[B_priv], qualname_alias(B_priv, ret_typ=B))
    assert (
        B(
            a=A(x=1, z="123", w=[6, 7]), y={"ney": 123, True: 1}, t=("123", 12)
        )._class_part
        != B_priv_as_B(
            a=A(x=1, z="123", w=[6, 7]),
            y={"hey": 123, "ney": 1},
            t=("123", 12),
            _h=1,
        )._class_part
    )


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


def test_datetime_field_round_trips_from_json_artifact():
    naive_obj = UsesDatetime(dt=datetime(2026, 7, 18, 12, 30, 45, 123456))
    aware_obj = UsesDatetime(dt=datetime(2026, 7, 18, 12, 30, tzinfo=UTC))

    assert naive_obj._artifact_data == {
        "|kind": "instance",
        "|class": "test_core.UsesDatetime",
        "|fields": {
            "dt": {"|kind": "datetime", "|value": "2026-07-18T12:30:45.123456"}
        },
    }

    for obj in (naive_obj, aware_obj):
        loaded = cast(UsesDatetime, _from_json(obj._artifact_data))
        assert loaded == obj
        assert isinstance(loaded.dt, datetime)
        assert loaded.dt.tzinfo == obj.dt.tzinfo
        assert loaded.object_id == obj.object_id


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
        fully_qualified_name=obj._stored_name,
        stored_name=obj._stored_name,
        artifact_data=obj._artifact_data,
        artifact_hash=obj._artifact_hash,
        class_part=obj._class_part,
    )

    loaded = NodePair.from_artifact(artifact)

    assert artifact.object_id == obj.object_id
    assert loaded == obj
    assert isinstance(loaded, NodePair)
    assert loaded._class_dir == obj._class_dir


def test_specs_held_in_fields_are_built_before_the_create_hook() -> None:
    first = Node(name="nested")
    second = WeightedNode(name="weighted", weight=2)
    parent = NestedDependencyParent(bundle=DependencyBundle(first=first, second=second))

    assert collect_declared_refs(parent) == (first, second)
    assert parent.create() == "Node(nested)"
    assert second.done


def test_declared_dependencies_are_created_before_the_create_hook(
    tmp_path: Path,
) -> None:
    parent = LoadsDeclaredDependencyParent(child=Node(name="declared"))

    log_path = tmp_path / "create.log"
    with _execution_log(log_path):
        assert parent.create() == "Node(declared)"

    log_text = log_path.read_text(encoding="utf-8")
    # Every spec actually built is announced, dependencies included.
    for obj in (parent.child, parent):
        assert f" I creating {obj._log_label} → " in log_text
        assert f" I finished {obj._log_label} ok · " in log_text


def test_a_dependency_is_planned() -> None:
    parent = ComputedDependencyParent(name="computed")

    assert collect_declared_refs(parent) == (parent.child,)
    assert parent.create() == "Node(computed)"


def test_a_dependency_is_computed_once_when_first_read() -> None:
    CountingDependencyParent.calls = 0
    parent = CountingDependencyParent(name="counting")

    assert parent == CountingDependencyParent(name="counting")
    assert repr(parent) == "CountingDependencyParent(name='counting')"
    assert CountingDependencyParent.calls == 0

    assert parent.object_id
    assert CountingDependencyParent.calls == 1

    assert parent.child is parent.child
    assert collect_declared_refs(parent) == (parent.child,)
    assert CountingDependencyParent.calls == 1


def test_a_dependency_is_stored_and_hashed_as_a_field() -> None:
    parent = ComputedDependencyParent(name="x")

    assert parent._artifact_data["|fields"] == {
        "name": "x",
        "child": Node(name="x")._artifact_data,
    }
    with pytest.raises(TypeError, match="unexpected keyword argument 'child'"):
        ComputedDependencyParent(name="x", child=Node(name="y"))  # ty: ignore[unknown-argument]


def test_a_dependency_returning_a_tree_plans_every_spec_in_it() -> None:
    parent = TreeDependencyParent(names=("b", "a"))

    refs = cast(tuple[Node, ...], collect_declared_refs(parent))
    assert {ref.name for ref in refs} == {"a", "b", "shared"}
    assert cast(Any, parent._artifact_data)["|fields"]["by_name"]["none"] is None


def test_a_spec_with_a_dependency_round_trips_through_its_artifact() -> None:
    parent = ComputedDependencyParent(name="x")

    assert Spec.from_artifact(ArtifactSpec.from_furu(parent)) == parent


def test_loading_an_artifact_whose_dependency_is_now_derived_differently_fails() -> (
    None
):
    # Its child is named after the number of calls, so a rebuilt spec gets another.
    stored = ArtifactSpec.from_furu(CountingDependencyParent(name="x"))

    with pytest.raises(ValueError, match="@furu.dependency"):
        Spec.from_artifact(stored)


def test_a_dependency_without_a_return_annotation_is_rejected() -> None:
    with pytest.raises(TypeError, match="needs a return annotation"):

        class Untyped(Spec[str]):
            @furu.dependency
            def child(self):
                return Node(name="x")

            def create(self) -> str:
                return ""


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


@dataclass(frozen=True)
class _TreePair:
    left: Node
    right: list[Node]


def test_create_and_load_existing_map_specs_through_pytrees() -> None:
    a, b = Node(name="tree-a"), Node(name="tree-b")
    tree = {"pair": _TreePair(left=a, right=[b, a]), "rest": (b, 3, "x")}
    expected = {
        "pair": {"left": "Node(tree-a)", "right": ["Node(tree-b)", "Node(tree-a)"]},
        "rest": ("Node(tree-b)", 3, "x"),
    }

    assert furu.create(tree) == expected
    assert furu.load_existing(tree) == expected
    assert furu.load_existing(a) == "Node(tree-a)"
    assert furu.create(_TreePair(left=a, right=[])) == {
        "left": "Node(tree-a)",
        "right": [],
    }


class _ParentIgnoringChild(Spec[str]):
    child: Node

    def create(self) -> str:
        return "parent"


def _forbid_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    def load_stored_result(obj: Spec, result_dir: Path) -> object:
        raise AssertionError(f"unexpected load of {obj._log_label}")

    monkeypatch.setattr(execution_module, "load_stored_result", load_stored_result)


def test_build_stores_results_without_loading_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached, fresh = Node(name="build-cached"), Node(name="build-fresh")
    furu.create(cached)
    _forbid_loading(monkeypatch)

    furu.build({"cached": cached, "fresh": [fresh]})

    assert fresh.done


def test_create_does_not_load_cached_declared_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _ParentIgnoringChild(child=Node(name="cached-child"))
    furu.create(parent.child)
    _forbid_loading(monkeypatch)

    assert furu.create(parent) == "parent"


def test_top_level_load_existing_accepts_list_and_logs_once(tmp_path: Path) -> None:
    nodes = [Node(name="load-a"), Node(name="load-b")]

    assert [node.create() for node in nodes] == ["Node(load-a)", "Node(load-b)"]

    log_path = tmp_path / "load-existing.log"
    with _execution_log(log_path):
        assert furu.load_existing(nodes) == ["Node(load-a)", "Node(load-b)"]

    info_lines = [
        line
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if " I " in line
    ]
    assert len(info_lines) == 1
    assert info_lines[0].endswith(
        f" I loaded 2 furu objects including {nodes[0]._log_label}"
    )


def test_furu_objects_block_nested_eager_traversal() -> None:
    node1 = Node(name="inner")
    node2 = WeightedNode(name="other", weight=3)
    child = NodePair(node1=node1, node2=node2, name="pair")
    parent = FuruBoundaryParent(child=child)

    assert parent.create() == "Node(inner)"

    assert collect_declared_refs(parent) == (child,)


def test_a_batched_create_loads_specs_from_fields_and_ones_it_asks_for() -> None:
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


def test_furu_from_artifact_infers_furu_object_type():
    obj = NodePair(
        name="x",
        node1=Node(name="y"),
        node2=WeightedNode(name="z", weight=1),
    )
    obj.create()
    artifact = ArtifactSpec(
        fully_qualified_name=obj._stored_name,
        stored_name=obj._stored_name,
        artifact_data=obj._artifact_data,
        artifact_hash=obj._artifact_hash,
        class_part=obj._class_part,
    )

    loaded = Spec.from_artifact(artifact)

    assert artifact.object_id == obj.object_id
    assert loaded == obj
    assert isinstance(loaded, NodePair)


def test_furu_from_artifact_accepts_artifact_spec():
    obj = Node(name="x")
    artifact = ArtifactSpec(
        fully_qualified_name=obj._stored_name,
        stored_name=obj._stored_name,
        artifact_data=obj._artifact_data,
        artifact_hash=obj._artifact_hash,
        class_part=obj._class_part,
    )

    loaded = Node.from_artifact(artifact)

    assert artifact.object_id == obj.object_id
    assert loaded == obj
    assert isinstance(loaded, Node)


def test_furu_from_artifact_type_mismatch_names_expected_and_loaded_type():
    obj = WeightedNode(name="x", weight=1)
    artifact = ArtifactSpec(
        fully_qualified_name=obj._stored_name,
        stored_name=obj._stored_name,
        artifact_data=obj._artifact_data,
        artifact_hash=obj._artifact_hash,
        class_part=obj._class_part,
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
        fully_qualified_name=obj._stored_name,
        stored_name=obj._stored_name,
        artifact_data=obj._artifact_data,
        artifact_hash=bad_hash,
        class_part=obj._class_part,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Artifact hash did not match loaded object: "
            + f"artifact={bad_hash[:5]}, loaded={obj._artifact_hash[:5]}"
        ),
    ):
        Node.from_artifact(artifact)


def test_furu_from_artifact_rejects_artifact_spec_class_part_mismatch():
    obj = Node(name="x")
    bad_class_part = "wrong-class-part"
    artifact = ArtifactSpec(
        fully_qualified_name=obj._stored_name,
        stored_name=obj._stored_name,
        artifact_data=obj._artifact_data,
        artifact_hash=obj._artifact_hash,
        class_part=bad_class_part,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Artifact class part did not match loaded object: "
            + f"artifact={bad_class_part[:5]}, "
            + f"loaded={obj._class_part[:5]}"
        ),
    ):
        Node.from_artifact(artifact)


def test_data_dir():
    node_pair = NodePair(
        name="x", node1=Node(name="y"), node2=WeightedNode(name="z", weight=1)
    )
    class_dir = (
        get_config().directories.anchored().objects
        / "test_core.NodePair"
        / "06b20558c682cb23a593"
    )
    assert node_pair._class_dir == class_dir
    assert node_pair._attempt_dir == class_dir / "attempts" / "0213f99a818a40d63cf9"
    assert fixed_run(node_pair) == (
        class_dir / "fixed" / "runs" / "0213f99a818a40d63cf9"
    )


def test_metadata_defaults_to_project_config_storage_and_any_worker():
    node = Node(name="x")

    assert node.metadata() == Metadata()
    assert node.runs_on(Worker())
    assert node._metadata.storage == get_config().directories.anchored().objects


def test_worker_here_counts_cuda_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,3")
    assert Worker.here().gpus == 2
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert Worker.here().gpus == 0
    assert Worker.here().cpus >= 1


def test_throttle_defaults_to_none():
    assert Node(name="x").throttle is None


def test_throttle_can_be_overridden_as_class_attribute():
    class LimitedNode(Spec[str]):
        name: str
        throttle = Throttle(max_running=5)

        def create(self) -> str:
            return self.name

    assert LimitedNode(name="x").throttle == Throttle(max_running=5)
    assert "throttle" not in {field.name for field in fields(LimitedNode)}


def test_throttle_does_not_affect_object_identity():
    original_throttle = ThrottleIdentityNode.throttle
    before = ThrottleIdentityNode(name="x")
    before_class_part = before._class_part
    before_artifact_hash = before._artifact_hash
    before_object_id = before.object_id

    try:
        ThrottleIdentityNode.throttle = Throttle(max_running=5)
        after = ThrottleIdentityNode(name="x")
    finally:
        ThrottleIdentityNode.throttle = original_throttle

    assert after._class_part == before_class_part
    assert after._artifact_hash == before_artifact_hash
    assert after.object_id == before_object_id


def test_runs_on_can_depend_on_fields():
    class HeavyNode(Spec[str]):
        name: str
        big: bool

        def runs_on(self, worker: Worker) -> bool:
            return worker.gpus in (1, 8) if self.big else worker.gpus == 0

        def create(self) -> str:
            return self.name

    big, small = HeavyNode(name="x", big=True), HeavyNode(name="x", big=False)
    assert big.runs_on(Worker(gpus=8)) and not big.runs_on(Worker(gpus=2))
    assert small.runs_on(Worker()) and not small.runs_on(Worker(gpus=1))


def test_metadata_storage_overrides_the_storage_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    node = CustomStorageNode(name="x")

    assert node._class_dir == (
        Path("custom/data/location") / "test_core.CustomStorageNode" / node._class_part
    )


def test_in_process_create_ignores_runs_on(
    tmp_path: Path,
) -> None:
    NoWorkerNode.storage_override = tmp_path / "store"
    node = NoWorkerNode(name="x")

    assert not node.done
    node.create()
    assert run_of(node).is_relative_to(tmp_path / "store")
    assert NoWorkerNode(name="x").load_existing() == "x"


def test_create_object_and_exists():
    node_pair = NodePair(
        name="x", node1=Node(name="y"), node2=WeightedNode(name="z", weight=1)
    )
    assert not node_pair.done
    for _i in range(3):
        assert node_pair.create() == {
            "node1": "Node(y)",
            "node2": "WNode(z:1)",
            "name": "x",
        }
    assert node_pair.done
    assert not replace(node_pair, name="y").done


def test_data_dir_is_user_data_subdirectory() -> None:
    obj = UserDataWritingValue(name="payload")

    assert obj.create() == "payload"

    run = run_of(obj)
    assert (data_dir_in(run) / "payload.txt").read_text(encoding="utf-8") == "payload"
    assert sorted(path.name for path in run.iterdir()) == [
        "data",
        "events.jsonl",
        "log",
        "provenance.json",
        "result",
    ]
    assert not obj._attempt_dir.exists()


def test_unused_data_dir_is_not_created_by_create() -> None:
    node = Node(name="unused-data")

    assert node.create() == "Node(unused-data)"

    assert not data_dir_in(run_of(node)).exists()


def test_directory_is_only_available_inside_own_create() -> None:
    node = Node(name="manual-data")

    with pytest.raises(RuntimeError, match=r"only available inside"):
        _ = node.directory

    # Outside access must not fabricate a working folder.
    assert not node._attempt_dir.exists()

    node.create()

    with pytest.raises(RuntimeError, match=r"only available inside"):
        _ = node.directory


def test_directory_of_another_spec_raises_inside_create() -> None:
    assert DirectoryPeekingValue().create() == "ok"


def test_batched_create_can_access_every_group_members_directory() -> None:
    objs = [BatchedDirectoryValue(key=1), BatchedDirectoryValue(key=2)]

    assert _load_or_create(objs) == ["dir:1", "dir:2"]


def test_scratch_survives_retries_and_is_deleted_after_create() -> None:
    obj = ScratchWritingValue(name="payload")
    scratch_dir = obj._attempt_dir / "scratch"
    scratch_dir.mkdir(parents=True)
    (scratch_dir / "download.csv").write_text("from-previous-attempt", encoding="utf-8")

    assert obj.create() == "from-previous-attempt"

    assert not scratch_dir.exists()
    assert not (run_of(obj) / "scratch").exists()


def test_a_spec_is_done_only_once_its_run_is_published() -> None:
    probe = DoneProbe()
    failing = FailingSingleValue(key=1)

    assert probe.create() is False  # while it is being computed
    assert probe.done
    with pytest.raises(RuntimeError, match="failed single for 1"):
        failing.create()
    assert failing._attempt_dir.exists()
    assert not failing.done


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
    run = run_of(node)
    assert node.delete(mode="force")
    assert not run.exists()
    assert not node.done
    assert node.create() == "Node(x)"


def test_delete_prompt_cancel() -> None:
    node = Node(name="x")

    assert node.create() == "Node(x)"
    with patch("builtins.input", return_value="n"):
        assert not node.delete()
    assert node.done


def test_delete_removes_the_working_folder_of_a_stopped_attempt() -> None:
    failing = FailingSingleValue(key=1)
    with pytest.raises(RuntimeError, match="failed single for 1"):
        failing.create()

    assert failing.delete(mode="force")

    assert not failing._attempt_dir.exists()
    assert not failing.delete(mode="force")


def test_delete_returns_false_when_missing() -> None:
    assert not Node(name="x").delete(mode="force")


def test_the_log_is_published_with_the_run_and_holds_only_what_create_wrote() -> None:
    node = LoggedLeaf(name="x")

    assert node.create() == "leaf:x"

    (line,) = log_path_in(run_of(node)).read_text(encoding="utf-8").splitlines()
    assert line.endswith(" I leaf detail for x  [tests/test_core.py:" + line[-4:])


def test_in_process_log_captures_stdlib_logging_without_touching_root() -> None:
    node = StdlibLoggingLeaf(name="x")
    root_level = logging.getLogger().level

    assert node.create() == "stdlib:x"

    assert " W some_library stdlib warning for x  [" in log_of(node)
    assert logging.getLogger().level == root_level


def test_nested_create_blocks_the_parent_and_logs_the_child_on_its_own(
    tmp_path: Path,
) -> None:
    child = LoggedLeaf(name="child")
    parent = LoggedParent(child_name="child")

    log_path = tmp_path / "create.log"
    with _execution_log(log_path):
        assert parent.create() == {"child": "leaf:child"}

    parent_log = log_of(parent)
    child_log = log_of(child)
    top_log = log_path.read_text(encoding="utf-8")

    # The first attempt blocks, the second reruns from scratch and finishes.
    assert parent_log.count("parent before child") == 2
    assert parent_log.count("parent after child") == 1
    assert ends_of(parent) == [
        {"outcome": "blocked", "waiting_on": [child.object_id]},
        {"outcome": "ok"},
    ]
    assert f"creating {child._log_label}" not in parent_log
    assert "leaf detail for child" not in parent_log
    assert "leaf detail for child" in child_log
    # What furu has to say goes to the terminal, not into a run's log.
    assert f"cached {child._log_label}" not in parent_log
    assert f"cached {child._log_label}" in top_log
    assert f"blocked {parent._log_label} · 1 missing dependency" in top_log
    assert f"finished {child._log_label} ok" in top_log


def test_cached_create_logs_only_cache_hit_info(
    tmp_path: Path,
) -> None:
    ObjectIdStorageValue.storage_override = tmp_path / "objects"
    obj = ObjectIdStorageValue(key=1)

    assert obj.create() == "object-id:1"

    log_path = tmp_path / "cached-create.log"
    with _execution_log(log_path):
        assert obj.create() == "object-id:1"

    log_text = log_path.read_text(encoding="utf-8")
    assert "building" not in log_text
    assert "creating " not in log_text
    assert "finished " not in log_text
    (line,) = [line for line in log_text.splitlines() if " I " in line]
    assert line.endswith(f" I cached {obj._log_label}")


def test_small_cache_summary_logs_labels_for_cached_and_missing_items(
    tmp_path: Path,
) -> None:
    ObjectIdStorageValue.storage_override = tmp_path / "objects"
    cached = ObjectIdStorageValue(key=1)
    missing = ObjectIdStorageValue(key=2)

    assert cached.create() == "object-id:1"

    log_path = tmp_path / "mixed-create.log"
    with _execution_log(log_path):
        assert _load_or_create([cached, missing]) == ["object-id:1", "object-id:2"]

    assert (
        f"building {missing._log_label}, cached {cached._log_label}"
        in log_path.read_text(encoding="utf-8")
    )


def test_create_hook_flavor_validation() -> None:
    with pytest.raises(TypeError, match="needs a batch key function"):

        class BareBatchedDecorator(Spec[int]):
            @furu.batched  # ty: ignore[invalid-argument-type]
            def create(  # ty: ignore[invalid-attribute-override]
                objs: list["BareBatchedDecorator"],
            ) -> list[int]:
                return [1 for _ in objs]

    with pytest.raises(TypeError, match="needs a batch key function"):
        furu.batched("not-a-function")  # ty: ignore[invalid-argument-type]

    with pytest.raises(TypeError, match=r"can only decorate create\(\)"):

        class BatchedNonCreateMethod(Spec[int]):
            @furu.batched(lambda _, __: (None, 1))
            def run(objs: list["BatchedNonCreateMethod"]) -> list[int]:
                return [1 for _ in objs]

    # Abstract intermediate bases defining no hook stay legal.
    class NoCreateHook(Spec[int]):
        value: int


def test_no_create_hook_loads_cached_result() -> None:
    obj = NoCreateHookValue(key=1)
    _save_result_bundle(
        ["cached:1"],
        result_dir_in(fixed_run(obj)),
        declared_type=str,
        result_codecs=(),
        data_dir=data_dir_in(fixed_run(obj)),
        label="NoCreateHookValue",
    )

    assert furu.create(obj) == "cached:1"


def test_no_create_hook_raises_only_for_missing_result() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "NoCreateHookValue cannot create missing results because it does not "
            r"define create\(\)"
        ),
    ):
        furu.create(NoCreateHookValue(key=2))


def test_no_create_hook_uses_post_lock_cache_recheck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obj = NoCreateHookValue(key=3)

    @contextmanager
    def fake_lock(lock_paths: list[Path], **_: object):
        assert lock_paths == [lock_path_in(obj._class_dir, obj._run_name)]
        _save_result_bundle(
            ["cached-after-lock:1"],
            result_dir_in(fixed_run(obj)),
            declared_type=str,
            result_codecs=(),
            data_dir=data_dir_in(fixed_run(obj)),
            label="NoCreateHookValue",
        )
        yield lambda: True

    monkeypatch.setattr(execution_module, "lock", fake_lock)

    assert furu.create(obj) == "cached-after-lock:1"


def test_single_object_on_batch_only_class_uses_create_batched() -> None:
    assert _load_or_create(BatchOnlyValue(key=1)) == "batch:1"
    assert BatchOnlyValue.batch_calls == [(1,)]


def test_instance_access_on_batched_create_runs_a_group_of_one() -> None:
    assert BatchOnlyValue.create is Spec.create
    assert BatchOnlyValue(key=7).create() == "batch:7"
    assert BatchOnlyValue.batch_calls == [(7,)]


def test_batched_create_hook_is_inherited_by_subclasses() -> None:
    objs = [InheritedHookBatchValue(key=1), InheritedHookBatchValue(key=2)]

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


def test_unbatched_specs_run_alone_with_their_own_running_metadata() -> None:
    first = MetadataTimingValue(key=1)
    second = MetadataTimingValue(key=2)
    MetadataTimingValue.siblings_by_key.update({1: first, 2: second})

    assert _load_or_create([first, second]) == ["timed:1", "timed:2"]
    assert MetadataTimingValue.create_events == [
        (1, True, False),
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


def test_batch_key_partitions_batched_create_calls() -> None:
    objs = [
        KeyedBatchValue(key=1, group="x"),
        KeyedBatchValue(key=2, group="y"),
        KeyedBatchValue(key=3, group="x"),
    ]

    assert _load_or_create(objs) == ["keyed:x:1", "keyed:y:2", "keyed:x:3"]
    assert KeyedBatchValue.batch_calls == [("x", (1, 3)), ("y", (2,))]


def test_batch_key_cap_chunks_batched_create_calls() -> None:
    objs = [KeyedBatchValue(key=key, group="x", cap=2) for key in range(5)]

    assert _load_or_create(objs) == [f"keyed:x:{key}" for key in range(5)]
    assert KeyedBatchValue.batch_calls == [
        ("x", (0, 1)),
        ("x", (2, 3)),
        ("x", (4,)),
    ]


class PerGpuBatch(furu.Spec[int]):
    key: int

    @furu.batched(lambda _, worker: (None, worker.gpus))
    def create(objs: list["PerGpuBatch"]) -> list[int]:
        return [len(objs)] * len(objs)


def test_in_process_batch_cap_sees_this_machine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    assert _load_or_create([PerGpuBatch(key=key) for key in range(3)]) == [2, 2, 1]


def test_batch_key_cap_must_be_a_positive_int() -> None:
    with pytest.raises(TypeError, match="cap must be a positive int"):
        _load_or_create([KeyedBatchValue(key=1, group="x", cap=0)])
    with pytest.raises(TypeError, match="cap must be a positive int"):
        _load_or_create([KeyedBatchValue(key=1, group="x", cap=True)])


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
    ObjectIdStorageValue.storage_override = tmp_path / "first"
    first = ObjectIdStorageValue(key=1)
    first_data_dir = data_dir_in(first._attempt_dir)

    ObjectIdStorageValue.storage_override = tmp_path / "second"
    second = ObjectIdStorageValue(key=1)
    second_data_dir = data_dir_in(second._attempt_dir)

    assert first.object_id == second.object_id
    assert first_data_dir != second_data_dir

    assert _load_or_create([first, second]) == ["object-id:1", "object-id:1"]
    assert ObjectIdStorageValue.create_calls == [1]


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
    assert lock_calls == [[lock_path_in(missing._class_dir, missing._run_name)]]


def test_a_worker_that_takes_the_lock_checks_for_a_published_run_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pending = CountedSingleValue(key=5)

    @contextmanager
    def fake_lock(lock_paths: list[Path], **_: object):
        assert lock_paths == [lock_path_in(pending._class_dir, pending._run_name)]
        _save_result_bundle(
            ["single:5"],
            result_dir_in(fixed_run(pending)),
            declared_type=str,
            result_codecs=(),
            data_dir=data_dir_in(fixed_run(pending)),
            label="CountedSingleValue",
        )
        yield lambda: True

    monkeypatch.setattr(execution_module, "lock", fake_lock)

    assert _load_or_create([pending]) == ["single:5"]
    assert CountedSingleValue.create_calls == []
    assert not pending._attempt_dir.exists()


def test_empty_list_returns_empty_list() -> None:
    assert _load_or_create([]) == []


def test_nested_create_loads_cached_result_without_recomputing(
    tmp_path: Path,
) -> None:
    ObjectIdStorageValue.storage_override = tmp_path / "data"
    cached = ObjectIdStorageValue(key=10)

    assert cached.create() == "object-id:10"
    ObjectIdStorageValue.create_calls.clear()

    with under_creation([ObjectIdStorageValue(key=0)]):
        assert cached.create() == "object-id:10"

    assert ObjectIdStorageValue.create_calls == []


def test_nested_create_reports_all_missing_dependencies(
    tmp_path: Path,
) -> None:
    ObjectIdStorageValue.storage_override = tmp_path / "data"
    first = ObjectIdStorageValue(key=11)
    second = ObjectIdStorageValue(key=12)

    parent = ObjectIdStorageValue(key=0)

    with under_creation([parent]), pytest.raises(_DependencyNotReady) as exc_info:
        _load_or_create([first, second])

    assert exc_info.value.dependencies == (first, second)
    assert ObjectIdStorageValue.create_calls == []
    assert not first.done
    assert not second.done


def test_nested_load_existing_and_provenance_raise_missing(tmp_path: Path) -> None:
    ObjectIdStorageValue.storage_override = tmp_path / "data"
    missing = ObjectIdStorageValue(key=13)

    with under_creation([ObjectIdStorageValue(key=0)]):
        with pytest.raises(furu.Missing):
            missing.load_existing()
        with pytest.raises(furu.Missing):
            missing.provenance()
        with pytest.raises(furu.Missing):
            furu.load_existing([missing])


def test_dependency_not_ready_is_not_caught_as_exception(
    tmp_path: Path,
) -> None:
    ObjectIdStorageValue.storage_override = tmp_path / "data"
    missing = ObjectIdStorageValue(key=14)

    with (
        pytest.raises(_DependencyNotReady),
        under_creation([ObjectIdStorageValue(key=0)]),
    ):
        try:
            missing.create()
        except Exception as exc:  # pragma: no cover
            raise AssertionError("ordinary Exception handler caught signal") from exc


def test_mixed_type_list_batches_each_type_when_its_first_member_is_next() -> None:
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
        ("batch_b", (20, 21)),
        ("single", (11,)),
    ]


def test_batched_compute_writes_result_layout_per_object() -> None:
    objs = [BatchOnlyValue(key=1), BatchOnlyValue(key=2)]

    assert _load_or_create(objs) == ["batch:1", "batch:2"]

    for obj, expected in zip(objs, ["batch:1", "batch:2"], strict=True):
        run = run_of(obj)
        assert result_manifest_path_in(run).exists()
        assert ends_of(obj) == [{"outcome": "ok"}]
        assert load_result_bundle(
            result_dir_in(run), [0], data_dir=data_dir_in(run), declared_type=str
        ) == [expected]


def test_a_batched_create_logs_to_its_leads_run() -> None:
    objs = [LoggedBatchValue(key=1), LoggedBatchValue(key=2)]

    assert _load_or_create(objs) == ["logged-batch:1", "logged-batch:2"]

    assert "batched detail for 1,2" in log_of(objs[0])
    assert not log_path_in(run_of(objs[1])).exists()


def test_unbatched_specs_log_to_their_own_runs() -> None:
    objs = [LoggedSingleValue(key=1), LoggedSingleValue(key=2)]

    assert _load_or_create(objs) == ["logged-single:1", "logged-single:2"]

    first_log, second_log = map(log_of, objs)
    assert "single detail for 1" in first_log
    assert "single detail for 2" not in first_log
    assert "single detail for 2" in second_log


def test_a_failed_batch_logs_the_traceback_once_and_ends_every_members_attempt() -> (
    None
):
    objs = [FailingBatchValue(key=1), FailingBatchValue(key=2)]

    with pytest.raises(RuntimeError, match="failed batch"):
        _load_or_create(objs)

    lead_log = log_of(objs[0])
    assert "create failed" in lead_log
    assert "Traceback (most recent call last):" in lead_log
    assert "furu-local-debug-value-should-not-leak" not in lead_log
    for obj in objs:
        assert ends_of(obj) == [
            {"outcome": "failed", "error": "RuntimeError: failed batch for [1, 2]"}
        ]


def test_a_failed_create_logs_the_traceback_once() -> None:
    obj = FailingSingleValue(key=1)

    with pytest.raises(RuntimeError, match="failed single"):
        obj.create()

    lines = log_of(obj).splitlines()
    assert re.match(rf"^\S+ E create failed for {re.escape(obj._log_label)}$", lines[0])
    assert lines[1] == "    Traceback (most recent call last):"
    assert lines.count("    RuntimeError: failed single for 1") == 1
    assert "Stack (most recent call last):" not in "\n".join(lines)


def test_base_exception_does_not_log_as_load_failure() -> None:
    obj = InterruptingValue(key=1)

    with pytest.raises(KeyboardInterrupt):
        obj.create()

    assert "create failed" not in log_of(obj)
    assert ends_of(obj) == [{"outcome": "failed", "error": "KeyboardInterrupt"}]


def test_partial_persistence_leaves_already_written_objects_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objs = [PartialBatchValue(key=1), PartialBatchValue(key=2)]
    real_publish = execution_module._publish
    call_count = 0

    def flaky_publish(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("stop after first store")
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(execution_module, "_publish", flaky_publish)

    with pytest.raises(RuntimeError, match="stop after first store"):
        _load_or_create(objs)

    assert objs[0].done
    assert not objs[1].done


def test_create_publicly_loads_or_computes_result() -> None:
    obj = CountedSingleValue(key=99)

    assert obj.create() == "single:99"
    assert obj.create() == "single:99"
    assert CountedSingleValue.create_calls == [99]


def test_public_create_batches_lists() -> None:
    objs = [BatchOnlyValue(key=1), BatchOnlyValue(key=2)]

    assert furu.create(objs) == ["batch:1", "batch:2"]
    assert BatchOnlyValue.batch_calls == [(1, 2)]
    # A second call is served from cache without invoking the hook again.
    assert furu.create(objs) == ["batch:1", "batch:2"]
    assert BatchOnlyValue.batch_calls == [(1, 2)]
