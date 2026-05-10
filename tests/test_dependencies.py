from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from pydantic import BaseModel, ConfigDict

import furu
from furu import Furu
from furu.dependencies import DependencyMetadata, DependencyRef
from furu.metadata import CompletedMetadata


def _completed_metadata(obj: Furu) -> CompletedMetadata:
    text = obj._metadata_path.read_text()
    assert json.loads(text)["kind"] == "completed"
    return CompletedMetadata.model_validate_json(text)


def _deps(obj: Furu) -> DependencyMetadata:
    return _completed_metadata(obj).dependencies


def _ids(refs: tuple[DependencyRef, ...] | list[DependencyRef]) -> set[str]:
    return {ref.object_id for ref in refs}


class Leaf(Furu[str]):
    name: str

    def _create(self) -> str:
        return f"leaf:{self.name}"


class Parent(Furu[str]):
    leaf: Leaf

    def _create(self) -> str:
        return f"parent({self.leaf.load_or_create()})"


def test_eager_dependency_from_direct_field() -> None:
    leaf = Leaf(name="x")
    parent = Parent(leaf=leaf)

    assert parent.load_or_create() == "parent(leaf:x)"

    deps = _deps(parent)
    assert _ids(deps.eager) == {leaf.object_id}
    assert deps.lazy == ()
    [eager_ref] = deps.eager
    assert eager_ref.via == "field"
    assert eager_ref.path == "leaf"
    assert eager_ref.class_name.endswith("Leaf")


@dataclass(frozen=True)
class Splits:
    train: Leaf
    validation: Leaf


class NestedDataclassParent(Furu[str]):
    splits: Splits

    def _create(self) -> str:
        train = self.splits.train.load_or_create()
        valid = self.splits.validation.load_or_create()
        return f"{train}+{valid}"


def test_eager_dependency_from_nested_dataclass_field() -> None:
    train = Leaf(name="train")
    validation = Leaf(name="valid")
    parent = NestedDataclassParent(splits=Splits(train=train, validation=validation))

    parent.load_or_create()

    deps = _deps(parent)
    assert _ids(deps.eager) == {train.object_id, validation.object_id}
    assert deps.lazy == ()
    paths = sorted(ref.path for ref in deps.eager if ref.path is not None)
    assert paths == ["splits.train", "splits.validation"]


class NestedListParent(Furu[str]):
    leaves: list[Leaf]

    def _create(self) -> str:
        return ",".join(leaf.load_or_create() for leaf in self.leaves)


def test_eager_dependency_from_list_field() -> None:
    a, b = Leaf(name="a"), Leaf(name="b")
    parent = NestedListParent(leaves=[a, b])

    parent.load_or_create()

    deps = _deps(parent)
    assert _ids(deps.eager) == {a.object_id, b.object_id}
    assert deps.lazy == ()


class NestedPydantic(BaseModel):
    model_config = ConfigDict(frozen=True)
    primary: Leaf
    secondary: Leaf


class NestedPydanticParent(Furu[str]):
    pair: NestedPydantic

    def _create(self) -> str:
        return f"{self.pair.primary.load_or_create()}|{self.pair.secondary.load_or_create()}"


def test_eager_dependency_from_pydantic_nested_field() -> None:
    a, b = Leaf(name="p"), Leaf(name="s")
    parent = NestedPydanticParent(pair=NestedPydantic(primary=a, secondary=b))

    parent.load_or_create()

    deps = _deps(parent)
    assert _ids(deps.eager) == {a.object_id, b.object_id}


class FuruSplits(Furu[dict[str, str]]):
    train: Leaf
    validation: Leaf

    def _create(self) -> dict[str, str]:
        return {
            "train": self.train.load_or_create(),
            "validation": self.validation.load_or_create(),
        }


class FuruSplitsParent(Furu[str]):
    splits: FuruSplits

    def _create(self) -> str:
        loaded = self.splits.load_or_create()
        return f"{loaded['train']}+{loaded['validation']}"


def test_furu_field_blocks_eager_traversal() -> None:
    train, validation = Leaf(name="t"), Leaf(name="v")
    splits = FuruSplits(train=train, validation=validation)
    parent = FuruSplitsParent(splits=splits)

    parent.load_or_create()

    parent_deps = _deps(parent)
    assert _ids(parent_deps.eager) == {splits.object_id}
    assert parent_deps.lazy == ()

    splits_deps = _deps(splits)
    assert _ids(splits_deps.eager) == {train.object_id, validation.object_id}


class FuruSplitsReachThrough(Furu[str]):
    splits: FuruSplits

    def _create(self) -> str:
        train = self.splits.train.load_or_create()
        validation = self.splits.validation.load_or_create()
        return f"{train}+{validation}"


def test_reach_through_furu_field_records_inner_as_lazy() -> None:
    train, validation = Leaf(name="t1"), Leaf(name="v1")
    splits = FuruSplits(train=train, validation=validation)
    parent = FuruSplitsReachThrough(splits=splits)

    parent.load_or_create()

    deps = _deps(parent)
    assert _ids(deps.eager) == {splits.object_id}
    assert _ids(deps.lazy) == {train.object_id, validation.object_id}


class Predictions(Furu[str]):
    leaf: Leaf

    def _create(self) -> str:
        return f"pred({self.leaf.load_or_create()})"


class Evaluation(Furu[str]):
    eval_data: Leaf

    @furu.dependency
    def predictions(self) -> Predictions:
        return Predictions(leaf=self.eval_data)

    def _create(self) -> str:
        return f"eval({self.predictions.load_or_create()},{self.eval_data.load_or_create()})"


def test_furu_dependency_descriptor_is_eager() -> None:
    eval_data = Leaf(name="e")
    obj = Evaluation(eval_data=eval_data)

    obj.load_or_create()

    deps = _deps(obj)
    eager_ids = _ids(deps.eager)
    assert eval_data.object_id in eager_ids
    assert obj.predictions.object_id in eager_ids
    assert deps.lazy == ()
    via_for = {ref.object_id: ref.via for ref in deps.eager}
    assert via_for[eval_data.object_id] == "field"
    assert via_for[obj.predictions.object_id] == "dependency"


def test_furu_dependency_caches_per_instance() -> None:
    obj = Evaluation(eval_data=Leaf(name="cache"))
    assert obj.predictions is obj.predictions


class Manifest(Furu[list[str]]):
    seed: int

    def _create(self) -> list[str]:
        return [f"chunk-{self.seed}-{i}" for i in range(3)]


class Chunk(Furu[str]):
    name: str

    def _create(self) -> str:
        return f"chunk:{self.name}"


class DatasetFromManifest(Furu[list[str]]):
    manifest: Manifest

    def _create(self) -> list[str]:
        names = self.manifest.load_or_create()
        return [Chunk(name=n).load_or_create() for n in names]


def test_lazy_dependency_from_manifest_fanout() -> None:
    manifest = Manifest(seed=7)
    dataset = DatasetFromManifest(manifest=manifest)

    dataset.load_or_create()

    deps = _deps(dataset)
    assert _ids(deps.eager) == {manifest.object_id}
    expected_lazy = {Chunk(name=f"chunk-7-{i}").object_id for i in range(3)}
    assert _ids(deps.lazy) == expected_lazy
    assert all(ref.via == "load_or_create" for ref in deps.lazy)


class OptionalLeaf(Furu[str]):
    name: str

    def _create(self) -> str:
        return f"opt:{self.name}"


class TryLoader(Furu[str]):
    name: str

    def _create(self) -> str:
        candidate = OptionalLeaf(name=self.name).try_load()
        return candidate if candidate is not None else f"missing:{self.name}"


def test_try_load_records_lazy_when_missing() -> None:
    obj = TryLoader(name="absent")

    assert obj.load_or_create() == "missing:absent"

    deps = _deps(obj)
    expected = OptionalLeaf(name="absent")
    assert _ids(deps.lazy) == {expected.object_id}
    [ref] = deps.lazy
    assert ref.via == "try_load"


def test_try_load_records_lazy_when_present() -> None:
    OptionalLeaf(name="present").load_or_create()
    obj = TryLoader(name="present")

    assert obj.load_or_create() == "opt:present"

    deps = _deps(obj)
    expected = OptionalLeaf(name="present")
    assert _ids(deps.lazy) == {expected.object_id}


class DuplicateCalls(Furu[str]):
    name: str

    def _create(self) -> str:
        a = Leaf(name=self.name).load_or_create()
        b = Leaf(name=self.name).load_or_create()
        return f"{a}|{b}"


def test_duplicate_lazy_calls_dedupe_to_one_entry() -> None:
    obj = DuplicateCalls(name="d")

    obj.load_or_create()

    deps = _deps(obj)
    assert len(deps.lazy) == 1
    assert deps.lazy[0].object_id == Leaf(name="d").object_id


class EagerAndLazySame(Furu[str]):
    leaf: Leaf

    def _create(self) -> str:
        return f"hit({self.leaf.load_or_create()})"


def test_load_or_create_on_eager_field_does_not_become_lazy() -> None:
    leaf = Leaf(name="shared")
    obj = EagerAndLazySame(leaf=leaf)

    obj.load_or_create()

    deps = _deps(obj)
    assert _ids(deps.eager) == {leaf.object_id}
    assert deps.lazy == ()


class GrandChild(Furu[str]):
    name: str

    def _create(self) -> str:
        return f"gc:{self.name}"


class Middle(Furu[str]):
    gc: GrandChild

    def _create(self) -> str:
        return f"mid({self.gc.load_or_create()})"


class Top(Furu[str]):
    name: str

    def _create(self) -> str:
        return Middle(gc=GrandChild(name=self.name)).load_or_create()


def test_no_transitive_dependency_promotion() -> None:
    top = Top(name="root")
    top.load_or_create()

    top_deps = _deps(top)
    middle = Middle(gc=GrandChild(name="root"))
    grandchild = GrandChild(name="root")

    assert _ids(top_deps.lazy) == {middle.object_id}
    assert _ids(top_deps.eager) == set()

    middle_deps = _deps(middle)
    assert _ids(middle_deps.eager) == {grandchild.object_id}
    assert middle_deps.lazy == ()

    grandchild_deps = _deps(grandchild)
    assert grandchild_deps.eager == ()
    assert grandchild_deps.lazy == ()


class GlobalShared(Furu[str]):
    def _create(self) -> str:
        return "global"


class BatchedConsumer(Furu[str]):
    key: int

    @classmethod
    def _create_batched(cls, objs) -> list[str]:
        shared = GlobalShared().load_or_create()
        return [f"{shared}:{obj.key}" for obj in objs]


def test_batched_records_shared_lazy_for_every_member() -> None:
    objs = [BatchedConsumer(key=k) for k in (1, 2, 3)]
    furu.load_or_create(objs)

    shared_id = GlobalShared().object_id
    for obj in objs:
        deps = _deps(obj)
        assert _ids(deps.lazy) == {shared_id}
        assert deps.eager == ()


class BatchedDynamicConsumer(Furu[str]):
    key: int

    @classmethod
    def _create_batched(cls, objs) -> list[str]:
        per_key = {obj.key: Leaf(name=f"k{obj.key}").load_or_create() for obj in objs}
        return [f"{obj.key}:{per_key[obj.key]}" for obj in objs]


def test_batched_per_object_dynamic_loads_apply_to_all() -> None:
    objs = [BatchedDynamicConsumer(key=k) for k in (1, 2)]
    furu.load_or_create(objs)

    expected = {Leaf(name="k1").object_id, Leaf(name="k2").object_id}
    for obj in objs:
        deps = _deps(obj)
        assert _ids(deps.lazy) == expected


class BatchedEagerField(Furu[str]):
    leaf: Leaf

    @classmethod
    def _create_batched(cls, objs) -> list[str]:
        return [f"b({obj.leaf.load_or_create()})" for obj in objs]


def test_batched_eager_field_loads_do_not_become_cross_object_lazy() -> None:
    a = BatchedEagerField(leaf=Leaf(name="a"))
    b = BatchedEagerField(leaf=Leaf(name="b"))
    furu.load_or_create([a, b])

    a_deps = _deps(a)
    b_deps = _deps(b)
    assert _ids(a_deps.eager) == {Leaf(name="a").object_id}
    assert a_deps.lazy == ()
    assert _ids(b_deps.eager) == {Leaf(name="b").object_id}
    assert b_deps.lazy == ()


class _OverrideBaseDep(Furu[str]):
    name: str

    @furu.dependency
    def child(self) -> Leaf:
        return Leaf(name=f"base:{self.name}")

    def _create(self) -> str:
        return self.child.load_or_create()


class _OverrideShadowedDep(_OverrideBaseDep):
    def child(self) -> Leaf:  # type: ignore[override]
        return Leaf(name=f"shadow:{self.name}")

    def _create(self) -> str:
        return self.child().load_or_create()


def test_subclass_can_shadow_dependency_descriptor() -> None:
    obj = _OverrideShadowedDep(name="x")
    obj.load_or_create()

    deps = _deps(obj)
    assert deps.eager == ()
    assert _ids(deps.lazy) == {Leaf(name="shadow:x").object_id}


def test_no_recorder_active_when_no_create_running() -> None:
    leaf = Leaf(name="standalone")

    leaf.load_or_create()

    deps = _deps(leaf)
    assert deps.eager == ()
    assert deps.lazy == ()


class _RunningMetadataSpy(Furu[str]):
    leaf: Leaf

    def _create(self) -> str:
        raw = json.loads(self._metadata_path.read_text())
        assert raw["kind"] == "running"
        assert len(raw["dependencies"]["eager"]) == 1
        assert raw["dependencies"]["eager"][0]["object_id"] == self.leaf.object_id
        assert raw["dependencies"]["lazy"] == []
        return self.leaf.load_or_create()


def test_running_metadata_carries_eager_dependencies() -> None:
    obj = _RunningMetadataSpy(leaf=Leaf(name="slow"))
    obj.load_or_create()


@pytest.mark.parametrize(
    "expected_via",
    [
        pytest.param("field", id="field-via"),
    ],
)
def test_dependency_ref_fields_populated(expected_via: str) -> None:
    leaf = Leaf(name="ref")
    parent = Parent(leaf=leaf)
    parent.load_or_create()

    [ref] = _deps(parent).eager
    assert ref.via == expected_via
    assert ref.object_id == leaf.object_id
    assert ref.class_name.endswith("Leaf")
    assert ref.data_path == leaf.data_dir
    assert ref.artifact_hash == leaf.artifact_hash
    assert ref.artifact_schema_hash == leaf.artifact_schema_hash
