import json
from typing import ClassVar

import pytest

import furu
from furu import Spec
from furu.resources import Worker
from furu.storage._layout import schema_snapshot_path_in
from furu.utils import _hash_dict_deterministically


class SnapshotValue(Spec[str]):
    name: str

    def create(self) -> str:
        return self.name


class FailingValue(Spec[str]):
    name: str

    def create(self) -> str:
        raise RuntimeError("boom")


class Pair(Spec[str]):
    label: str
    values: tuple[SnapshotValue, ...]

    def create(self) -> str:
        return self.label


class ReturnsLeaves(Spec[str]):
    label: str
    returned: ClassVar[tuple[str, ...]] = ()

    @furu.dependency
    def leaves(self) -> tuple[SnapshotValue, ...]:
        return tuple(SnapshotValue(name=name) for name in self.returned)

    def create(self) -> str:
        return self.label


def _train_id() -> str:
    class Train(Spec[int]):
        seed: int

        def create(self) -> int:
            return self.seed

    Train.__qualname__ = "Train"
    return Train(seed=1).object_id


def _batched_train_id() -> str:
    class Train(Spec[int]):
        seed: int

        def batch_key(self, worker: Worker) -> tuple[None, int]:
            return (None, 8)

        @furu.batched(batch_key)
        def create(specs: list["Train"]) -> list[int]:
            return [spec.seed for spec in specs]

    Train.__qualname__ = "Train"
    return Train(seed=1).object_id


def test_an_id_is_one_fingerprint_per_field_and_a_held_spec_counts_as_its_id() -> None:
    a, b = SnapshotValue(name="a"), SnapshotValue(name="b")
    fingerprint = _hash_dict_deterministically

    assert Pair(label="x", values=(a, b))._artifact_hash == fingerprint(
        {
            "label": fingerprint("x"),
            "values": fingerprint(
                {
                    "|kind": "tuple",
                    "|value": [
                        {"|kind": "spec", "|value": a.object_id},
                        {"|kind": "spec", "|value": b.object_id},
                    ],
                }
            ),
        }
    )


def test_what_a_dependency_returns_is_part_of_the_id_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def object_id(returned: tuple[str, ...]) -> str:
        monkeypatch.setattr(ReturnsLeaves, "returned", returned)
        return ReturnsLeaves(label="x").object_id

    assert object_id(("a", "b")) == object_id(("a", "b"))
    assert object_id(("a", "b")) != object_id(("b", "a"))


def test_batching_a_class_keeps_its_ids() -> None:
    assert _train_id() == _batched_train_id()


def test_schema_snapshot_written_on_first_store_and_never_rewritten() -> None:
    first = SnapshotValue(name="first")
    schema_path = schema_snapshot_path_in(first._base_dir.parent)
    assert not schema_path.exists()

    first.create()
    assert json.loads(schema_path.read_text()) == first._schema_data

    stat_before = schema_path.stat()
    second = SnapshotValue(name="second")
    assert schema_snapshot_path_in(second._base_dir.parent) == schema_path
    second.create()
    stat_after = schema_path.stat()
    assert (stat_after.st_ino, stat_after.st_mtime_ns) == (
        stat_before.st_ino,
        stat_before.st_mtime_ns,
    )


def test_schema_snapshot_not_written_when_create_fails() -> None:
    failing = FailingValue(name="x")
    with pytest.raises(RuntimeError, match="boom"):
        failing.create()
    assert not schema_snapshot_path_in(failing._base_dir.parent).exists()
