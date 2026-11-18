from typing import ClassVar

import pytest

import furu
from furu import Spec
from furu.resources import Worker
from furu.utils import _hash_dict_deterministically


class SnapshotValue(Spec[str]):
    name: str

    def create(self) -> str:
        return self.name


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
