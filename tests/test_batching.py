from typing import ClassVar, Self

import pytest

import furu
from furu import Furu, load_or_create


class ScalarHookProbe(Furu[int]):
    key: int

    create_calls: ClassVar[list[int]] = []

    def _create(self) -> int:
        ScalarHookProbe.create_calls.append(self.key)
        return self.key * 10


class InheritedScalarHookProbe(ScalarHookProbe):
    label: str


class BatchHookProbe(Furu[int]):
    key: int

    batches: ClassVar[list[list[int]]] = []

    @classmethod
    def _create_batched(  # ty: ignore[invalid-method-override]
        cls, items: list[Self]
    ) -> list[int]:
        BatchHookProbe.batches.append([item.key for item in items])
        return [item.key * 10 for item in items]


class InheritedBatchHookProbe(BatchHookProbe):
    label: str


class BothHooksProbe(Furu[int]):
    key: int

    def _create(self) -> int:
        return self.key

    @classmethod
    def _create_batched(  # ty: ignore[invalid-method-override]
        cls, items: list[Self]
    ) -> list[int]:
        return [item.key for item in items]


class NeitherHookProbe(Furu[int]):
    key: int


class WrongLengthBatchProbe(Furu[int]):
    key: int

    @classmethod
    def _create_batched(  # ty: ignore[invalid-method-override]
        cls, items: list[Self]
    ) -> list[int]:
        return []


class RaisingBatchProbe(Furu[int]):
    key: int

    @classmethod
    def _create_batched(  # ty: ignore[invalid-method-override]
        cls, items: list[Self]
    ) -> list[int]:
        raise RuntimeError("batch failed")


class ScalarMixedA(Furu[int]):
    key: int

    create_calls: ClassVar[list[int]] = []

    def _create(self) -> int:
        ScalarMixedA.create_calls.append(self.key)
        return self.key


class ScalarMixedB(Furu[int]):
    key: int

    create_calls: ClassVar[list[int]] = []

    def _create(self) -> int:
        ScalarMixedB.create_calls.append(self.key)
        return self.key


@pytest.fixture(autouse=True)
def _reset_probe_state() -> None:
    ScalarHookProbe.create_calls.clear()
    BatchHookProbe.batches.clear()
    ScalarMixedA.create_calls.clear()
    ScalarMixedB.create_calls.clear()


def test_scalar_only_hook_is_valid() -> None:
    assert ScalarHookProbe(key=1).load_or_create() == 10


def test_batch_only_hook_is_valid() -> None:
    assert BatchHookProbe(key=1).load_or_create() == 10


def test_dual_hook_class_is_rejected() -> None:
    with pytest.raises(TypeError, match="exactly one"):
        BothHooksProbe(key=1).load_or_create()


def test_missing_hook_class_is_rejected() -> None:
    with pytest.raises(TypeError, match="exactly one"):
        NeitherHookProbe(key=1).load_or_create()


def test_inherited_scalar_only_hook_is_valid() -> None:
    assert InheritedScalarHookProbe(key=2, label="x").load_or_create() == 20


def test_inherited_batch_only_hook_is_valid() -> None:
    assert (
        load_or_create(
            [
                InheritedBatchHookProbe(key=2, label="x"),
                InheritedBatchHookProbe(key=3, label="y"),
            ]
        )
        == [20, 30]
    )


def test_module_load_or_create_matches_instance_method() -> None:
    obj = ScalarHookProbe(key=7)

    assert furu.load_or_create(obj) == 70
    assert obj.load_or_create() == 70
    assert ScalarHookProbe.create_calls == [7]


def test_scalar_call_uses_scalar_hook() -> None:
    assert ScalarHookProbe(key=3).load_or_create() == 30
    assert ScalarHookProbe.create_calls == [3]


def test_batch_only_scalar_call_uses_singleton_batch() -> None:
    assert BatchHookProbe(key=4).load_or_create() == 40
    assert BatchHookProbe.batches == [[4]]


def test_batch_only_scalar_fallback_rejects_wrong_length_output() -> None:
    obj = WrongLengthBatchProbe(key=1)

    with pytest.raises(ValueError, match="returned 0 results for 1 items"):
        obj.load_or_create()

    assert not obj.is_completed()


def test_empty_iterable_returns_empty_list() -> None:
    assert load_or_create([]) == []


def test_mixed_concrete_classes_are_rejected_before_work() -> None:
    with pytest.raises(TypeError, match="same concrete Furu type"):
        load_or_create([ScalarMixedA(key=1), ScalarMixedB(key=1)])

    assert ScalarMixedA.create_calls == []
    assert ScalarMixedB.create_calls == []


def test_duplicate_items_are_computed_once_and_fanned_out_in_order() -> None:
    items = [
        BatchHookProbe(key=2),
        BatchHookProbe(key=1),
        BatchHookProbe(key=2),
        BatchHookProbe(key=3),
    ]

    assert load_or_create(items) == [20, 10, 20, 30]
    assert BatchHookProbe.batches == [[2, 1, 3]]


def test_batch_path_skips_cached_hits() -> None:
    cached = BatchHookProbe(key=1)
    fresh = BatchHookProbe(key=2)

    assert cached.load_or_create() == 10
    BatchHookProbe.batches.clear()

    assert load_or_create([cached, fresh]) == [10, 20]
    assert BatchHookProbe.batches == [[2]]


def test_scalar_list_path_falls_back_to_per_item_scalar_execution() -> None:
    items = [ScalarHookProbe(key=1), ScalarHookProbe(key=2), ScalarHookProbe(key=1)]

    assert load_or_create(items) == [10, 20, 10]
    assert ScalarHookProbe.create_calls == [1, 2]


def test_batch_list_path_calls_create_batched_once_for_unique_missing_items() -> None:
    items = [
        BatchHookProbe(key=5),
        BatchHookProbe(key=6),
        BatchHookProbe(key=5),
    ]

    assert load_or_create(items) == [50, 60, 50]
    assert BatchHookProbe.batches == [[5, 6]]


def test_batch_list_preserves_caller_order() -> None:
    items = [
        BatchHookProbe(key=9),
        BatchHookProbe(key=4),
        BatchHookProbe(key=9),
        BatchHookProbe(key=1),
    ]

    assert load_or_create(items) == [90, 40, 90, 10]


def test_batch_exception_commits_nothing() -> None:
    items = [RaisingBatchProbe(key=1), RaisingBatchProbe(key=2)]

    with pytest.raises(RuntimeError, match="batch failed"):
        load_or_create(items)

    assert all(not item.is_completed() for item in items)


def test_wrong_length_batch_result_commits_nothing() -> None:
    items = [WrongLengthBatchProbe(key=1), WrongLengthBatchProbe(key=2)]

    with pytest.raises(ValueError, match="returned 0 results for 2 items"):
        load_or_create(items)

    assert all(not item.is_completed() for item in items)
