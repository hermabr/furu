from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, assert_type

import furu


class TypingChild(furu.Spec[int]):
    def create(self) -> int:
        return 1


class TypingParent(furu.Spec[str]):
    def create(self) -> str:
        return str(
            self.cached_child.create() + sum(child.create() for child in self.children)
        )

    @furu.dependency
    def cached_child(self) -> TypingChild:
        return TypingChild()

    @furu.dependency()
    def children(self) -> list[TypingChild]:
        return [TypingChild()]


class TypingBatched(furu.Spec[str]):
    key: int

    def batch_key(self, worker: furu.Worker) -> tuple[int, int]:
        return (self.key % 2, 1_000)

    @furu.batched(batch_key)
    def create(objs: list[TypingBatched]) -> list[str]:
        return [str(obj.key) for obj in objs]


@dataclass(frozen=True)
class TypingRefOutput:
    weights: furu.Ref[list[int]]


if TYPE_CHECKING:
    parent = TypingParent()
    assert_type(parent.cached_child, TypingChild)
    assert_type(parent.children, list[TypingChild])
    assert_type(parent.children[0], TypingChild)
    assert_type(furu.load_existing([parent.cached_child]), list[int])
    assert_type(furu.create(parent.cached_child), int)
    assert_type(furu.create([parent.cached_child]), list[int])
    assert_type(TypingChild().create(), int)
    assert_type(TypingBatched(key=1).create(), str)
    assert_type(furu.create([TypingBatched(key=1)]), list[str])
    child, batched = TypingChild(), TypingBatched(key=1)
    children: list[TypingChild] = [child]
    by_name: dict[str, TypingChild] = {"a": child}
    assert_type(furu.create(children), list[int])
    assert_type(furu.create((child, batched)), tuple[int, str])
    assert_type(furu.create((child, batched, child)), tuple[int, str, int])
    assert_type(furu.create(by_name), dict[str, int])
    assert_type(furu.create({"a": child}), dict[str, int])
    assert_type(furu.load_existing(child), int)
    assert_type(furu.load_existing((child, batched)), tuple[int, str])
    # Shapes Python's type system cannot map Spec[T] -> T through are Any.
    assert_type(furu.create([[child]]), Any)
    assert_type(furu.create({"a": [child]}), Any)
    assert_type(furu.create(TypingRefOutput(weights=furu.ref([1]))), dict[str, Any])
    typed_ref = furu.ref([1, 2, 3])
    assert_type(typed_ref, furu.Ref[list[int]])
    assert_type(typed_ref.load(), list[int])
    TypingRefOutput(weights=typed_ref)
    # Populating a Ref[T] field requires furu.ref(); a bare T is a type error.
    TypingRefOutput(weights=[1, 2, 3])  # ty: ignore[invalid-argument-type]
