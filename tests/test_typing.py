from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, assert_type

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


@furu.spec
def typed_letter_count(source: str, letter: str) -> int:
    return source.count(letter)


@furu.spec()
def typed_letter_count_with_parentheses(source: str, letter: str) -> int:
    return source.count(letter)


class TypingFunctionParent(furu.Spec[int]):
    child: furu.Spec[int]

    def create(self) -> int:
        return self.child.create()


class TypingBatch(furu.Spec[int]):
    value: int

    @furu.batched(lambda _: (None, 8))
    def create(objs: list[Self]) -> list[int]:
        return [obj.value for obj in objs]


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
    assert_type(TypingBatch(value=1).create(), int)
    assert_type(TypingBatch.create([TypingBatch(value=1)]), list[int])
    assert_type(typed_letter_count(source="banana", letter="a"), furu.Spec[int])
    assert_type(
        TypingFunctionParent(
            child=typed_letter_count(source="banana", letter="a")
        ).child,
        furu.Spec[int],
    )
    assert_type(
        typed_letter_count_with_parentheses(source="banana", letter="a"),
        furu.Spec[int],
    )
    typed_ref = furu.ref([1, 2, 3])
    assert_type(typed_ref, furu.Ref[list[int]])
    assert_type(typed_ref.load(), list[int])
    TypingRefOutput(weights=typed_ref)
    # Populating a Ref[T] field requires furu.ref(); a bare T is a type error.
    TypingRefOutput(weights=[1, 2, 3])  # ty: ignore[invalid-argument-type]
