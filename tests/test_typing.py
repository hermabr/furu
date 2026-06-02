from __future__ import annotations

from typing import TYPE_CHECKING, assert_type

import furu


class TypingChild(furu.Furu[int]):
    def create(self) -> int:
        return 1


class TypingParent(furu.Furu[str]):
    def create(self) -> str:
        return str(
            self.cached_child.create()
            + self.uncached_child.create()
            + sum(child.create() for child in self.children)
        )

    @furu.dependency
    def cached_child(self) -> TypingChild:
        return TypingChild()

    @furu.dependency(cached=False)
    def uncached_child(self) -> TypingChild:
        return TypingChild()

    @furu.dependency
    def children(self) -> list[TypingChild]:
        return [TypingChild()]


@furu.furu_method
def typed_letter_count(source: str, letter: str) -> int:
    return source.count(letter)


if TYPE_CHECKING:
    parent = TypingParent()
    assert_type(parent.cached_child, TypingChild)
    assert_type(parent.uncached_child, TypingChild)
    assert_type(parent.children, list[TypingChild])
    assert_type(parent.children[0], TypingChild)
    assert_type(typed_letter_count(source="banana", letter="a"), furu.Furu[int])
