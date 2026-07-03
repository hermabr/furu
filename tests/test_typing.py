from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, assert_type

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


class TypingIntCodec(furu.Codec[int]):
    @classmethod
    def matches(cls, value: object) -> bool:
        return isinstance(value, int)

    def save(self, value: int, artifact_dir: Path) -> Mapping[str, object]:
        (artifact_dir / "value.txt").write_text(str(value), encoding="utf-8")
        return {}

    def load(self, metadata: Mapping[str, object], artifact_dir: Path) -> int:
        return int((artifact_dir / "value.txt").read_text(encoding="utf-8"))


@dataclass(frozen=True)
class TypingRefOutput:
    value: furu.Ref[int]


if TYPE_CHECKING:
    parent = TypingParent()
    assert_type(parent.cached_child, TypingChild)
    assert_type(parent.children, list[TypingChild])
    assert_type(parent.children[0], TypingChild)
    assert_type(furu.load_existing([parent.cached_child]), list[int])
    assert_type(furu.create(parent.cached_child), int)
    assert_type(furu.create([parent.cached_child]), list[int])
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
    assert_type(furu.ref(1, codec=TypingIntCodec), furu.Ref[int])
    assert_type(
        TypingRefOutput(value=furu.ref(1, codec=TypingIntCodec)).value, furu.Ref[int]
    )
    TypingRefOutput(value=1)  # ty: ignore[invalid-argument-type]
