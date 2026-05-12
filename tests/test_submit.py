from typing import ClassVar

import pytest

from furu import Furu, submit


SUBMIT_EVENTS: list[tuple[str, int]] = []


class SubmitLeaf(Furu[str]):
    key: int
    create_calls: ClassVar[list[int]] = []

    def create(self) -> str:
        type(self).create_calls.append(self.key)
        SUBMIT_EVENTS.append(("leaf", self.key))
        return f"leaf:{self.key}"


class SubmitDeclaredParent(Furu[str]):
    key: int
    child: SubmitLeaf
    create_calls: ClassVar[list[int]] = []

    def create(self) -> str:
        type(self).create_calls.append(self.key)
        SUBMIT_EVENTS.append(("declared", self.key))
        return f"declared:{self.child.load_or_create()}"


class SubmitLazyParent(Furu[str]):
    key: int
    create_calls: ClassVar[list[int]] = []

    def create(self) -> str:
        type(self).create_calls.append(self.key)
        SUBMIT_EVENTS.append(("lazy", self.key))
        return f"lazy:{SubmitLeaf(key=self.key).load_or_create()}"


class SubmitLazyDeclaredParent(Furu[str]):
    key: int
    create_calls: ClassVar[list[int]] = []

    def create(self) -> str:
        type(self).create_calls.append(self.key)
        SUBMIT_EVENTS.append(("lazy-declared", self.key))
        return SubmitDeclaredParent(
            key=self.key,
            child=SubmitLeaf(key=self.key),
        ).load_or_create()


@pytest.fixture(autouse=True)
def _reset_submit_state() -> None:
    SUBMIT_EVENTS.clear()
    SubmitLeaf.create_calls.clear()
    SubmitDeclaredParent.create_calls.clear()
    SubmitLazyParent.create_calls.clear()
    SubmitLazyDeclaredParent.create_calls.clear()


def test_submit_runs_declared_dependencies_before_dependents() -> None:
    child = SubmitLeaf(key=1)
    parent = SubmitDeclaredParent(key=2, child=child)

    submit([parent])

    assert SUBMIT_EVENTS == [("leaf", 1), ("declared", 2)]
    assert parent.load_or_create() == "declared:leaf:1"
    assert SubmitLeaf.create_calls == [1]
    assert SubmitDeclaredParent.create_calls == [2]


def test_submit_adds_lazy_dependencies_and_retries_parent() -> None:
    parent = SubmitLazyParent(key=3)

    submit([parent])

    assert SUBMIT_EVENTS == [("lazy", 3), ("leaf", 3), ("lazy", 3)]
    assert parent.load_or_create() == "lazy:leaf:3"
    assert SubmitLeaf.create_calls == [3]
    assert SubmitLazyParent.create_calls == [3, 3]


def test_submit_merges_declared_dependencies_of_lazy_dependencies() -> None:
    parent = SubmitLazyDeclaredParent(key=4)

    submit([parent])

    assert SUBMIT_EVENTS == [
        ("lazy-declared", 4),
        ("leaf", 4),
        ("declared", 4),
        ("lazy-declared", 4),
    ]
    assert parent.load_or_create() == "declared:leaf:4"
    assert SubmitLeaf.create_calls == [4]
    assert SubmitDeclaredParent.create_calls == [4]
    assert SubmitLazyDeclaredParent.create_calls == [4, 4]
