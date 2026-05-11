from __future__ import annotations

from typing import ClassVar

import furu
from furu import Furu


class SubmittedLeaf(Furu[str]):
    name: str
    create_calls: ClassVar[list[str]] = []

    def create(self) -> str:
        type(self).create_calls.append(self.name)
        return f"leaf:{self.name}"


class SubmittedDeclaredParent(Furu[str]):
    child: SubmittedLeaf
    create_calls: ClassVar[list[str]] = []

    def create(self) -> str:
        type(self).create_calls.append(self.child.name)
        return self.child.load_or_create()


class SubmittedDynamicParent(Furu[str]):
    name: str
    create_calls: ClassVar[list[str]] = []

    def create(self) -> str:
        type(self).create_calls.append(self.name)
        return SubmittedLeaf(name=self.name).load_or_create()


class SubmittedTryLoadParent(Furu[str]):
    name: str
    create_calls: ClassVar[list[str]] = []

    def create(self) -> str:
        type(self).create_calls.append(self.name)
        return SubmittedLeaf(name=self.name).try_load()


def setup_function() -> None:
    SubmittedLeaf.create_calls.clear()
    SubmittedDeclaredParent.create_calls.clear()
    SubmittedDynamicParent.create_calls.clear()
    SubmittedTryLoadParent.create_calls.clear()


def test_submit_single_local_executor_loads_result_from_storage() -> None:
    obj = SubmittedLeaf(name="single")

    submission = furu.submit(obj, executor=furu.LocalExecutor(num_workers=2))

    assert submission.result() == "leaf:single"
    assert SubmittedLeaf.create_calls == ["single"]


def test_furu_submit_method_delegates_to_public_submit() -> None:
    obj = SubmittedLeaf(name="method")

    submission = obj.submit(executor=furu.LocalExecutor(num_workers=1))

    assert submission.result() == "leaf:method"


def test_submit_list_preserves_input_order() -> None:
    objs = [
        SubmittedLeaf(name="a"),
        SubmittedLeaf(name="b"),
        SubmittedLeaf(name="c"),
    ]

    submission = furu.submit(objs, executor=furu.LocalExecutor(num_workers=3))

    assert submission.result() == ["leaf:a", "leaf:b", "leaf:c"]


def test_submit_runs_declared_dependencies_before_parent() -> None:
    child = SubmittedLeaf(name="declared")
    parent = SubmittedDeclaredParent(child=child)

    submission = furu.submit(parent, executor=furu.LocalExecutor(num_workers=2))

    assert submission.result() == "leaf:declared"
    assert SubmittedLeaf.create_calls == ["declared"]
    assert SubmittedDeclaredParent.create_calls == ["declared"]


def test_submit_dynamic_load_or_create_retries_parent_after_dependency() -> None:
    parent = SubmittedDynamicParent(name="dynamic")

    submission = furu.submit(parent, executor=furu.LocalExecutor(num_workers=2))

    assert submission.result() == "leaf:dynamic"
    assert SubmittedLeaf.create_calls == ["dynamic"]
    assert SubmittedDynamicParent.create_calls == ["dynamic", "dynamic"]


def test_submit_dynamic_try_load_retries_parent_after_dependency() -> None:
    parent = SubmittedTryLoadParent(name="try-load")

    submission = furu.submit(parent, executor=furu.LocalExecutor(num_workers=2))

    assert submission.result() == "leaf:try-load"
    assert SubmittedLeaf.create_calls == ["try-load"]
    assert SubmittedTryLoadParent.create_calls == ["try-load", "try-load"]
