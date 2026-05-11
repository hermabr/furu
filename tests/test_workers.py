from __future__ import annotations

from typing import ClassVar

import pytest

import furu
from furu import Furu
from furu.graph import node_key_for
from furu.submission import SubmissionFailed
from furu.worker_context import _DependencyNotReady, worker_execution_context


class WorkerLeaf(Furu[str]):
    name: str
    create_calls: ClassVar[list[str]] = []

    def create(self) -> str:
        type(self).create_calls.append(self.name)
        return f"leaf:{self.name}"


class DeclaredWorkerParent(Furu[str]):
    child: WorkerLeaf
    create_calls: ClassVar[int] = 0

    def create(self) -> str:
        type(self).create_calls += 1
        return self.child.load_or_create()


class DynamicWorkerParent(Furu[str]):
    name: str
    create_calls: ClassVar[int] = 0

    def create(self) -> str:
        type(self).create_calls += 1
        return WorkerLeaf(name=self.name).load_or_create()


class TryLoadWorkerParent(Furu[str]):
    name: str
    create_calls: ClassVar[int] = 0

    def create(self) -> str:
        type(self).create_calls += 1
        return WorkerLeaf(name=self.name).try_load()


class FailingWorkerLeaf(Furu[str]):
    def create(self) -> str:
        raise ValueError("worker boom")


class FailingWorkerParent(Furu[str]):
    child: FailingWorkerLeaf

    def create(self) -> str:
        return self.child.load_or_create()


class WorkerBatchOnly(Furu[str]):
    key: int
    batch_calls: ClassVar[list[tuple[int, ...]]] = []

    @classmethod
    def create_batched(cls, objs) -> list[str]:
        keys = tuple(obj.key for obj in objs)
        cls.batch_calls.append(keys)
        return [f"batch:{obj.key}" for obj in objs]


@pytest.fixture(autouse=True)
def _reset_worker_test_state() -> None:
    WorkerLeaf.create_calls.clear()
    DeclaredWorkerParent.create_calls = 0
    DynamicWorkerParent.create_calls = 0
    TryLoadWorkerParent.create_calls = 0
    WorkerBatchOnly.batch_calls.clear()


def test_local_submit_executes_declared_dependency_graph() -> None:
    parent = DeclaredWorkerParent(child=WorkerLeaf(name="declared"))

    submission = furu.submit(parent, executor=furu.LocalExecutor(num_workers=2))

    assert submission.result(timeout_s=5) == "leaf:declared"
    assert WorkerLeaf.create_calls == ["declared"]
    assert DeclaredWorkerParent.create_calls == 1


def test_local_submit_discovers_dynamic_load_or_create_dependency() -> None:
    parent = DynamicWorkerParent(name="dynamic")

    submission = parent.submit(executor=furu.LocalExecutor(num_workers=1))

    assert submission.result(timeout_s=5) == "leaf:dynamic"
    assert WorkerLeaf.create_calls == ["dynamic"]
    assert DynamicWorkerParent.create_calls == 2


def test_local_submit_discovers_dynamic_try_load_dependency() -> None:
    parent = TryLoadWorkerParent(name="try-load")

    submission = furu.submit(parent, executor=furu.LocalExecutor(num_workers=1))

    assert submission.result(timeout_s=5) == "leaf:try-load"
    assert WorkerLeaf.create_calls == ["try-load"]
    assert TryLoadWorkerParent.create_calls == 2


def test_local_submit_preserves_sequence_shape_and_order() -> None:
    objs = [WorkerBatchOnly(key=2), WorkerBatchOnly(key=1)]

    submission = furu.submit(objs, executor=furu.LocalExecutor(num_workers=2))

    assert submission.result(timeout_s=5) == ["batch:2", "batch:1"]
    assert sorted(WorkerBatchOnly.batch_calls) == [(1,), (2,)]


def test_worker_failure_marks_submission_failed() -> None:
    parent = FailingWorkerParent(child=FailingWorkerLeaf())

    submission = furu.submit(parent, executor=furu.LocalExecutor(num_workers=2))

    with pytest.raises(SubmissionFailed, match="worker boom"):
        submission.result(timeout_s=5)


def test_worker_load_or_create_reports_all_missing_sequence_dependencies() -> None:
    first = WorkerLeaf(name="first")
    second = WorkerLeaf(name="second")

    with worker_execution_context(current_node=node_key_for(first), lease_id="lease"):
        with pytest.raises(_DependencyNotReady) as exc_info:
            furu.load_or_create([first, second])

    exc = exc_info.value
    assert exc.call_kind == "load_or_create"
    assert exc.dependencies == (first, second)
    assert exc.keys == (node_key_for(first), node_key_for(second))
