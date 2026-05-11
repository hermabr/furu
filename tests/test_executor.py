from __future__ import annotations

from typing import ClassVar

import pytest

import furu
from furu import Furu
from furu.executor import InMemoryScheduler, MaxSuspensionsExceeded, WorkerRunner


class ExecutorLeaf(Furu[str]):
    name: str
    create_calls: ClassVar[list[str]] = []

    def create(self) -> str:
        type(self).create_calls.append(self.name)
        return f"leaf:{self.name}"


class DeclaredParent(Furu[str]):
    left: ExecutorLeaf
    right: ExecutorLeaf

    @furu.dependency
    def duplicate_left(self) -> ExecutorLeaf:
        return ExecutorLeaf(name=self.left.name)

    def create(self) -> str:
        return self.left.load_or_create()


class DynamicParent(Furu[str]):
    name: str
    create_calls: ClassVar[int] = 0

    def create(self) -> str:
        type(self).create_calls += 1
        return ExecutorLeaf(name=self.name).load_or_create()


class TryLoadOnlyParent(Furu[str]):
    name: str

    def create(self) -> str:
        try:
            ExecutorLeaf(name=self.name).try_load()
        except NotImplementedError:
            return "missing"
        return "loaded"


class RediscoveringParent(Furu[str]):
    create_calls: ClassVar[int] = 0

    def create(self) -> str:
        type(self).create_calls += 1
        return ExecutorLeaf(
            name=f"rediscovered-{type(self).create_calls}"
        ).load_or_create()


@pytest.fixture(autouse=True)
def _reset_executor_trackers() -> None:
    ExecutorLeaf.create_calls.clear()
    DynamicParent.create_calls = 0
    RediscoveringParent.create_calls = 0


def test_scheduler_submit_plans_declared_dependencies_and_dedupes() -> None:
    left = ExecutorLeaf(name="left")
    right = ExecutorLeaf(name="right")
    parent = DeclaredParent(left=left, right=right)
    scheduler = InMemoryScheduler()

    scheduler.submit([parent, parent])

    assert set(scheduler.jobs) == {parent.object_id, left.object_id, right.object_id}
    assert scheduler.jobs[parent.object_id].dependencies == {
        left.object_id,
        right.object_id,
    }
    assert scheduler.jobs[left.object_id].dependencies == set()
    assert scheduler.jobs[right.object_id].dependencies == set()


def test_worker_schedules_dynamic_missing_dependencies_and_retries_parent() -> None:
    parent = DynamicParent(name="dynamic")
    scheduler = InMemoryScheduler()
    scheduler.submit(parent)

    WorkerRunner(scheduler).run_until_complete()

    child = ExecutorLeaf(name="dynamic")
    assert parent.try_load() == "leaf:dynamic"
    assert DynamicParent.create_calls == 2
    assert ExecutorLeaf.create_calls == ["dynamic"]
    assert scheduler.jobs[parent.object_id].dependencies == {child.object_id}
    assert {job.state for job in scheduler.jobs.values()} == {"completed"}


def test_executor_mode_load_or_create_loads_cached_dependency_without_suspending() -> (
    None
):
    child = ExecutorLeaf(name="cached")
    assert child.load_or_create() == "leaf:cached"

    parent = DynamicParent(name="cached")
    scheduler = InMemoryScheduler()
    scheduler.submit(parent)

    WorkerRunner(scheduler).run_until_complete()

    assert parent.try_load() == "leaf:cached"
    assert DynamicParent.create_calls == 1
    assert scheduler.jobs[parent.object_id].dependencies == set()


def test_try_load_inside_executor_create_does_not_schedule_missing_dependency() -> None:
    parent = TryLoadOnlyParent(name="optional")
    child = ExecutorLeaf(name="optional")
    scheduler = InMemoryScheduler()
    scheduler.submit(parent)

    WorkerRunner(scheduler).run_until_complete()

    assert parent.try_load() == "missing"
    assert child.object_id not in scheduler.jobs


def test_repeated_dynamic_suspension_beyond_limit_fails_job() -> None:
    parent = RediscoveringParent()
    scheduler = InMemoryScheduler(max_suspensions_per_job=1)
    scheduler.submit(parent)

    with pytest.raises(MaxSuspensionsExceeded, match="suspended more than 1 times"):
        WorkerRunner(scheduler).run_until_complete()

    job = scheduler.jobs[parent.object_id]
    assert job.state == "failed"
    assert job.suspensions == 2
