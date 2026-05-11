from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import ClassVar

import pytest

import furu
from furu.config import _FuruDirectories, config
from furu.execution import BlockedOnDependencies, executor_job_context


@pytest.fixture(autouse=True)
def _isolate_data_dir(tmp_path: Path) -> None:
    config.directories = _FuruDirectories(data=tmp_path / "data")


class ExecutorLeaf(furu.Furu[str]):
    name: str
    create_calls: ClassVar[list[str]] = []

    def create(self) -> str:
        type(self).create_calls.append(self.name)
        return f"leaf:{self.name}"


class ExecutorParent(furu.Furu[str]):
    child: ExecutorLeaf
    create_calls: ClassVar[int] = 0

    def create(self) -> str:
        type(self).create_calls += 1
        return f"parent:{self.child.load_or_create()}"


class DynamicParent(furu.Furu[str]):
    name: str
    create_calls: ClassVar[int] = 0

    @cached_property
    def dynamic_child(self) -> ExecutorLeaf:
        return ExecutorLeaf(name=self.name)

    def create(self) -> str:
        type(self).create_calls += 1
        return f"dynamic:{self.dynamic_child.load_or_create()}"


class AlwaysBlocked(furu.Furu[str]):
    create_calls: ClassVar[int] = 0

    def create(self) -> str:
        type(self).create_calls += 1
        ExecutorLeaf(name=f"rediscovered-{type(self).create_calls}").load_or_create()
        return "unreachable"


@pytest.fixture(autouse=True)
def _reset_executor_trackers() -> None:
    ExecutorLeaf.create_calls.clear()
    ExecutorParent.create_calls = 0
    DynamicParent.create_calls = 0
    AlwaysBlocked.create_calls = 0


def test_scheduler_plans_declared_dependencies() -> None:
    child = ExecutorLeaf(name="a")
    parent = ExecutorParent(child=child)
    scheduler = furu.InMemoryScheduler()

    scheduler.submit(parent)

    assert set(scheduler.jobs) == {parent.object_id, child.object_id}
    assert scheduler.jobs[parent.object_id].dependencies == {child.object_id}
    assert scheduler.jobs[child.object_id].dependencies == set()


def test_executor_runs_declared_dependency_before_parent() -> None:
    child = ExecutorLeaf(name="a")
    parent = ExecutorParent(child=child)

    scheduler = furu.execute_local(parent)

    assert scheduler.failed_jobs() == ()
    assert parent.try_load() == "parent:leaf:a"
    assert ExecutorLeaf.create_calls == ["a"]
    assert ExecutorParent.create_calls == 1
    assert scheduler.jobs[parent.object_id].state == "completed"
    assert scheduler.jobs[child.object_id].state == "completed"


def test_load_or_create_blocks_on_missing_dependency_in_executor_context() -> None:
    child = ExecutorLeaf(name="missing")

    with executor_job_context(), pytest.raises(BlockedOnDependencies) as exc_info:
        child.load_or_create()

    assert exc_info.value.deps == (child,)
    assert not child._internal_furu_dir.exists()


def test_dynamic_dependency_suspends_and_retries_parent() -> None:
    parent = DynamicParent(name="late")

    scheduler = furu.execute_local(parent)

    child = parent.dynamic_child
    assert scheduler.failed_jobs() == ()
    assert scheduler.jobs[parent.object_id].dependencies == {child.object_id}
    assert scheduler.jobs[parent.object_id].suspensions == 1
    assert DynamicParent.create_calls == 2
    assert child.try_load() == "leaf:late"
    assert parent.try_load() == "dynamic:leaf:late"


def test_cached_dynamic_dependency_does_not_suspend() -> None:
    child = ExecutorLeaf(name="cached")
    child.load_or_create()
    parent = DynamicParent(name="cached")

    scheduler = furu.execute_local(parent)

    assert scheduler.failed_jobs() == ()
    assert scheduler.jobs[parent.object_id].dependencies == set()
    assert scheduler.jobs[parent.object_id].suspensions == 0
    assert DynamicParent.create_calls == 1


def test_repeated_suspension_past_limit_fails_job() -> None:
    parent = AlwaysBlocked()

    scheduler = furu.execute_local(parent, max_suspensions_per_job=1)

    job = scheduler.jobs[parent.object_id]
    assert job.state == "failed"
    assert job.error is not None
    assert "suspended more than 1 times" in str(job.error)
