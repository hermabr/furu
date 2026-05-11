from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest

import furu
from furu import Furu
from furu.executor import (
    ExcessiveSuspensions,
    InMemoryScheduler,
    LocalExecutor,
    LocalExecutorFailed,
    Planner,
)


class ExecutorLeaf(Furu[str]):
    name: str
    create_calls: ClassVar[list[str]] = []

    def create(self) -> str:
        type(self).create_calls.append(self.name)
        return f"leaf:{self.name}"


class ExecutorParent(Furu[str]):
    child: ExecutorLeaf
    create_calls: ClassVar[int] = 0

    def create(self) -> str:
        type(self).create_calls += 1
        return f"parent:{self.child.load_or_create()}"


class DynamicParent(Furu[str]):
    name: str
    create_calls: ClassVar[int] = 0

    def create(self) -> str:
        type(self).create_calls += 1
        return f"dynamic:{ExecutorLeaf(name=self.name).load_or_create()}"


class OptionalParent(Furu[str]):
    name: str

    def create(self) -> str:
        try:
            ExecutorLeaf(name=self.name).try_load()
        except NotImplementedError:
            return "optional-missing"
        return "optional-loaded"


class AlwaysNewDependency(Furu[str]):
    create_calls: ClassVar[int] = 0

    def create(self) -> str:
        type(self).create_calls += 1
        return ExecutorLeaf(name=f"new-{type(self).create_calls}").load_or_create()


@dataclass(frozen=True)
class LeafBundle:
    child: ExecutorLeaf


class PlannedMiddle(Furu[str]):
    bundle: LeafBundle

    def create(self) -> str:
        return self.bundle.child.load_or_create()


class PlannedRoot(Furu[str]):
    middle: PlannedMiddle
    extra: ExecutorLeaf

    @furu.dependency
    def computed(self) -> ExecutorLeaf:
        return self.extra

    def create(self) -> str:
        return self.middle.load_or_create()


@pytest.fixture(autouse=True)
def _reset_executor_test_state() -> None:
    ExecutorLeaf.create_calls.clear()
    ExecutorParent.create_calls = 0
    DynamicParent.create_calls = 0
    AlwaysNewDependency.create_calls = 0


def test_planner_walks_declared_dependency_graph_and_dedupes_by_object_id() -> None:
    child = ExecutorLeaf(name="child")
    extra = ExecutorLeaf(name="extra")
    middle = PlannedMiddle(bundle=LeafBundle(child=child))
    root = PlannedRoot(middle=middle, extra=extra)

    graph = Planner().plan(root)

    assert set(graph.artifacts) == {
        root.object_id,
        middle.object_id,
        child.object_id,
        extra.object_id,
    }
    assert graph.edges == {
        (root.object_id, middle.object_id),
        (root.object_id, extra.object_id),
        (middle.object_id, child.object_id),
    }


def test_scheduler_claims_only_queued_jobs_with_completed_dependencies() -> None:
    child = ExecutorLeaf(name="ready")
    parent = ExecutorParent(child=child)
    scheduler = InMemoryScheduler(root_ids=[parent.object_id])

    scheduler.submit(parent, dependencies=[child.object_id])
    scheduler.submit(child)

    first = scheduler.claim_ready()
    assert first is not None
    assert first.object_id == child.object_id
    assert scheduler.claim_ready() is None

    scheduler.complete_job(child.object_id)
    second = scheduler.claim_ready()

    assert second is not None
    assert second.object_id == parent.object_id


def test_local_executor_runs_declared_dependencies_before_parent() -> None:
    child = ExecutorLeaf(name="declared")
    parent = ExecutorParent(child=child)

    assert LocalExecutor(max_workers=2).run(parent) == "parent:leaf:declared"
    assert ExecutorLeaf.create_calls == ["declared"]
    assert ExecutorParent.create_calls == 1


def test_local_executor_schedules_missing_runtime_dependency_and_reruns_parent() -> (
    None
):
    parent = DynamicParent(name="runtime")

    assert LocalExecutor(max_workers=2).run(parent) == "dynamic:leaf:runtime"
    assert ExecutorLeaf.create_calls == ["runtime"]
    assert DynamicParent.create_calls == 2


def test_try_load_inside_executor_job_does_not_schedule_work() -> None:
    parent = OptionalParent(name="optional")

    assert LocalExecutor(max_workers=1).run(parent) == "optional-missing"
    assert ExecutorLeaf.create_calls == []


def test_excessive_repeated_suspensions_fail_clearly() -> None:
    with pytest.raises(LocalExecutorFailed, match="suspended 2 times") as exc_info:
        LocalExecutor(max_workers=1, max_suspensions_per_job=1).run(
            AlwaysNewDependency()
        )

    assert isinstance(exc_info.value.__cause__, ExcessiveSuspensions)
