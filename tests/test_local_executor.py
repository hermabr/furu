from __future__ import annotations

from typing import ClassVar

import pytest

import furu
from furu import BlockedOnDependencies, Furu, Scheduler, run_local_executor


class Leaf(Furu[str]):
    name: str

    def create(self) -> str:
        return f"leaf:{self.name}"


class Pair(Furu[str]):
    left: Leaf
    right: Leaf

    def create(self) -> str:
        left = self.left.load_or_create()
        right = self.right.load_or_create()
        return f"pair[{left},{right}]"


class WithComputedDep(Furu[str]):
    name: str

    @furu.dependency
    def base(self) -> Leaf:
        return Leaf(name=self.name)

    def create(self) -> str:
        return f"computed:{self.base.load_or_create()}"


class DynamicParent(Furu[str]):
    name: str
    create_calls: ClassVar[list[str]] = []

    def create(self) -> str:
        type(self).create_calls.append(self.name)
        child = Leaf(name=self.name).load_or_create()
        return f"dynamic[{child}]"


class TwoDynamicChildren(Furu[str]):
    name: str

    def create(self) -> str:
        a = Leaf(name=f"{self.name}-a").load_or_create()
        b = Leaf(name=f"{self.name}-b").load_or_create()
        return f"two[{a};{b}]"


class BatchDynamicChildren(Furu[str]):
    name: str

    def create(self) -> str:
        results = furu.load_or_create(
            [Leaf(name=f"{self.name}-a"), Leaf(name=f"{self.name}-b")]
        )
        return f"batch[{results[0]};{results[1]}]"


class AlwaysSuspends(Furu[str]):
    name: str

    def create(self) -> str:
        # References a brand-new Leaf each call so suspension never converges
        # (used to exercise max_suspensions_per_job).
        import secrets

        return Leaf(name=secrets.token_hex(4)).load_or_create()


class FailingCreate(Furu[str]):
    name: str

    def create(self) -> str:
        raise RuntimeError(f"boom: {self.name}")


@pytest.fixture(autouse=True)
def _reset_call_trackers() -> None:
    DynamicParent.create_calls.clear()


def test_run_local_executor_executes_single_final() -> None:
    leaf = Leaf(name="single")
    scheduler = run_local_executor([leaf])

    assert leaf.load_or_create() == "leaf:single"
    job = scheduler.jobs()[leaf.object_id]
    assert job.state == "completed"


def test_declared_field_dependencies_are_executed_in_order() -> None:
    left = Leaf(name="left")
    right = Leaf(name="right")
    pair = Pair(left=left, right=right)

    scheduler = run_local_executor([pair])

    assert pair.load_or_create() == "pair[leaf:left,leaf:right]"
    assert scheduler.jobs()[left.object_id].state == "completed"
    assert scheduler.jobs()[right.object_id].state == "completed"
    assert scheduler.jobs()[pair.object_id].state == "completed"


def test_computed_dependency_is_scheduled() -> None:
    parent = WithComputedDep(name="x")

    scheduler = run_local_executor([parent])

    assert parent.load_or_create() == "computed:leaf:x"
    assert scheduler.jobs()[Leaf(name="x").object_id].state == "completed"
    assert scheduler.jobs()[parent.object_id].state == "completed"


def test_dynamic_dependency_suspends_and_retries_once_dep_completes() -> None:
    parent = DynamicParent(name="dyn")

    scheduler = run_local_executor([parent])

    assert parent.load_or_create() == "dynamic[leaf:dyn]"
    assert scheduler.jobs()[Leaf(name="dyn").object_id].state == "completed"
    assert scheduler.jobs()[parent.object_id].state == "completed"
    # First call was aborted on suspension; second call after dep computed.
    assert DynamicParent.create_calls == ["dyn", "dyn"]


def test_batch_dynamic_load_or_create_discovers_all_missing_at_once() -> None:
    parent = BatchDynamicChildren(name="b")

    scheduler = run_local_executor([parent])

    assert parent.load_or_create() == "batch[leaf:b-a;leaf:b-b]"
    assert scheduler.jobs()[Leaf(name="b-a").object_id].state == "completed"
    assert scheduler.jobs()[Leaf(name="b-b").object_id].state == "completed"


def test_two_sequential_dynamic_deps_resolve_incrementally() -> None:
    parent = TwoDynamicChildren(name="seq")

    scheduler = run_local_executor([parent])

    assert parent.load_or_create() == "two[leaf:seq-a;leaf:seq-b]"
    assert scheduler.jobs()[Leaf(name="seq-a").object_id].state == "completed"
    assert scheduler.jobs()[Leaf(name="seq-b").object_id].state == "completed"


def test_shared_dependency_between_finals_is_computed_once() -> None:
    shared = Leaf(name="shared")
    a = Pair(left=shared, right=Leaf(name="other-a"))
    b = Pair(left=shared, right=Leaf(name="other-b"))

    scheduler = run_local_executor([a, b])

    # One Job entry per unique object_id (shared appears once).
    assert sum(1 for _ in scheduler.jobs()) == 5
    assert scheduler.jobs()[shared.object_id].state == "completed"


def test_failing_create_marks_job_failed_and_run_raises() -> None:
    obj = FailingCreate(name="bad")

    with pytest.raises(RuntimeError, match="local executor finished"):
        run_local_executor([obj])


def test_max_suspensions_per_job_marks_runaway_job_failed() -> None:
    obj = AlwaysSuspends(name="never")
    scheduler = Scheduler(max_suspensions_per_job=2)

    with pytest.raises(RuntimeError, match="local executor finished"):
        run_local_executor([obj], scheduler=scheduler)

    job = scheduler.jobs()[obj.object_id]
    assert job.state == "failed"
    assert job.suspensions == 3
    assert isinstance(job.error, RuntimeError)
    assert "exceeded max_suspensions_per_job" in str(job.error)


def test_blocked_on_dependencies_only_raises_inside_executor_mode() -> None:
    # Outside executor mode, load_or_create computes inline as before.
    obj = Pair(left=Leaf(name="inline-l"), right=Leaf(name="inline-r"))
    assert obj.load_or_create() == "pair[leaf:inline-l,leaf:inline-r]"


def test_already_cached_artifact_is_recognized_at_claim_time() -> None:
    leaf = Leaf(name="precached")
    assert leaf.load_or_create() == "leaf:precached"  # warm cache

    parent = Pair(left=leaf, right=Leaf(name="other"))
    scheduler = run_local_executor([parent])

    assert scheduler.jobs()[leaf.object_id].state == "completed"
    assert scheduler.jobs()[parent.object_id].state == "completed"


def test_blocked_on_dependencies_exception_carries_deps() -> None:
    deps = (Leaf(name="x"), Leaf(name="y"))
    exc = BlockedOnDependencies(deps)
    assert exc.deps == deps
    assert "blocked on 2 dependencies" in str(exc)


def test_scheduler_submit_registers_full_subgraph() -> None:
    pair = Pair(left=Leaf(name="l"), right=Leaf(name="r"))
    scheduler = Scheduler()
    scheduler.submit([pair])

    job_ids = set(scheduler.jobs())
    assert pair.object_id in job_ids
    assert Leaf(name="l").object_id in job_ids
    assert Leaf(name="r").object_id in job_ids
    assert scheduler.jobs()[pair.object_id].dependencies == {
        Leaf(name="l").object_id,
        Leaf(name="r").object_id,
    }


def test_scheduler_dedupes_identical_artifacts_across_finals() -> None:
    shared = Leaf(name="shared")
    a = Pair(left=shared, right=Leaf(name="x"))
    b = Pair(left=shared, right=Leaf(name="y"))

    scheduler = Scheduler()
    scheduler.submit([a, b])

    assert scheduler.jobs()[shared.object_id] is scheduler.jobs()[shared.object_id]
    assert len(scheduler.jobs()) == 5
