from __future__ import annotations

import threading
from typing import ClassVar

import pytest

import furu
from furu import Furu, run_local
from furu.execution import BlockedOnDependencies, _enter_executor_job, load_or_create
from furu.executor import LocalExecutor
from furu.planner import build_plan
from furu.scheduler import Scheduler, UpstreamFailure


class Leaf(Furu[str]):
    name: str

    def create(self) -> str:
        return f"leaf:{self.name}"


class TwoChild(Furu[str]):
    left: Leaf
    right: Leaf

    def create(self) -> str:
        left = self.left.load_or_create()
        right = self.right.load_or_create()
        return f"{left}|{right}"


class LazyChain(Furu[str]):
    seed: str

    def create(self) -> str:
        inner = Leaf(name=self.seed).load_or_create()
        wrapped = Leaf(name=f"wrap-{inner}").load_or_create()
        return f"chain:{wrapped}"


class ComputedDep(Furu[str]):
    base: str

    @furu.dependency
    def child(self) -> Leaf:
        return Leaf(name=self.base)

    def create(self) -> str:
        return self.child.load_or_create()


class FailingLeaf(Furu[str]):
    name: str

    def create(self) -> str:
        raise RuntimeError(f"boom-{self.name}")


class DependsOnFailing(Furu[str]):
    child: FailingLeaf

    def create(self) -> str:
        return self.child.load_or_create()


class CountingLeaf(Furu[str]):
    label: str
    call_count: ClassVar[dict[str, int]] = {}

    def create(self) -> str:
        type(self).call_count[self.label] = type(self).call_count.get(self.label, 0) + 1
        return f"counted:{self.label}"


class SuspendingParent(Furu[str]):
    label: str
    call_count: ClassVar[dict[str, int]] = {}

    def create(self) -> str:
        type(self).call_count[self.label] = type(self).call_count.get(self.label, 0) + 1
        child = CountingLeaf(label=self.label).load_or_create()
        return f"parent:{child}"


class AlwaysBlocking(Furu[str]):
    seed: int

    def create(self) -> str:
        # Each rerun introduces a brand-new dependency, so the job keeps
        # suspending until max_suspensions is exceeded.
        counter = self.seed
        while True:
            Leaf(name=f"endless-{counter}").load_or_create()
            counter += 1


@pytest.fixture(autouse=True)
def _reset_counters() -> None:
    CountingLeaf.call_count.clear()
    SuspendingParent.call_count.clear()


def test_build_plan_collects_artifacts_and_edges() -> None:
    leaf = Leaf(name="x")
    pair = TwoChild(left=leaf, right=Leaf(name="y"))

    plan = build_plan([pair])

    assert set(plan.artifacts) == {
        pair.object_id,
        leaf.object_id,
        Leaf(name="y").object_id,
    }
    assert plan.edges[pair.object_id] == {leaf.object_id, Leaf(name="y").object_id}
    assert plan.edges[leaf.object_id] == set()


def test_build_plan_dedups_shared_dependencies() -> None:
    shared = Leaf(name="shared")
    pair = TwoChild(left=shared, right=shared)

    plan = build_plan([pair])

    assert set(plan.artifacts) == {pair.object_id, shared.object_id}
    assert plan.edges[pair.object_id] == {shared.object_id}


def test_build_plan_rejects_non_furu_input() -> None:
    with pytest.raises(TypeError, match="expected Furu objects"):
        build_plan(["not-a-furu"])  # ty: ignore[invalid-argument-type]


def test_scheduler_completes_chain_in_dependency_order() -> None:
    a = Leaf(name="a")
    b = Leaf(name="b")
    scheduler = Scheduler()
    scheduler.submit(a)
    scheduler.submit(b, dependencies=[a.object_id])

    first = scheduler.claim_ready()
    assert first is not None and first.artifact is a
    assert scheduler.claim_ready() is None

    scheduler.complete_job(a.object_id)
    second = scheduler.claim_ready()
    assert second is not None and second.artifact is b

    scheduler.complete_job(b.object_id)
    assert scheduler.is_finished()


def test_scheduler_add_dependencies_requeues_running_job() -> None:
    parent = Leaf(name="parent")
    dep = Leaf(name="dep")
    scheduler = Scheduler()
    scheduler.submit(parent)

    job = scheduler.claim_ready()
    assert job is not None and job.artifact is parent

    scheduler.submit(dep)
    scheduler.add_dependencies(parent.object_id, [dep.object_id])

    # Parent is now blocked until dep completes
    next_claim = scheduler.claim_ready()
    assert next_claim is not None and next_claim.artifact is dep

    scheduler.complete_job(dep.object_id)
    again = scheduler.claim_ready()
    assert again is not None and again.artifact is parent
    assert again.suspension_count == 1


def test_scheduler_propagates_failure_to_dependent_queued_jobs() -> None:
    child = Leaf(name="child")
    parent = Leaf(name="parent")
    scheduler = Scheduler()
    scheduler.submit(child)
    scheduler.submit(parent, dependencies=[child.object_id])

    claim = scheduler.claim_ready()
    assert claim is not None and claim.artifact is child
    scheduler.fail_job(child.object_id, RuntimeError("nope"))

    assert scheduler.claim_ready() is None
    assert scheduler.is_finished()
    failures = scheduler.failures()
    assert set(failures) == {child.object_id, parent.object_id}
    assert isinstance(failures[parent.object_id], UpstreamFailure)


def test_load_or_create_inside_executor_context_raises_blocked() -> None:
    missing = Leaf(name="not-yet")
    with _enter_executor_job():
        with pytest.raises(BlockedOnDependencies) as exc:
            load_or_create(missing)
    assert [d.object_id for d in exc.value.deps] == [missing.object_id]
    # The raise happens before mkdir-ing the artifact directory.
    assert not missing._internal_furu_dir.exists()


def test_load_or_create_inside_executor_returns_cached_results_normally() -> None:
    leaf = Leaf(name="cached")
    assert leaf.load_or_create() == "leaf:cached"

    with _enter_executor_job():
        assert load_or_create(leaf) == "leaf:cached"


def test_run_local_single_artifact() -> None:
    assert run_local(Leaf(name="solo")) == "leaf:solo"


def test_run_local_executes_declared_subgraph() -> None:
    pair = TwoChild(left=Leaf(name="l"), right=Leaf(name="r"))

    assert run_local(pair) == "leaf:l|leaf:r"


def test_run_local_handles_lazy_dependencies_via_suspension() -> None:
    obj = LazyChain(seed="seed")

    assert run_local(obj) == "chain:leaf:wrap-leaf:seed"


def test_run_local_handles_at_furu_dependency() -> None:
    obj = ComputedDep(base="base")

    assert run_local(obj) == "leaf:base"


def test_run_local_returns_list_when_given_sequence() -> None:
    leaves = [Leaf(name="a"), Leaf(name="b"), Leaf(name="c")]

    results = run_local(leaves)

    assert results == ["leaf:a", "leaf:b", "leaf:c"]


def test_run_local_reuses_cached_artifacts_for_shared_dependencies() -> None:
    shared = CountingLeaf(label="shared")
    pair = TwoChild(left=Leaf(name="x"), right=Leaf(name="y"))  # warmup unrelated
    run_local(pair)

    parents = [
        SuspendingParent(label="shared"),
        SuspendingParent(label="shared"),  # same artifact, dedup
    ]
    run_local(parents)

    assert CountingLeaf.call_count[shared.label] == 1
    assert SuspendingParent.call_count["shared"] >= 1


def test_run_local_propagates_failure_as_runtime_error() -> None:
    parent = DependsOnFailing(child=FailingLeaf(name="x"))

    with pytest.raises(RuntimeError, match="failed job"):
        run_local(parent)


def test_run_local_with_many_workers_completes_independent_jobs() -> None:
    leaves = [Leaf(name=f"l{i}") for i in range(16)]

    assert run_local(leaves, max_workers=8) == [f"leaf:l{i}" for i in range(16)]


def test_max_suspensions_failure_is_reported_clearly() -> None:
    executor = LocalExecutor(max_workers=2, max_suspensions=3)
    obj = AlwaysBlocking(seed=0)

    with pytest.raises(RuntimeError, match="exceeded max_suspensions"):
        executor.run([obj])


def test_executor_does_not_use_public_load_or_create_for_claimed_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method_calls: list[Furu[str]] = []
    original = Furu.load_or_create

    def tracking_method(self: Furu[str], use_lock: bool = True) -> str:
        method_calls.append(self)
        return original(self, use_lock=use_lock)

    monkeypatch.setattr(Furu, "load_or_create", tracking_method)

    # The worker should run create() directly for the root rather than calling
    # the root's own load_or_create(). Children's load_or_create() (inside
    # create()) still goes through the method.
    leaf = Leaf(name="solo")
    run_local(leaf)
    assert leaf not in method_calls


def test_scheduler_is_thread_safe_under_concurrent_workers() -> None:
    scheduler = Scheduler()
    leaves = [Leaf(name=f"t{i}") for i in range(32)]
    for leaf in leaves:
        scheduler.submit(leaf)

    completed: set[str] = set()
    completed_lock = threading.Lock()

    def worker() -> None:
        while True:
            if scheduler.is_finished():
                return
            job = scheduler.claim_ready()
            if job is None:
                scheduler.wait_for_change(timeout=0.01)
                continue
            with completed_lock:
                completed.add(job.artifact.object_id)
            scheduler.complete_job(job.artifact.object_id)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert completed == {leaf.object_id for leaf in leaves}
