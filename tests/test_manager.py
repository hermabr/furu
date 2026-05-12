from typing import ClassVar

import pytest

import furu
from furu import Furu, Manager
from furu.execution.manager import Job
from furu.metadata import ArtifactSpec
from furu.storage_layout import run_log_path_in


class TrackingLeaf(Furu[int]):
    n: int
    create_calls: ClassVar[list[int]] = []

    def create(self) -> int:
        type(self).create_calls.append(self.n)
        return self.n * 2


class TrackingMid(Furu[int]):
    label: str
    child: TrackingLeaf
    create_calls: ClassVar[list[str]] = []

    def create(self) -> int:
        type(self).create_calls.append(self.label)
        return self.child.load_or_create() + 1


class LazyChildLoader(Furu[int]):
    base: int
    create_calls: ClassVar[list[int]] = []

    def create(self) -> int:
        type(self).create_calls.append(self.base)
        return self.base + TrackingLeaf(n=self.base).load_or_create()


class AlwaysFails(Furu[int]):
    name: str

    def create(self) -> int:
        raise RuntimeError(f"intentional failure: {self.name}")


class DependsOnFailing(Furu[int]):
    label: str
    child: AlwaysFails

    def create(self) -> int:
        return self.child.load_or_create() + 1


@pytest.fixture(autouse=True)
def _reset_tracking() -> None:
    TrackingLeaf.create_calls.clear()
    TrackingMid.create_calls.clear()
    LazyChildLoader.create_calls.clear()


# --- Manager state-machine unit tests (no HTTP) ---


def test_from_objs_splits_into_ready_and_blocked():
    leaf = TrackingLeaf(n=1)
    mid = TrackingMid(label="m", child=leaf)

    manager = Manager.from_objs([mid])

    assert set(manager.ready) == {leaf.object_id}
    assert set(manager.blocked) == {mid.object_id}
    assert manager.running == {}
    assert manager.completed == {}
    assert manager.failed == {}


def test_get_job_returns_stop_when_nothing_to_do():
    manager = Manager.from_objs([])
    assert manager.get_job() == "stop"


def test_get_job_then_finish_promotes_dependent():
    leaf = TrackingLeaf(n=1)
    mid = TrackingMid(label="m", child=leaf)
    manager = Manager.from_objs([mid])

    job = manager.get_job()
    assert isinstance(job, Job)
    assert job.artifact.object_id == leaf.object_id
    assert set(manager.running) == {job.lease_id}

    manager.finish(job.lease_id, success=True)

    assert set(manager.ready) == {mid.object_id}
    assert set(manager.completed) == {leaf.object_id}
    assert manager.blocked == {}


def test_get_job_returns_wait_when_only_running_remains():
    leaf = TrackingLeaf(n=1)
    mid = TrackingMid(label="m", child=leaf)
    manager = Manager.from_objs([mid])

    job = manager.get_job()
    assert isinstance(job, Job)
    # mid is still blocked, leaf is running
    assert manager.get_job() == "wait"


def test_get_job_stops_when_only_blocked_remains_after_failure():
    leaf = AlwaysFails(name="bad")
    parent = DependsOnFailing(label="p", child=leaf)
    manager = Manager.from_objs([parent])

    job = manager.get_job()
    assert isinstance(job, Job)
    manager.finish(job.lease_id, success=False)

    # parent stays in blocked (no cascade); manager halts
    assert manager.get_job() == "stop"
    assert set(manager.failed) == {leaf.object_id}
    assert set(manager.blocked) == {parent.object_id}
    assert manager.unresolved_object_ids() == [parent.object_id]


def test_report_blocked_extends_dag_and_links_parent():
    parent = LazyChildLoader(base=7)
    manager = Manager.from_objs([parent])

    job = manager.get_job()
    assert isinstance(job, Job)
    assert job.artifact.object_id == parent.object_id

    dep = TrackingLeaf(n=7)
    manager.report_blocked(job.lease_id, [ArtifactSpec.from_furu(dep)])

    # parent is now blocked on the new dep; dep is ready
    assert set(manager.ready) == {dep.object_id}
    assert set(manager.blocked) == {parent.object_id}
    assert manager.running == {}


def test_report_blocked_with_already_completed_dep_re_readies_parent():
    parent = LazyChildLoader(base=7)
    manager = Manager.from_objs([parent])

    parent_job = manager.get_job()
    assert isinstance(parent_job, Job)
    dep = TrackingLeaf(n=7)
    manager.report_blocked(parent_job.lease_id, [ArtifactSpec.from_furu(dep)])

    dep_job = manager.get_job()
    assert isinstance(dep_job, Job)
    assert dep_job.artifact.object_id == dep.object_id
    manager.finish(dep_job.lease_id, success=True)

    # parent should now be in ready
    assert set(manager.ready) == {parent.object_id}


# --- End-to-end tests through Manager.submit() (FastAPI + worker threads) ---


def test_submit_runs_single_zero_dependency_node():
    leaf = TrackingLeaf(n=3)

    manager = Manager.submit([leaf], n_workers=1)

    assert TrackingLeaf.create_calls == [3]
    assert leaf.status() == "completed"
    assert leaf.load_or_create() == 6
    assert set(manager.completed) == {leaf.object_id}
    assert manager.blocked == {} and manager.failed == {}


def test_submit_runs_static_dependencies_in_order():
    leaf = TrackingLeaf(n=4)
    mid = TrackingMid(label="m", child=leaf)

    Manager.submit([mid], n_workers=1)

    assert TrackingLeaf.create_calls == [4]
    assert TrackingMid.create_calls == ["m"]
    assert mid.load_or_create() == 9


def test_submit_handles_shared_dependency_only_once():
    shared = TrackingLeaf(n=5)
    left = TrackingMid(label="L", child=shared)
    right = TrackingMid(label="R", child=shared)

    Manager.submit([left, right], n_workers=1)

    assert TrackingLeaf.create_calls == [5]
    assert sorted(TrackingMid.create_calls) == ["L", "R"]


def test_submit_discovers_lazy_dependencies_and_reruns_parent():
    parent = LazyChildLoader(base=7)

    Manager.submit([parent], n_workers=1)

    assert TrackingLeaf.create_calls == [7]
    assert LazyChildLoader.create_calls == [7, 7]
    assert parent.load_or_create() == 21
    parent_log = run_log_path_in(parent.data_dir).read_text(encoding="utf-8")
    assert (
        "load_or_create deferred: load_or_create discovered "
        "1 missing dependency/dependencies" in parent_log
    )


def test_submit_skips_already_completed_objects():
    leaf = TrackingLeaf(n=8)
    leaf.load_or_create()
    TrackingLeaf.create_calls.clear()
    mid = TrackingMid(label="cached-child", child=leaf)

    Manager.submit([mid], n_workers=1)

    assert TrackingLeaf.create_calls == []
    assert TrackingMid.create_calls == ["cached-child"]


def test_submit_empty_list_is_noop():
    manager = Manager.submit([], n_workers=1)
    assert manager.completed == {}
    assert manager.blocked == {}
    assert manager.failed == {}


def test_submit_with_multiple_workers_runs_independent_nodes():
    leaves = [TrackingLeaf(n=i) for i in range(8)]

    Manager.submit(leaves, n_workers=4)

    assert sorted(TrackingLeaf.create_calls) == list(range(8))
    for leaf in leaves:
        assert leaf.status() == "completed"


def test_submit_reports_failure_in_completed_state():
    failing = AlwaysFails(name="boom")
    parent = DependsOnFailing(label="p", child=failing)

    manager = furu.Manager.submit([parent], n_workers=1)

    assert failing.object_id in manager.failed
    # parent is unresolved because its dep failed
    assert parent.object_id in manager.blocked
    assert manager.unresolved_object_ids() == [parent.object_id]
