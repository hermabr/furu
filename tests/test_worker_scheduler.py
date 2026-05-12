import time
from uuid import UUID

import pytest
from pydantic import ValidationError

from furu import Furu
from furu.execution.scheduler import FailedJob, Scheduler, RunningJob
from furu.metadata import ArtifactSpec
from furu.worker.loop import worker_loop
from furu.worker.protocol import FinishFailedRequest, FinishSuccessRequest, Job


class SchedulerLeaf(Furu[int]):
    value: int

    def create(self) -> int:
        return self.value


class SchedulerParent(Furu[int]):
    child: SchedulerLeaf

    def create(self) -> int:
        return self.child.load_or_create() + 1


class SchedulerLazyParent(Furu[int]):
    value: int

    def create(self) -> int:
        return SchedulerLeaf(value=self.value).load_or_create() + 1


def test_scheduler_init_partitions_ready_and_blocked() -> None:
    leaf = SchedulerLeaf(value=1)
    parent = SchedulerParent(child=leaf)

    scheduler = Scheduler([parent])

    assert set(scheduler.ready) == {leaf.object_id}
    assert set(scheduler.blocked) == {parent.object_id}
    assert scheduler.running == {}


def test_scheduler_finish_moves_dependents_to_ready() -> None:
    leaf = SchedulerLeaf(value=1)
    parent = SchedulerParent(child=leaf)
    scheduler = Scheduler([parent])

    job = scheduler.get_job()
    assert isinstance(job, Job)
    assert job.lease_id != leaf.object_id
    assert UUID(job.lease_id).version == 4
    assert set(scheduler.running) == {job.lease_id}
    running_job = scheduler.running[job.lease_id]
    assert isinstance(running_job, RunningJob)
    assert running_job.node.obj is leaf

    scheduler.finish(job.lease_id, FinishSuccessRequest())

    assert scheduler.running == {}
    assert set(scheduler.completed) == {leaf.object_id}
    assert set(scheduler.ready) == {parent.object_id}
    assert scheduler.blocked == {}


def test_scheduler_block_discovers_lazy_dependency_and_reruns_parent() -> None:
    parent = SchedulerLazyParent(value=2)
    dependency = SchedulerLeaf(value=2)
    scheduler = Scheduler([parent])

    parent_job = scheduler.get_job()
    assert isinstance(parent_job, Job)
    assert parent_job.lease_id != parent.object_id

    scheduler.block(
        parent_job.lease_id,
        [ArtifactSpec.from_furu(dependency)],
    )

    assert set(scheduler.ready) == {dependency.object_id}
    assert set(scheduler.blocked) == {parent.object_id}

    dependency_job = scheduler.get_job()
    assert isinstance(dependency_job, Job)
    scheduler.finish(dependency_job.lease_id, FinishSuccessRequest())

    assert set(scheduler.ready) == {parent.object_id}
    assert scheduler.blocked == {}


def test_scheduler_block_discovers_multiple_lazy_dependencies_together() -> None:
    parent = SchedulerLazyParent(value=2)
    dependencies = [SchedulerLeaf(value=2), SchedulerLeaf(value=3)]
    scheduler = Scheduler([parent])

    parent_job = scheduler.get_job()
    assert isinstance(parent_job, Job)

    scheduler.block(
        parent_job.lease_id,
        [ArtifactSpec.from_furu(dependency) for dependency in dependencies],
    )

    assert set(scheduler.ready) == {dependency.object_id for dependency in dependencies}
    assert set(scheduler.blocked) == {parent.object_id}

    parent_node = scheduler.nodes_by_id[parent.object_id]
    assert {node.obj.object_id for node in parent_node.dependencies} == {
        dependency.object_id for dependency in dependencies
    }
    for dependency in dependencies:
        dependency_node = scheduler.nodes_by_id[dependency.object_id]
        assert parent_node in dependency_node.dependents


def test_scheduler_uses_new_lease_when_blocked_job_is_released() -> None:
    parent = SchedulerLazyParent(value=2)
    dependency = SchedulerLeaf(value=2)
    scheduler = Scheduler([parent])

    first_parent_job = scheduler.get_job()
    assert isinstance(first_parent_job, Job)

    scheduler.block(
        first_parent_job.lease_id,
        [ArtifactSpec.from_furu(dependency)],
    )

    dependency_job = scheduler.get_job()
    assert isinstance(dependency_job, Job)
    scheduler.finish(dependency_job.lease_id, FinishSuccessRequest())

    second_parent_job = scheduler.get_job()
    assert isinstance(second_parent_job, Job)
    assert second_parent_job.lease_id != first_parent_job.lease_id
    assert second_parent_job.artifact.object_id == parent.object_id

    with pytest.raises(KeyError, match="unknown running lease_id"):
        scheduler.finish(
            first_parent_job.lease_id,
            FinishSuccessRequest(),
        )

    assert set(scheduler.running) == {second_parent_job.lease_id}


def test_scheduler_failed_job_finishes_with_error() -> None:
    leaf = SchedulerLeaf(value=1)
    scheduler = Scheduler([leaf])
    job = scheduler.get_job()
    assert isinstance(job, Job)

    scheduler.finish(job.lease_id, FinishFailedRequest(error="boom"))

    assert set(scheduler.failed) == {leaf.object_id}
    failed_job = scheduler.failed[leaf.object_id]
    assert isinstance(failed_job, FailedJob)
    assert failed_job.lease_id == job.lease_id
    assert failed_job.node.obj is leaf
    assert failed_job.error == "boom"
    with pytest.raises(RuntimeError, match="failed jobs"):
        scheduler.raise_for_failure()


def test_finish_request_requires_error_for_failed_status() -> None:
    with pytest.raises(ValidationError, match="Field required"):
        FinishFailedRequest.model_validate({})


def test_worker_loop_exits_when_server_is_unavailable() -> None:
    started = time.monotonic()

    worker_loop(
        server_url="http://127.0.0.1:1",
        unavailable_timeout=0.01,
        retry_interval=0.01,
    )

    assert time.monotonic() - started < 1
