import time
from uuid import UUID

import pytest
from pydantic import ValidationError

from furu import Furu
from furu.execution.manager import FailedJob, Manager, RunningJob
from furu.metadata import ArtifactSpec
from furu.worker.loop import worker_loop
from furu.worker.protocol import FinishFailedRequest, FinishSuccessRequest, Job


class ManagerLeaf(Furu[int]):
    value: int

    def create(self) -> int:
        return self.value


class ManagerParent(Furu[int]):
    child: ManagerLeaf

    def create(self) -> int:
        return self.child.load_or_create() + 1


class ManagerLazyParent(Furu[int]):
    value: int

    def create(self) -> int:
        return ManagerLeaf(value=self.value).load_or_create() + 1


def test_manager_submit_partitions_ready_and_blocked() -> None:
    leaf = ManagerLeaf(value=1)
    parent = ManagerParent(child=leaf)

    manager = Manager.submit([parent])

    assert set(manager.ready) == {leaf.object_id}
    assert set(manager.blocked) == {parent.object_id}
    assert manager.running == {}


def test_manager_finish_moves_dependents_to_ready() -> None:
    leaf = ManagerLeaf(value=1)
    parent = ManagerParent(child=leaf)
    manager = Manager.submit([parent])

    job = manager.get_job()
    assert isinstance(job, Job)
    assert job.lease_id != leaf.object_id
    assert UUID(job.lease_id).version == 4
    assert set(manager.running) == {job.lease_id}
    running_job = manager.running[job.lease_id]
    assert isinstance(running_job, RunningJob)
    assert running_job.node.obj is leaf

    manager.finish(job.lease_id, FinishSuccessRequest())

    assert manager.running == {}
    assert set(manager.completed) == {leaf.object_id}
    assert set(manager.ready) == {parent.object_id}
    assert manager.blocked == {}


def test_manager_block_discovers_lazy_dependency_and_reruns_parent() -> None:
    parent = ManagerLazyParent(value=2)
    dependency = ManagerLeaf(value=2)
    manager = Manager.submit([parent])

    parent_job = manager.get_job()
    assert isinstance(parent_job, Job)
    assert parent_job.lease_id != parent.object_id

    manager.block(
        parent_job.lease_id,
        [ArtifactSpec.from_furu(dependency)],
    )

    assert set(manager.ready) == {dependency.object_id}
    assert set(manager.blocked) == {parent.object_id}

    dependency_job = manager.get_job()
    assert isinstance(dependency_job, Job)
    manager.finish(dependency_job.lease_id, FinishSuccessRequest())

    assert set(manager.ready) == {parent.object_id}
    assert manager.blocked == {}


def test_manager_uses_new_lease_when_blocked_job_is_released() -> None:
    parent = ManagerLazyParent(value=2)
    dependency = ManagerLeaf(value=2)
    manager = Manager.submit([parent])

    first_parent_job = manager.get_job()
    assert isinstance(first_parent_job, Job)

    manager.block(
        first_parent_job.lease_id,
        [ArtifactSpec.from_furu(dependency)],
    )

    dependency_job = manager.get_job()
    assert isinstance(dependency_job, Job)
    manager.finish(dependency_job.lease_id, FinishSuccessRequest())

    second_parent_job = manager.get_job()
    assert isinstance(second_parent_job, Job)
    assert second_parent_job.lease_id != first_parent_job.lease_id
    assert second_parent_job.artifact.object_id == parent.object_id

    with pytest.raises(KeyError, match="unknown running lease_id"):
        manager.finish(
            first_parent_job.lease_id,
            FinishSuccessRequest(),
        )

    assert set(manager.running) == {second_parent_job.lease_id}


def test_manager_failed_job_finishes_with_error() -> None:
    leaf = ManagerLeaf(value=1)
    manager = Manager.submit([leaf])
    job = manager.get_job()
    assert isinstance(job, Job)

    manager.finish(job.lease_id, FinishFailedRequest(error="boom"))

    assert set(manager.failed) == {leaf.object_id}
    failed_job = manager.failed[leaf.object_id]
    assert isinstance(failed_job, FailedJob)
    assert failed_job.lease_id == job.lease_id
    assert failed_job.node.obj is leaf
    assert failed_job.error == "boom"
    with pytest.raises(RuntimeError, match="failed jobs"):
        manager.raise_for_failure()


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
