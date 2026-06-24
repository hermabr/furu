import hashlib
import logging
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import TypeAdapter, ValidationError

import furu.worker.loop as worker_loop_module
from furu import Furu
from furu._storage_layout import execution_coordinator_log_path_in
from furu.config import get_config
from furu.dag import _add_to_dag
from furu.execution import api
from furu.execution.api import create_execution_coordinator_api_app
from furu.execution.execution_coordinator import (
    ExecutionCoordinator,
    FailedJob,
    RunningJob,
)
from furu.execution.server import execution_coordinator_server
from furu.metadata import ArtifactSpec
from furu.resources import ResourceRequest, ResourceRequirements
from furu.worker.backends.local import LocalThreadWorkerBackend, LocalThreadWorkerPool
from furu.worker.loop import worker_loop
from furu.worker.protocol import (
    Job,
    JobBlockedResult,
    JobCompletedResult,
    JobFailedResult,
    JobResultRequest,
    LeaseJobResponse,
)

ANY_RESOURCES = ResourceRequest()


@contextmanager
def _captured_furu_logs(caplog: pytest.LogCaptureFixture) -> Iterator[None]:
    furu_logger = logging.getLogger("furu")
    furu_logger.addHandler(caplog.handler)
    try:
        caplog.set_level(logging.INFO, logger="furu")
        yield
    finally:
        furu_logger.removeHandler(caplog.handler)


def _new_execution_coordinator(
    objs: Sequence[Furu[Any]],
    *,
    max_retries_per_object: int | None = None,
) -> ExecutionCoordinator:
    if max_retries_per_object is None:
        max_retries_per_object = get_config().worker.max_retries_per_object
    coordinator = ExecutionCoordinator(max_retries_per_object=max_retries_per_object)
    _add_to_dag(coordinator, objs)
    digest = hashlib.blake2s(digest_size=16)
    for obj in objs:
        digest.update(obj.object_id.encode("utf-8"))
        digest.update(b"\0")
    coordinator.executor_id = digest.hexdigest()
    return coordinator


def _new_local_pool(
    *,
    server_url: str = "http://execution-coordinator.test",
    auth_token: str = "secret",
    max_workers: int = 1,
    max_failed_restarts: int = 3,
    resource_request: ResourceRequest | None = None,
    scale_interval: float = 1.0,
) -> LocalThreadWorkerPool:
    pool_holder: list[LocalThreadWorkerPool] = []
    pool = LocalThreadWorkerPool(
        _server_url=server_url,
        _auth_token=auth_token,
        _max_workers=max_workers,
        _max_failed_restarts=max_failed_restarts,
        _resource_request=resource_request or ResourceRequest(),
        _scale_interval=scale_interval,
        _worker_idle_timeout=get_config().worker.idle_timeout_seconds,
        _client=api.PoolApiClient(server_url=server_url, auth_token=auth_token),
        _stop_event=threading.Event(),
        _unhealthy_event=threading.Event(),
        _scale_thread=threading.Thread(
            target=lambda: pool_holder[0]._scale_loop(),
            name="furu-local-worker-pool-scale",
        ),
        _threads=[],
        _failed_threads=[],
    )
    pool_holder.append(pool)
    return pool


class ExecutionCoordinatorLeaf(Furu[int]):
    value: int

    def create(self) -> int:
        return self.value


class FlakyExecutionCoordinatorLeaf(Furu[int]):
    value: int
    attempts_by_value: ClassVar[dict[int, int]] = {}

    def create(self) -> int:
        attempts = type(self).attempts_by_value.get(self.value, 0) + 1
        type(self).attempts_by_value[self.value] = attempts
        if attempts == 1:
            raise RuntimeError(f"temporary failure: {self.value}")
        return self.value


class LimitedExecutionCoordinatorLeaf(Furu[int]):
    value: int
    max_workers: ClassVar[int | None] = 2

    def create(self) -> int:
        return self.value


class ExecutionCoordinatorParent(Furu[int]):
    child: ExecutionCoordinatorLeaf

    def create(self) -> int:
        return self.child.create() + 1


class ExecutionCoordinatorLazyParent(Furu[int]):
    value: int

    def create(self) -> int:
        return ExecutionCoordinatorLeaf(value=self.value).create() + 1


def test_execution_coordinator_init_partitions_ready_and_blocked() -> None:
    leaf = ExecutionCoordinatorLeaf(value=1)
    parent = ExecutionCoordinatorParent(child=leaf)

    coordinator = _new_execution_coordinator([parent])

    assert set(coordinator.ready) == {leaf.object_id}
    assert set(coordinator.blocked) == {parent.object_id}
    assert coordinator.running == {}


def test_execution_coordinator_executor_id_is_stable_hash_of_root_object_tuple() -> (
    None
):
    left = ExecutionCoordinatorLeaf(value=1)
    right = ExecutionCoordinatorLeaf(value=2)

    coordinator = _new_execution_coordinator([left, right])

    assert len(coordinator.executor_id) == 32
    assert int(coordinator.executor_id, 16) >= 0
    assert (
        _new_execution_coordinator([left, right]).executor_id == coordinator.executor_id
    )
    assert (
        _new_execution_coordinator([right, left]).executor_id != coordinator.executor_id
    )
    assert (
        coordinator.executor_dir
        == get_config().run_directories.executions / coordinator.executor_id
    )


def test_execution_coordinator_max_retries_per_object_defaults_to_config() -> None:
    coordinator = _new_execution_coordinator([ExecutionCoordinatorLeaf(value=1)])

    assert (
        coordinator.max_retries_per_object == get_config().worker.max_retries_per_object
    )


def test_execution_coordinator_job_result_completed_moves_dependents_to_ready() -> None:
    leaf = ExecutionCoordinatorLeaf(value=1)
    parent = ExecutionCoordinatorParent(child=leaf)
    coordinator = _new_execution_coordinator([parent])

    job = coordinator.lease_job(resources=ANY_RESOURCES)
    assert isinstance(job, Job)
    assert job.lease_id != leaf.object_id
    assert UUID(job.lease_id).version == 4
    assert set(coordinator.running) == {job.lease_id}
    running_job = coordinator.running[job.lease_id]
    assert isinstance(running_job, RunningJob)
    assert running_job.node.obj is leaf

    coordinator.job_result(job.lease_id, JobCompletedResult())

    assert coordinator.running == {}
    assert set(coordinator.completed) == {leaf.object_id}
    assert set(coordinator.ready) == {parent.object_id}
    assert coordinator.blocked == {}


def test_execution_coordinator_lease_job_returns_wait_when_only_running_jobs_can_unblock_work() -> (
    None
):
    leaf = ExecutionCoordinatorLeaf(value=1)
    parent = ExecutionCoordinatorParent(child=leaf)
    coordinator = _new_execution_coordinator([parent])

    job = coordinator.lease_job(resources=ANY_RESOURCES)
    assert isinstance(job, Job)

    assert coordinator.lease_job(resources=ANY_RESOURCES) == "wait"
    assert not coordinator.done.is_set()


def test_execution_coordinator_job_result_blocked_discovers_lazy_dependency_and_reruns_parent() -> (
    None
):
    parent = ExecutionCoordinatorLazyParent(value=2)
    dependency = ExecutionCoordinatorLeaf(value=2)
    coordinator = _new_execution_coordinator([parent])

    parent_job = coordinator.lease_job(resources=ANY_RESOURCES)
    assert isinstance(parent_job, Job)
    assert parent_job.lease_id != parent.object_id

    coordinator.job_result(
        parent_job.lease_id,
        JobBlockedResult(dependencies=[ArtifactSpec.from_furu(dependency)]),
    )

    assert set(coordinator.ready) == {dependency.object_id}
    assert set(coordinator.blocked) == {parent.object_id}

    dependency_job = coordinator.lease_job(resources=ANY_RESOURCES)
    assert isinstance(dependency_job, Job)
    coordinator.job_result(dependency_job.lease_id, JobCompletedResult())

    assert set(coordinator.ready) == {parent.object_id}
    assert coordinator.blocked == {}


def test_execution_coordinator_job_result_blocked_ignores_completed_lazy_dependency() -> (
    None
):
    parent = ExecutionCoordinatorLazyParent(value=2)
    dependency = ExecutionCoordinatorLeaf(value=2)
    dependency.create()
    coordinator = _new_execution_coordinator([parent])

    parent_job = coordinator.lease_job(resources=ANY_RESOURCES)
    assert isinstance(parent_job, Job)

    coordinator.job_result(
        parent_job.lease_id,
        JobBlockedResult(dependencies=[ArtifactSpec.from_furu(dependency)]),
    )

    assert set(coordinator.ready) == {parent.object_id}
    assert coordinator.blocked == {}
    assert dependency.object_id not in coordinator.nodes_by_id


def test_execution_coordinator_job_result_blocked_discovers_multiple_lazy_dependencies_together() -> (
    None
):
    parent = ExecutionCoordinatorLazyParent(value=2)
    dependencies = [
        ExecutionCoordinatorLeaf(value=2),
        ExecutionCoordinatorLeaf(value=3),
    ]
    coordinator = _new_execution_coordinator([parent])

    parent_job = coordinator.lease_job(resources=ANY_RESOURCES)
    assert isinstance(parent_job, Job)

    coordinator.job_result(
        parent_job.lease_id,
        JobBlockedResult(
            dependencies=[
                ArtifactSpec.from_furu(dependency) for dependency in dependencies
            ]
        ),
    )

    assert set(coordinator.ready) == {
        dependency.object_id for dependency in dependencies
    }
    assert set(coordinator.blocked) == {parent.object_id}

    parent_node = coordinator.nodes_by_id[parent.object_id]
    assert {node.obj.object_id for node in parent_node.dependencies} == {
        dependency.object_id for dependency in dependencies
    }
    for dependency in dependencies:
        dependency_node = coordinator.nodes_by_id[dependency.object_id]
        assert parent_node in dependency_node.dependents


def test_execution_coordinator_uses_new_lease_when_blocked_job_is_released() -> None:
    parent = ExecutionCoordinatorLazyParent(value=2)
    dependency = ExecutionCoordinatorLeaf(value=2)
    coordinator = _new_execution_coordinator([parent])

    first_parent_job = coordinator.lease_job(resources=ANY_RESOURCES)
    assert isinstance(first_parent_job, Job)

    coordinator.job_result(
        first_parent_job.lease_id,
        JobBlockedResult(dependencies=[ArtifactSpec.from_furu(dependency)]),
    )

    dependency_job = coordinator.lease_job(resources=ANY_RESOURCES)
    assert isinstance(dependency_job, Job)
    coordinator.job_result(dependency_job.lease_id, JobCompletedResult())

    second_parent_job = coordinator.lease_job(resources=ANY_RESOURCES)
    assert isinstance(second_parent_job, Job)
    assert second_parent_job.lease_id != first_parent_job.lease_id
    assert second_parent_job.artifact.object_id == parent.object_id

    assert set(coordinator.running) == {second_parent_job.lease_id}
    assert set(coordinator.completed) == {dependency.object_id}


def test_execution_coordinator_job_result_failed_finishes_with_error() -> None:
    leaf = ExecutionCoordinatorLeaf(value=1)
    coordinator = _new_execution_coordinator([leaf], max_retries_per_object=0)
    job = coordinator.lease_job(resources=ANY_RESOURCES)
    assert isinstance(job, Job)

    coordinator.job_result(job.lease_id, JobFailedResult(error="boom"))

    assert coordinator.running == {}
    assert set(coordinator.failed) == {leaf.object_id}
    failed_job = coordinator.failed[leaf.object_id]
    assert failed_job.failed_attempts == 1
    assert isinstance(failed_job, FailedJob)
    assert failed_job.lease_id == job.lease_id
    assert failed_job.node.obj is leaf
    assert failed_job.error == "boom"
    log_text = execution_coordinator_log_path_in(coordinator.executor_dir).read_text(
        encoding="utf-8"
    )
    assert f"failed {leaf._log_label}" in log_text
    assert "will retry" not in log_text
    assert "boom" in log_text
    assert "furu execution coordinator finished with error" in log_text
    with pytest.raises(RuntimeError, match="failed jobs"):
        coordinator.raise_for_failure()


def test_execution_coordinator_job_result_failed_retries_before_finishing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    leaf = ExecutionCoordinatorLeaf(value=1)
    coordinator = _new_execution_coordinator([leaf], max_retries_per_object=2)

    first_job = coordinator.lease_job(resources=ANY_RESOURCES)
    assert isinstance(first_job, Job)
    with _captured_furu_logs(caplog):
        coordinator.job_result(first_job.lease_id, JobFailedResult(error="boom 1"))

    assert set(coordinator.failed) == {leaf.object_id}
    failed_job = coordinator.failed[leaf.object_id]
    assert failed_job.failed_attempts == 1
    assert failed_job.lease_id == first_job.lease_id
    assert failed_job.error == "boom 1"
    assert set(coordinator.ready) == {leaf.object_id}
    assert not coordinator.done.is_set()

    second_job = coordinator.lease_job(resources=ANY_RESOURCES)
    assert isinstance(second_job, Job)
    assert second_job.lease_id != first_job.lease_id
    with _captured_furu_logs(caplog):
        coordinator.job_result(second_job.lease_id, JobFailedResult(error="boom 2"))

    assert set(coordinator.failed) == {leaf.object_id}
    failed_job = coordinator.failed[leaf.object_id]
    assert failed_job.failed_attempts == 2
    assert failed_job.lease_id == second_job.lease_id
    assert failed_job.error == "boom 2"
    assert set(coordinator.ready) == {leaf.object_id}
    assert not coordinator.done.is_set()

    third_job = coordinator.lease_job(resources=ANY_RESOURCES)
    assert isinstance(third_job, Job)
    coordinator.job_result(third_job.lease_id, JobFailedResult(error="boom 3"))

    assert coordinator.running == {}
    assert coordinator.ready == {}
    assert set(coordinator.failed) == {leaf.object_id}
    failed_job = coordinator.failed[leaf.object_id]
    assert failed_job.failed_attempts == 3
    assert failed_job.lease_id == third_job.lease_id
    assert failed_job.error == "boom 3"
    assert coordinator.done.is_set()
    log_text = execution_coordinator_log_path_in(coordinator.executor_dir).read_text(
        encoding="utf-8"
    )
    assert log_text.count("will retry") == 2
    assert any(
        "will retry" in message and "boom 1" in message for message in caplog.messages
    )
    assert any(
        "will retry" in message and "boom 2" in message for message in caplog.messages
    )
    assert f"lease={third_job.lease_id}" in log_text
    assert "failed_retry=1 failed=0" in log_text
    assert "failed_retry=0 failed=1" in log_text


def test_execution_coordinator_job_result_failed_retry_can_later_complete() -> None:
    leaf = ExecutionCoordinatorLeaf(value=1)
    coordinator = _new_execution_coordinator([leaf], max_retries_per_object=1)

    first_job = coordinator.lease_job(resources=ANY_RESOURCES)
    assert isinstance(first_job, Job)
    coordinator.job_result(first_job.lease_id, JobFailedResult(error="boom"))

    failed_job = coordinator.failed[leaf.object_id]
    assert failed_job.failed_attempts == 1
    assert failed_job.lease_id == first_job.lease_id

    retry_job = coordinator.lease_job(resources=ANY_RESOURCES)
    assert isinstance(retry_job, Job)
    coordinator.job_result(retry_job.lease_id, JobCompletedResult())

    assert coordinator.failed == {}
    assert set(coordinator.completed) == {leaf.object_id}
    assert coordinator.done.is_set()


def test_execution_coordinator_run_retries_failed_worker_result() -> None:
    FlakyExecutionCoordinatorLeaf.attempts_by_value.clear()
    value = uuid4().int
    leaf = FlakyExecutionCoordinatorLeaf(value=value)
    objs = [leaf]

    returned = ExecutionCoordinator.run(
        objs,
        max_retries_per_object=1,
        worker_backends=(LocalThreadWorkerBackend(),),
    )

    assert FlakyExecutionCoordinatorLeaf.attempts_by_value == {value: 2}
    assert returned is objs
    assert leaf.create() == value


class GpuLeaf(Furu[int]):
    value: int

    @property
    def resource_requirements(self) -> ResourceRequirements | None:
        return ResourceRequirements(gpus=(1, None))

    def create(self) -> int:
        return self.value


class CpuOnlyLeaf(Furu[int]):
    value: int

    @property
    def resource_requirements(self) -> ResourceRequirements | None:
        return ResourceRequirements(gpus=(0, 0))

    def create(self) -> int:
        return self.value


class MemoryLeaf(Furu[int]):
    value: int

    @property
    def resource_requirements(self) -> ResourceRequirements | None:
        return ResourceRequirements(memory_gib=(8, None))

    def create(self) -> int:
        return self.value


class DynamicCpuSeed(Furu[int]):
    value: int
    create_calls: ClassVar[list[int]] = []

    @property
    def resource_requirements(self) -> ResourceRequirements | None:
        return ResourceRequirements(gpus=(0, 0))

    def create(self) -> int:
        type(self).create_calls.append(self.value)
        return self.value


class DynamicGpuAfterSeed(Furu[int]):
    parent: DynamicCpuSeed
    value: int
    create_calls: ClassVar[list[int]] = []

    @property
    def resource_requirements(self) -> ResourceRequirements | None:
        return ResourceRequirements(gpus=(1, None))

    def create(self) -> int:
        type(self).create_calls.append(self.value)
        return self.parent.create() + self.value


class DynamicCpuAfterGpu(Furu[int]):
    parent: DynamicGpuAfterSeed
    value: int
    create_calls: ClassVar[list[int]] = []

    @property
    def resource_requirements(self) -> ResourceRequirements | None:
        return ResourceRequirements(gpus=(0, 0))

    def create(self) -> int:
        type(self).create_calls.append(self.value)
        return self.parent.create() + self.value


class DynamicGpuAfterCpu(Furu[int]):
    parent: DynamicCpuAfterGpu
    value: int
    create_calls: ClassVar[list[int]] = []

    @property
    def resource_requirements(self) -> ResourceRequirements | None:
        return ResourceRequirements(gpus=(1, None))

    def create(self) -> int:
        type(self).create_calls.append(self.value)
        return self.parent.create() + self.value


def test_count_satisfiable_jobs_caps_at_max_workers_and_filters_by_requirements() -> (
    None
):
    coordinator = _new_execution_coordinator(
        [
            ExecutionCoordinatorLeaf(value=1),
            ExecutionCoordinatorLeaf(value=2),
            GpuLeaf(value=3),
        ]
    )

    assert (
        coordinator.count_satisfiable_jobs(resources=ResourceRequest(), max_workers=10)
        == 2
    )
    assert (
        coordinator.count_satisfiable_jobs(resources=ResourceRequest(), max_workers=1)
        == 1
    )
    assert (
        coordinator.count_satisfiable_jobs(
            resources=ResourceRequest(gpus=1), max_workers=10
        )
        == 3
    )


def test_worker_cap_limits_satisfiable_jobs_and_leases() -> None:
    limited = [LimitedExecutionCoordinatorLeaf(value=value) for value in range(3)]
    uncapped = ExecutionCoordinatorLeaf(value=10)
    coordinator = _new_execution_coordinator([*limited, uncapped])

    assert (
        coordinator.count_satisfiable_jobs(resources=ResourceRequest(), max_workers=10)
        == 3
    )

    first = coordinator.lease_job(resources=ResourceRequest())
    second = coordinator.lease_job(resources=ResourceRequest())
    third = coordinator.lease_job(resources=ResourceRequest())

    assert isinstance(first, Job)
    assert isinstance(second, Job)
    assert isinstance(third, Job)
    limited_ids = {obj.object_id for obj in limited}
    leased_limited_ids = {first.artifact.object_id, second.artifact.object_id}
    assert leased_limited_ids < limited_ids
    assert third.artifact.object_id == uncapped.object_id
    assert (
        coordinator.count_satisfiable_jobs(resources=ResourceRequest(), max_workers=10)
        == 0
    )
    assert coordinator.lease_job(resources=ResourceRequest()) == "wait"

    coordinator.job_result(first.lease_id, JobCompletedResult())
    fourth = coordinator.lease_job(resources=ResourceRequest())

    assert isinstance(fourth, Job)
    assert fourth.artifact.object_id in limited_ids - leased_limited_ids


def test_lease_job_filters_by_worker_resources() -> None:
    cpu_leaf = CpuOnlyLeaf(value=1)
    gpu_leaf = GpuLeaf(value=2)
    coordinator = _new_execution_coordinator([cpu_leaf, gpu_leaf])

    cpu_job = coordinator.lease_job(resources=ResourceRequest(gpus=0))
    assert isinstance(cpu_job, Job)
    assert cpu_job.artifact.object_id == cpu_leaf.object_id

    assert coordinator.lease_job(resources=ResourceRequest(gpus=0)) == "wait"

    gpu_job = coordinator.lease_job(resources=ResourceRequest(gpus=1))
    assert isinstance(gpu_job, Job)
    assert gpu_job.artifact.object_id == gpu_leaf.object_id


def test_lease_job_filters_by_worker_memory_gib() -> None:
    memory_leaf = MemoryLeaf(value=1)
    coordinator = _new_execution_coordinator([memory_leaf])

    assert coordinator.lease_job(resources=ResourceRequest(memory_gib=7)) == "wait"

    memory_job = coordinator.lease_job(resources=ResourceRequest(memory_gib=8))
    assert isinstance(memory_job, Job)
    assert memory_job.artifact.object_id == memory_leaf.object_id


def test_execution_coordinator_run_dynamically_allocates_local_workers_for_later_resource_stages() -> (
    None
):
    for cls in (
        DynamicCpuSeed,
        DynamicGpuAfterSeed,
        DynamicCpuAfterGpu,
        DynamicGpuAfterCpu,
    ):
        cls.create_calls.clear()

    seed_value = uuid4().int
    seed = DynamicCpuSeed(value=seed_value)
    first_gpu = DynamicGpuAfterSeed(parent=seed, value=20)
    second_cpu = DynamicCpuAfterGpu(parent=first_gpu, value=30)
    final_gpus = [
        DynamicGpuAfterCpu(parent=second_cpu, value=value) for value in range(4)
    ]

    ExecutionCoordinator.run(
        final_gpus,
        worker_backends=(
            LocalThreadWorkerBackend(
                max_workers=1,
                resource_request=ResourceRequest(gpus=0),
            ),
            LocalThreadWorkerBackend(
                max_workers=3,
                resource_request=ResourceRequest(gpus=1),
            ),
        ),
    )

    assert DynamicCpuSeed.create_calls == [seed_value]
    assert DynamicGpuAfterSeed.create_calls == [20]
    assert DynamicCpuAfterGpu.create_calls == [30]
    assert sorted(DynamicGpuAfterCpu.create_calls) == [0, 1, 2, 3]
    assert [obj.create() for obj in final_gpus] == [
        seed_value + 50 + value for value in range(4)
    ]


def test_local_pool_scale_spawns_workers_up_to_max_as_satisfiable_count_grows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = iter([0, 2, 10, 10])
    release_workers = threading.Event()

    def fake_count(
        self: api.PoolApiClient, *, resources: object, max_workers: int
    ) -> int:
        return min(next(counts), max_workers)

    monkeypatch.setattr(api.PoolApiClient, "count_satisfiable_jobs", fake_count)
    monkeypatch.setattr(
        worker_loop_module,
        "worker_loop",
        lambda *, server_url, auth_token, resource_request, idle_timeout: (
            release_workers.wait(timeout=5)
        ),
    )

    pool = _new_local_pool(max_workers=3)

    try:
        pool._scale_once()
        assert len(pool._threads) == 0

        pool._scale_once()
        assert len(pool._threads) == 2

        pool._scale_once()
        assert len(pool._threads) == 3

        pool._scale_once()
        assert len(pool._threads) == 3
    finally:
        release_workers.set()
        for thread in pool._threads:
            thread.join(timeout=5)


def test_local_pool_scale_uses_unique_worker_names_after_worker_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_started = threading.Semaphore(0)
    release_workers = threading.Event()
    calls = 0

    def fake_count(
        self: api.PoolApiClient, *, resources: object, max_workers: int
    ) -> int:
        return max_workers

    def worker_loop(
        *,
        server_url: str,
        auth_token: str,
        resource_request: ResourceRequest,
        idle_timeout: float,
        component: str,
    ) -> None:
        nonlocal calls
        calls += 1
        worker_started.release()
        if calls > 1:
            release_workers.wait(timeout=5)

    monkeypatch.setattr(api.PoolApiClient, "count_satisfiable_jobs", fake_count)
    monkeypatch.setattr(worker_loop_module, "worker_loop", worker_loop)

    pool = _new_local_pool(max_workers=3)

    try:
        pool._scale_once()
        for _ in range(3):
            assert worker_started.acquire(timeout=5)
        pool._threads[0].join(timeout=5)

        pool._scale_once()
        assert worker_started.acquire(timeout=5)

        worker_names = [thread.name for thread in pool._threads]
        assert len(set(worker_names)) == 3
        assert all(name.startswith("local-worker-") for name in worker_names)
    finally:
        release_workers.set()
        for thread in pool._threads:
            thread.join(timeout=5)


def test_local_pool_scale_does_not_count_normal_exits_as_restarts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    starts = 0

    def worker_loop(
        *,
        server_url: str,
        auth_token: str,
        resource_request: ResourceRequest,
        idle_timeout: float,
        component: str,
    ) -> None:
        nonlocal starts
        starts += 1

    monkeypatch.setattr(
        api.PoolApiClient,
        "count_satisfiable_jobs",
        lambda self, *, resources, max_workers: max_workers,
    )
    monkeypatch.setattr(worker_loop_module, "worker_loop", worker_loop)

    pool = _new_local_pool(max_workers=1, max_failed_restarts=0)

    for _ in range(3):
        pool._scale_once()
        pool._threads[0].join(timeout=5)

    assert starts == 3
    assert pool._failed_threads == []
    assert not any(thread.is_alive() for thread in pool._threads)


def test_local_pool_scale_counts_crashes_against_restart_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def worker_loop(
        *,
        server_url: str,
        auth_token: str,
        resource_request: ResourceRequest,
        idle_timeout: float,
        component: str,
    ) -> None:
        raise RuntimeError("worker boom")

    monkeypatch.setattr(
        api.PoolApiClient,
        "count_satisfiable_jobs",
        lambda self, *, resources, max_workers: max_workers,
    )
    monkeypatch.setattr(worker_loop_module, "worker_loop", worker_loop)

    pool = _new_local_pool(max_workers=1, max_failed_restarts=1)

    pool._scale_once()
    pool._threads[0].join(timeout=5)
    assert not pool._unhealthy_event.is_set()

    pool._scale_once()
    pool._threads[0].join(timeout=5)
    assert pool._unhealthy_event.is_set()

    pool._scale_once()

    assert len(pool._failed_threads) == 2
    assert pool._threads == []


def test_execution_coordinator_run_fails_when_worker_pool_reports_unhealthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def crashing_worker_loop(
        *,
        server_url: str,
        auth_token: str,
        resource_request: ResourceRequest,
        idle_timeout: float,
        component: str,
    ) -> None:
        raise RuntimeError("worker boom")

    monkeypatch.setattr(worker_loop_module, "worker_loop", crashing_worker_loop)

    with pytest.raises(
        RuntimeError,
        match="local worker pool became unhealthy: RuntimeError: worker boom",
    ):
        ExecutionCoordinator.run(
            [ExecutionCoordinatorLeaf(value=42)],
            worker_backends=(
                LocalThreadWorkerBackend(
                    max_workers=1,
                    max_failed_restarts=0,
                    resource_request=ResourceRequest(),
                    scale_interval=0.05,
                ),
            ),
        )


def test_execution_coordinator_run_uses_worker_backend() -> None:
    class RecordingBackend:
        execution_coordinator_listen_host = "0.0.0.0"

        def __init__(self) -> None:
            self.bound_ports: list[int] = []
            self.auth_tokens: list[str] = []

        def start_pool(
            self,
            *,
            bound_port: int,
            auth_token: str,
            executor_dir: Path,
        ) -> LocalThreadWorkerPool:
            self.bound_ports.append(bound_port)
            self.auth_tokens.append(auth_token)
            return LocalThreadWorkerBackend(
                max_workers=1,
                resource_request=ResourceRequest(),
                scale_interval=1.0,
            ).start_pool(
                bound_port=bound_port,
                auth_token=auth_token,
                executor_dir=executor_dir,
            )

    leaf = ExecutionCoordinatorLeaf(value=11)
    objs = [leaf]
    backend = RecordingBackend()

    returned = ExecutionCoordinator.run(objs, worker_backends=(backend,))

    assert returned is objs
    assert leaf.status() == "completed"
    assert leaf.create() == 11
    assert len(backend.bound_ports) == 1
    assert backend.bound_ports[0] > 0
    assert len(backend.auth_tokens) == 1
    assert backend.auth_tokens[0]


def test_execution_coordinator_run_passes_executor_dir_to_worker_backend() -> None:
    class RecordingBackend:
        execution_coordinator_listen_host = "127.0.0.1"

        def __init__(self) -> None:
            self.executor_dirs: list[Path] = []

        def start_pool(
            self,
            *,
            bound_port: int,
            auth_token: str,
            executor_dir: Path,
        ) -> LocalThreadWorkerPool:
            self.executor_dirs.append(executor_dir)
            return LocalThreadWorkerBackend(
                max_workers=1,
                resource_request=ResourceRequest(),
                scale_interval=1.0,
            ).start_pool(
                bound_port=bound_port,
                auth_token=auth_token,
                executor_dir=executor_dir,
            )

    leaf = ExecutionCoordinatorLeaf(value=12)
    expected_executor_dir = _new_execution_coordinator([leaf]).executor_dir
    backend = RecordingBackend()

    ExecutionCoordinator.run([leaf], worker_backends=(backend,))

    assert backend.executor_dirs == [expected_executor_dir]


def test_execution_coordinator_run_writes_log_to_executor_dir() -> None:
    leaf = ExecutionCoordinatorLeaf(value=14)
    coordinator = _new_execution_coordinator([leaf])

    ExecutionCoordinator.run([leaf], worker_backends=(LocalThreadWorkerBackend(),))

    log_path = execution_coordinator_log_path_in(coordinator.executor_dir)
    assert execution_coordinator_log_path_in(coordinator.executor_dir) == log_path
    assert log_path.parent == coordinator.executor_dir

    log_text = log_path.read_text(encoding="utf-8")
    assert "starting exec=" in log_text
    assert "server listening on " in log_text
    assert f"creating {leaf._log_label}" not in log_text
    assert f"(object_id={leaf.object_id})" not in log_text
    assert f"leased {leaf._log_label} to local-worker-0" in log_text
    assert "worker=local-worker-0" in log_text
    assert leaf.object_id in log_text
    assert f"completed {leaf._log_label} ok" in log_text
    assert "progress 1/1 · 0 running" in log_text
    assert "failed_retry=0 failed=0" in log_text
    assert "furu execution coordinator finished successfully" in log_text


def test_execution_coordinator_run_returns_when_all_objects_are_already_completed() -> (
    None
):
    class UnexpectedBackend:
        execution_coordinator_listen_host = "127.0.0.1"

        def start_pool(
            self,
            *,
            bound_port: int,
            auth_token: str,
            executor_dir: Path,
        ) -> LocalThreadWorkerPool:
            raise AssertionError("coordinator started workers with no runnable objects")

    leaf = ExecutionCoordinatorLeaf(value=15)
    leaf.create()
    objs = [leaf]
    coordinator = _new_execution_coordinator(objs)

    assert coordinator.nodes_by_id == {}

    returned = ExecutionCoordinator.run(objs, worker_backends=(UnexpectedBackend(),))

    assert returned is objs
    log_text = execution_coordinator_log_path_in(coordinator.executor_dir).read_text(
        encoding="utf-8"
    )
    assert "all objects already exist; no execution coordinator work to run" in log_text
    assert "server listening on " not in log_text
    assert "furu execution coordinator finished successfully" in log_text


def test_execution_coordinator_run_starts_backend_pool_and_stops_and_joins_when_done() -> (
    None
):
    class RecordingPool:
        def __init__(self) -> None:
            self.events: list[str] = []
            self.stop_timeouts: list[float] = []
            self.worker_thread: threading.Thread | None = None

        def stop(self, *, timeout: float) -> None:
            self.events.append("stop")
            self.stop_timeouts.append(timeout)
            if self.worker_thread is not None:
                self.worker_thread.join(timeout=timeout)

    class RecordingBackend:
        execution_coordinator_listen_host = "127.0.0.1"

        def __init__(self, pool: RecordingPool) -> None:
            self.pool = pool

        def start_pool(
            self,
            *,
            bound_port: int,
            auth_token: str,
            executor_dir: Path,
        ) -> RecordingPool:
            self.pool.events.append("start_pool")
            server_url = f"http://127.0.0.1:{bound_port}"
            client = api.WorkerApiClient(server_url=server_url, auth_token=auth_token)
            failure_client = api.PoolApiClient(
                server_url=server_url, auth_token=auth_token
            )

            def complete_job() -> None:
                try:
                    job = client.lease_job(resources=ANY_RESOURCES)
                    if not isinstance(job, Job):
                        raise AssertionError(f"expected job, got {job!r}")
                    client.job_result(job.lease_id, JobCompletedResult())
                except BaseException as exc:
                    failure_client.fail(message=f"recording backend failed: {exc!r}")

            self.pool.worker_thread = threading.Thread(target=complete_job)
            self.pool.worker_thread.start()
            return self.pool

    pool = RecordingPool()

    ExecutionCoordinator.run(
        [ExecutionCoordinatorLeaf(value=13)],
        worker_backends=(RecordingBackend(pool),),
        port=0,
    )

    assert pool.events == ["start_pool", "stop"]
    assert pool.stop_timeouts == [5]


def test_execution_coordinator_run_stops_backend_pool_when_interrupted() -> None:
    class InterruptingEvent(threading.Event):
        def wait(self, timeout: float | None = None) -> bool:
            raise KeyboardInterrupt

    class InterruptingCoordinator(ExecutionCoordinator):
        def __init__(self, *, max_retries_per_object: int) -> None:
            super().__init__(max_retries_per_object=max_retries_per_object)
            self.done = InterruptingEvent()

    class RecordingPool:
        def __init__(self) -> None:
            self.events: list[str] = []
            self.stop_timeouts: list[float] = []

        def stop(self, *, timeout: float) -> None:
            self.events.append("stop")
            self.stop_timeouts.append(timeout)

    class RecordingBackend:
        execution_coordinator_listen_host = "127.0.0.1"

        def __init__(self, pool: RecordingPool) -> None:
            self.pool = pool

        def start_pool(
            self,
            *,
            bound_port: int,
            auth_token: str,
            executor_dir: Path,
        ) -> RecordingPool:
            self.pool.events.append("start_pool")
            return self.pool

    pool = RecordingPool()

    with pytest.raises(KeyboardInterrupt):
        InterruptingCoordinator.run(
            [ExecutionCoordinatorLeaf(value=13013)],
            worker_backends=(RecordingBackend(pool),),
            port=0,
        )

    assert pool.events == ["start_pool", "stop"]
    assert pool.stop_timeouts == [5]


def test_execution_coordinator_run_uses_worker_backend_execution_coordinator_listen_host() -> (
    None
):
    class RecordingPool:
        worker_thread: threading.Thread | None = None

        def stop(self, *, timeout: float) -> None:
            if self.worker_thread is not None:
                self.worker_thread.join(timeout=timeout)

    class RecordingBackend:
        execution_coordinator_listen_host = "127.0.0.1"

        def __init__(self) -> None:
            self.server_urls: list[str] = []

        def start_pool(
            self,
            *,
            bound_port: int,
            auth_token: str,
            executor_dir: Path,
        ) -> RecordingPool:
            server_url = f"http://{self.execution_coordinator_listen_host}:{bound_port}"
            self.server_urls.append(server_url)
            pool = RecordingPool()
            client = api.WorkerApiClient(server_url=server_url, auth_token=auth_token)
            failure_client = api.PoolApiClient(
                server_url=server_url, auth_token=auth_token
            )

            def complete_job() -> None:
                try:
                    job = client.lease_job(resources=ANY_RESOURCES)
                    if not isinstance(job, Job):
                        raise AssertionError(f"expected job, got {job!r}")
                    client.job_result(job.lease_id, JobCompletedResult())
                except BaseException as exc:
                    failure_client.fail(message=f"recording backend failed: {exc!r}")

            pool.worker_thread = threading.Thread(target=complete_job)
            pool.worker_thread.start()
            return pool

    backend = RecordingBackend()

    ExecutionCoordinator.run(
        [ExecutionCoordinatorLeaf(value=15)], worker_backends=(backend,), port=0
    )

    assert len(backend.server_urls) == 1
    assert backend.server_urls[0].startswith("http://127.0.0.1:")


def test_execution_coordinator_server_exposes_bound_host_and_port() -> None:
    coordinator = _new_execution_coordinator([ExecutionCoordinatorLeaf(value=12)])

    with execution_coordinator_server(
        coordinator, bind_host="127.0.0.1", port=0
    ) as server:
        assert server.bound_host == "127.0.0.1"
        assert server.bound_port > 0
        assert server.auth_token


def test_execution_coordinator_server_rejects_requests_without_auth_token() -> None:
    coordinator = _new_execution_coordinator([ExecutionCoordinatorLeaf(value=12)])

    with execution_coordinator_server(
        coordinator, bind_host="127.0.0.1", port=0
    ) as server:
        response = httpx.post(f"{server.server_url}/worker/lease_job")
        assert response.status_code == 401
        assert response.json() == {
            "detail": "invalid furu execution coordinator auth token"
        }

        response = httpx.post(
            f"{server.server_url}/worker/lease_job",
            headers={"Authorization": "Bearer wrong"},
        )
        assert response.status_code == 401
        assert response.json() == {
            "detail": "invalid furu execution coordinator auth token"
        }

        response = httpx.post(
            f"{server.server_url}/worker/lease_job",
            headers={"Authorization": f"Bearer {server.auth_token}"},
            json={"resources": {"cpus": 1, "gpus": 0, "memory_gib": 0}},
        )
        assert response.status_code == 200


def test_execution_coordinator_run_requires_explicit_worker_backends() -> None:
    with pytest.raises(TypeError, match="worker_backends"):
        ExecutionCoordinator.run([ExecutionCoordinatorLeaf(value=12)])  # ty: ignore[missing-argument]


def test_execution_coordinator_run_rejects_empty_worker_backends() -> None:
    with pytest.raises(ValueError, match="not enough values to unpack"):
        ExecutionCoordinator.run(
            [ExecutionCoordinatorLeaf(value=12)], worker_backends=()
        )


def test_execution_coordinator_run_rejects_conflicting_execution_coordinator_listen_host() -> (
    None
):
    with pytest.raises(ValueError, match="too many values to unpack"):
        ExecutionCoordinator.run(
            [ExecutionCoordinatorLeaf(value=12)],
            worker_backends=(
                LocalThreadWorkerBackend(execution_coordinator_listen_host="127.0.0.1"),
                LocalThreadWorkerBackend(execution_coordinator_listen_host="0.0.0.0"),
            ),
        )


def test_job_result_request_requires_error_for_failed_status() -> None:
    with pytest.raises(ValidationError, match="Field required"):
        JobFailedResult.model_validate({"status": "failed"})


def test_job_result_request_uses_status_discriminator() -> None:
    adapter = TypeAdapter(JobResultRequest)

    assert adapter.validate_python({"status": "completed"}) == JobCompletedResult()
    assert adapter.validate_python(
        {"status": "failed", "error": "boom"}
    ) == JobFailedResult(error="boom")
    assert adapter.validate_python(
        {"status": "blocked", "dependencies": []}
    ) == JobBlockedResult(dependencies=[])
    with pytest.raises(ValidationError, match="Input tag 'skipped'"):
        adapter.validate_python({"status": "skipped"})


def test_worker_loop_raises_when_server_is_unavailable() -> None:
    with pytest.raises(RuntimeError, match="failed"):
        worker_loop(
            server_url="http://127.0.0.1:1",
            auth_token="test-token",
            resource_request=ResourceRequest(),
            idle_timeout=get_config().worker.idle_timeout_seconds,
            component="test-worker",
        )


def test_worker_loop_exits_after_idle_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TestClient:
        lease_calls = 0

        def __init__(self, server_url: str, *, auth_token: str) -> None:
            pass

        def lease_job(self, *, resources: ResourceRequest) -> LeaseJobResponse:
            self.lease_calls += 1
            return "wait"

    test_client = TestClient("http://worker.test", auth_token="test-token")
    monkeypatch.setattr(
        api,
        "WorkerApiClient",
        lambda server_url, *, auth_token: test_client,
    )

    worker_loop(
        server_url="http://worker.test",
        auth_token="test-token",
        resource_request=ResourceRequest(),
        idle_timeout=0,
        component="test-worker",
    )

    assert test_client.lease_calls == 1


def test_worker_loop_logs_task_requests_and_received_task(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    leaf = ExecutionCoordinatorLeaf(value=1)
    job = Job(lease_id="lease-1", artifact=ArtifactSpec.from_furu(leaf))
    leases: list[LeaseJobResponse] = [job, "stop"]

    class TestClient:
        def __init__(self, server_url: str, *, auth_token: str) -> None:
            pass

        def lease_job(self, *, resources: ResourceRequest) -> LeaseJobResponse:
            return leases.pop(0)

        def job_result(self, lease_id: str, request: JobResultRequest) -> None:
            pass

    def ensure_single_result(obj: Furu[object]) -> None:
        pass

    test_client = TestClient("http://worker.test", auth_token="test-token")
    monkeypatch.setattr(
        api,
        "WorkerApiClient",
        lambda server_url, *, auth_token: test_client,
    )
    monkeypatch.setattr(
        worker_loop_module, "_ensure_single_result", ensure_single_result
    )

    with _captured_furu_logs(caplog):
        worker_loop(
            server_url="http://worker.test",
            auth_token="test-token",
            resource_request=ResourceRequest(),
            idle_timeout=get_config().worker.idle_timeout_seconds,
            component="test-worker",
        )

    assert "worker requesting new task from server" not in caplog.messages
    assert f"received {leaf._log_label}" in caplog.messages
    assert any(
        message.startswith(f"finished {leaf._log_label} ok ·")
        for message in caplog.messages
    )


def test_worker_loop_logs_stop_and_first_wait(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    leases: list[LeaseJobResponse] = ["wait", "wait", "stop"]

    class TestClient:
        def __init__(self, server_url: str, *, auth_token: str) -> None:
            pass

        def lease_job(self, *, resources: ResourceRequest) -> LeaseJobResponse:
            return leases.pop(0)

    test_client = TestClient("http://worker.test", auth_token="test-token")
    monkeypatch.setattr(
        api,
        "WorkerApiClient",
        lambda server_url, *, auth_token: test_client,
    )
    monkeypatch.setattr(worker_loop_module.time, "sleep", lambda seconds: None)

    with _captured_furu_logs(caplog):
        worker_loop(
            server_url="http://worker.test",
            auth_token="test-token",
            resource_request=ResourceRequest(),
            idle_timeout=get_config().worker.idle_timeout_seconds,
            component="test-worker",
        )

    assert "worker requesting new task from server" not in caplog.messages
    assert caplog.messages.count("worker told to wait") == 1
    assert "worker told to stop" in caplog.messages


def test_worker_loop_exits_after_exceeding_max_consecutive_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaf = ExecutionCoordinatorLeaf(value=1)
    jobs = [
        Job(lease_id="lease-1", artifact=ArtifactSpec.from_furu(leaf)),
        Job(lease_id="lease-2", artifact=ArtifactSpec.from_furu(leaf)),
        Job(lease_id="lease-3", artifact=ArtifactSpec.from_furu(leaf)),
    ]

    class TestClient:
        lease_calls: int
        results: list[tuple[str, JobResultRequest]]

        def __init__(self, server_url: str, *, auth_token: str) -> None:
            self.lease_calls = 0
            self.results = []

        def lease_job(self, *, resources: ResourceRequest) -> LeaseJobResponse:
            self.lease_calls += 1
            return jobs.pop(0)

        def job_result(self, lease_id: str, request: JobResultRequest) -> None:
            self.results.append((lease_id, request))

    def ensure_single_result(obj: Furu[object]) -> None:
        raise RuntimeError("worker task failed")

    test_client = TestClient("http://worker.test", auth_token="test-token")
    monkeypatch.setattr(
        api,
        "WorkerApiClient",
        lambda server_url, *, auth_token: test_client,
    )
    monkeypatch.setattr(
        worker_loop_module, "_ensure_single_result", ensure_single_result
    )

    worker_loop(
        server_url="http://worker.test",
        auth_token="test-token",
        resource_request=ResourceRequest(),
        idle_timeout=get_config().worker.idle_timeout_seconds,
        max_consecutive_failures=2,
        component="test-worker",
    )

    assert test_client.lease_calls == 3
    assert [lease_id for lease_id, _request in test_client.results] == [
        "lease-1",
        "lease-2",
        "lease-3",
    ]
    assert all(
        isinstance(request, JobFailedResult)
        for _lease_id, request in test_client.results
    )


def test_worker_loop_resets_consecutive_failures_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaf = ExecutionCoordinatorLeaf(value=1)
    leases: list[LeaseJobResponse] = [
        Job(lease_id="lease-1", artifact=ArtifactSpec.from_furu(leaf)),
        Job(lease_id="lease-2", artifact=ArtifactSpec.from_furu(leaf)),
        Job(lease_id="lease-3", artifact=ArtifactSpec.from_furu(leaf)),
        "stop",
    ]

    class TestClient:
        lease_calls: int
        results: list[JobResultRequest]

        def __init__(self, server_url: str, *, auth_token: str) -> None:
            self.lease_calls = 0
            self.results = []

        def lease_job(self, *, resources: ResourceRequest) -> LeaseJobResponse:
            self.lease_calls += 1
            return leases.pop(0)

        def job_result(self, lease_id: str, request: JobResultRequest) -> None:
            self.results.append(request)

    calls = 0

    def ensure_single_result(obj: Furu[object]) -> None:
        nonlocal calls
        calls += 1
        if calls in (1, 3):
            raise RuntimeError("worker task failed")

    test_client = TestClient("http://worker.test", auth_token="test-token")
    monkeypatch.setattr(
        api,
        "WorkerApiClient",
        lambda server_url, *, auth_token: test_client,
    )
    monkeypatch.setattr(
        worker_loop_module, "_ensure_single_result", ensure_single_result
    )

    worker_loop(
        server_url="http://worker.test",
        auth_token="test-token",
        resource_request=ResourceRequest(),
        idle_timeout=get_config().worker.idle_timeout_seconds,
        max_consecutive_failures=2,
        component="test-worker",
    )

    assert test_client.lease_calls == 4
    assert [result.status for result in test_client.results] == [
        "failed",
        "completed",
        "failed",
    ]


def test_worker_loop_does_not_swallow_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaf = ExecutionCoordinatorLeaf(value=1)
    job = Job(lease_id="lease-1", artifact=ArtifactSpec.from_furu(leaf))

    class TestClient:
        calls: list[str]
        lease_resources: list[ResourceRequest]

        def __init__(self, server_url: str, *, auth_token: str) -> None:
            self.calls = []
            self.lease_resources = []

        def lease_job(self, *, resources: ResourceRequest) -> LeaseJobResponse:
            self.calls.append("lease_job")
            self.lease_resources.append(resources)
            return job

        def job_result(self, lease_id: str, request: JobResultRequest) -> None:
            self.calls.append("job_result")

    def ensure_single_result(obj: Furu[object]) -> None:
        raise KeyboardInterrupt

    test_client = TestClient("http://worker.test", auth_token="test-token")
    monkeypatch.setattr(
        api,
        "WorkerApiClient",
        lambda server_url, *, auth_token: test_client,
    )
    monkeypatch.setattr(
        worker_loop_module, "_ensure_single_result", ensure_single_result
    )

    with pytest.raises(KeyboardInterrupt):
        worker_loop(
            server_url="http://worker.test",
            auth_token="test-token",
            resource_request=ResourceRequest(gpus=1),
            idle_timeout=get_config().worker.idle_timeout_seconds,
            component="test-worker",
        )

    assert test_client.calls == ["lease_job"]
    assert test_client.lease_resources == [ResourceRequest(gpus=1)]


def test_client_job_result_uses_job_result_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[tuple[str, str, dict[str, str], object | None]] = []

    def request(
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json: object | None,
        timeout: float,
    ) -> httpx.Response:
        requests.append((method, url, headers, json))
        return httpx.Response(
            200, json={"ok": True}, request=httpx.Request(method, url)
        )

    monkeypatch.setattr(httpx, "request", request)

    api.WorkerApiClient(
        server_url="http://worker.test/", auth_token="secret-token"
    ).job_result("lease-1", JobBlockedResult(dependencies=[]))

    assert requests == [
        (
            "POST",
            "http://worker.test/worker/job_result/lease-1",
            {"Authorization": "Bearer secret-token"},
            {"status": "blocked", "dependencies": []},
        )
    ]


def test_client_job_result_rejects_non_ok_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def request(
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json: object | None,
        timeout: float,
    ) -> httpx.Response:
        return httpx.Response(
            200, json={"ok": False}, request=httpx.Request(method, url)
        )

    monkeypatch.setattr(httpx, "request", request)

    with pytest.raises(ValidationError, match="Input should be True"):
        api.WorkerApiClient(
            server_url="http://worker.test", auth_token="secret-token"
        ).job_result("lease-1", JobCompletedResult())


def test_client_lease_job_rejects_empty_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def request(
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json: object | None,
        timeout: float,
    ) -> httpx.Response:
        return httpx.Response(200, request=httpx.Request(method, url))

    monkeypatch.setattr(httpx, "request", request)

    with pytest.raises(RuntimeError, match="returned an empty response"):
        api.WorkerApiClient(
            server_url="http://worker.test", auth_token="secret-token"
        ).lease_job(resources=ANY_RESOURCES)


def test_client_lease_job_posts_resource_request_to_lease_job_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaf = ExecutionCoordinatorLeaf(value=1)
    requests: list[tuple[str, str, dict[str, str], object | None]] = []

    def request(
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json: object | None,
        timeout: float,
    ) -> httpx.Response:
        requests.append((method, url, headers, json))
        return httpx.Response(
            200,
            json={
                "lease_id": "lease-1",
                "artifact": ArtifactSpec.from_furu(leaf).model_dump(mode="json"),
            },
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr(httpx, "request", request)

    job = api.WorkerApiClient(
        server_url="http://worker.test/",
        auth_token="secret-token",
    ).lease_job(resources=ResourceRequest(cpus=2, gpus=1, memory_gib=16))

    assert isinstance(job, Job)
    assert requests == [
        (
            "POST",
            "http://worker.test/worker/lease_job",
            {"Authorization": "Bearer secret-token"},
            {"resources": {"cpus": 2, "gpus": 1, "memory_gib": 16}},
        )
    ]


def test_execution_coordinator_api_rejects_missing_auth_token() -> None:
    app = create_execution_coordinator_api_app(
        _new_execution_coordinator([ExecutionCoordinatorLeaf(value=1)]),
        auth_token="secret",
    )
    client = TestClient(app)

    response = client.post("/worker/lease_job")

    assert response.status_code == 401
    assert response.json() == {
        "detail": "invalid furu execution coordinator auth token"
    }


def test_execution_coordinator_api_rejects_wrong_auth_token() -> None:
    app = create_execution_coordinator_api_app(
        _new_execution_coordinator([ExecutionCoordinatorLeaf(value=1)]),
        auth_token="secret",
    )
    client = TestClient(app)

    response = client.post(
        "/worker/lease_job",
        headers={"Authorization": "Bearer wrong"},
    )

    assert response.status_code == 401
    assert response.json() == {
        "detail": "invalid furu execution coordinator auth token"
    }


def test_execution_coordinator_api_fail_endpoint_sets_finish_error_and_done() -> None:
    coordinator = _new_execution_coordinator([ExecutionCoordinatorLeaf(value=1)])
    app = create_execution_coordinator_api_app(coordinator, auth_token="secret")
    client = TestClient(app)

    response = client.post(
        "/pool/fail",
        headers={"Authorization": "Bearer secret"},
        json={"message": "pool broke"},
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert coordinator.done.is_set()
    with pytest.raises(RuntimeError, match="pool broke"):
        coordinator.raise_for_failure()


def test_execution_coordinator_api_accepts_matching_auth_token() -> None:
    app = create_execution_coordinator_api_app(
        _new_execution_coordinator([ExecutionCoordinatorLeaf(value=1)]),
        auth_token="secret",
    )
    client = TestClient(app)

    response = client.post(
        "/worker/lease_job",
        headers={"Authorization": "Bearer secret"},
        json={"resources": {"cpus": 1, "gpus": 0, "memory_gib": 0}},
    )

    assert response.status_code == 200
    assert response.json()["artifact"]["artifact_data"]["|fields"] == {"value": 1}
