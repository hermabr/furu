import threading
from pathlib import Path
from typing import Any, ClassVar, cast
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import TypeAdapter, ValidationError

import furu.worker.loop as worker_loop_module
from furu import Furu
from furu.config import get_config
from furu.execution import api
from furu.execution.api import create_manager_api_app
from furu.execution.manager import (
    FailedJob,
    Manager,
    RunningJob,
)
from furu.execution.server import _run_until_done, manager_server
from furu.metadata import ArtifactSpec
from furu.resources import ResourceRequest, ResourceRequirements
from furu._storage_layout import manager_log_path_in
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


def _new_local_pool(
    *,
    server_url: str = "http://manager.test",
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


def test_manager_init_partitions_ready_and_blocked() -> None:
    leaf = ManagerLeaf(value=1)
    parent = ManagerParent(child=leaf)

    manager = Manager([parent])

    assert set(manager.ready) == {leaf.object_id}
    assert set(manager.blocked) == {parent.object_id}
    assert manager.running == {}


def test_manager_executor_id_is_stable_hash_of_root_object_tuple() -> None:
    left = ManagerLeaf(value=1)
    right = ManagerLeaf(value=2)

    manager = Manager([left, right])

    assert len(manager.executor_id) == 32
    assert int(manager.executor_id, 16) >= 0
    assert Manager([left, right]).executor_id == manager.executor_id
    assert Manager([right, left]).executor_id != manager.executor_id
    assert (
        manager.executor_dir
        == get_config().directories.executions / manager.executor_id
    )


def test_manager_job_result_completed_moves_dependents_to_ready() -> None:
    leaf = ManagerLeaf(value=1)
    parent = ManagerParent(child=leaf)
    manager = Manager([parent])

    job = manager.lease_job(resources=ANY_RESOURCES)
    assert isinstance(job, Job)
    assert job.lease_id != leaf.object_id
    assert UUID(job.lease_id).version == 4
    assert set(manager.running) == {job.lease_id}
    running_job = manager.running[job.lease_id]
    assert isinstance(running_job, RunningJob)
    assert running_job.node.obj is leaf

    manager.job_result(job.lease_id, JobCompletedResult())

    assert manager.running == {}
    assert set(manager.completed) == {leaf.object_id}
    assert set(manager.ready) == {parent.object_id}
    assert manager.blocked == {}


def test_manager_lease_job_returns_wait_when_only_running_jobs_can_unblock_work() -> (
    None
):
    leaf = ManagerLeaf(value=1)
    parent = ManagerParent(child=leaf)
    manager = Manager([parent])

    job = manager.lease_job(resources=ANY_RESOURCES)
    assert isinstance(job, Job)

    assert manager.lease_job(resources=ANY_RESOURCES) == "wait"
    assert not manager.done.is_set()


def test_manager_job_result_blocked_discovers_lazy_dependency_and_reruns_parent() -> (
    None
):
    parent = ManagerLazyParent(value=2)
    dependency = ManagerLeaf(value=2)
    manager = Manager([parent])

    parent_job = manager.lease_job(resources=ANY_RESOURCES)
    assert isinstance(parent_job, Job)
    assert parent_job.lease_id != parent.object_id

    manager.job_result(
        parent_job.lease_id,
        JobBlockedResult(dependencies=[ArtifactSpec.from_furu(dependency)]),
    )

    assert set(manager.ready) == {dependency.object_id}
    assert set(manager.blocked) == {parent.object_id}

    dependency_job = manager.lease_job(resources=ANY_RESOURCES)
    assert isinstance(dependency_job, Job)
    manager.job_result(dependency_job.lease_id, JobCompletedResult())

    assert set(manager.ready) == {parent.object_id}
    assert manager.blocked == {}


def test_manager_job_result_blocked_ignores_completed_lazy_dependency() -> None:
    parent = ManagerLazyParent(value=2)
    dependency = ManagerLeaf(value=2)
    dependency.load_or_create()
    manager = Manager([parent])

    parent_job = manager.lease_job(resources=ANY_RESOURCES)
    assert isinstance(parent_job, Job)

    manager.job_result(
        parent_job.lease_id,
        JobBlockedResult(dependencies=[ArtifactSpec.from_furu(dependency)]),
    )

    assert set(manager.ready) == {parent.object_id}
    assert manager.blocked == {}
    assert dependency.object_id not in manager.nodes_by_id


def test_manager_job_result_blocked_discovers_multiple_lazy_dependencies_together() -> (
    None
):
    parent = ManagerLazyParent(value=2)
    dependencies = [ManagerLeaf(value=2), ManagerLeaf(value=3)]
    manager = Manager([parent])

    parent_job = manager.lease_job(resources=ANY_RESOURCES)
    assert isinstance(parent_job, Job)

    manager.job_result(
        parent_job.lease_id,
        JobBlockedResult(
            dependencies=[
                ArtifactSpec.from_furu(dependency) for dependency in dependencies
            ]
        ),
    )

    assert set(manager.ready) == {dependency.object_id for dependency in dependencies}
    assert set(manager.blocked) == {parent.object_id}

    parent_node = manager.nodes_by_id[parent.object_id]
    assert {node.obj.object_id for node in parent_node.dependencies} == {
        dependency.object_id for dependency in dependencies
    }
    for dependency in dependencies:
        dependency_node = manager.nodes_by_id[dependency.object_id]
        assert parent_node in dependency_node.dependents


def test_manager_uses_new_lease_when_blocked_job_is_released() -> None:
    parent = ManagerLazyParent(value=2)
    dependency = ManagerLeaf(value=2)
    manager = Manager([parent])

    first_parent_job = manager.lease_job(resources=ANY_RESOURCES)
    assert isinstance(first_parent_job, Job)

    manager.job_result(
        first_parent_job.lease_id,
        JobBlockedResult(dependencies=[ArtifactSpec.from_furu(dependency)]),
    )

    dependency_job = manager.lease_job(resources=ANY_RESOURCES)
    assert isinstance(dependency_job, Job)
    manager.job_result(dependency_job.lease_id, JobCompletedResult())

    second_parent_job = manager.lease_job(resources=ANY_RESOURCES)
    assert isinstance(second_parent_job, Job)
    assert second_parent_job.lease_id != first_parent_job.lease_id
    assert second_parent_job.artifact.object_id == parent.object_id

    assert set(manager.running) == {second_parent_job.lease_id}
    assert set(manager.completed) == {dependency.object_id}


def test_manager_job_result_failed_finishes_with_error() -> None:
    leaf = ManagerLeaf(value=1)
    manager = Manager([leaf])
    job = manager.lease_job(resources=ANY_RESOURCES)
    assert isinstance(job, Job)

    manager.job_result(job.lease_id, JobFailedResult(error="boom"))

    assert manager.running == {}
    assert set(manager.failed) == {leaf.object_id}
    failed_job = manager.failed[leaf.object_id]
    assert isinstance(failed_job, FailedJob)
    assert failed_job.lease_id == job.lease_id
    assert failed_job.node.obj is leaf
    assert failed_job.error == "boom"
    log_text = manager_log_path_in(manager.executor_dir).read_text(encoding="utf-8")
    assert "job failed:" in log_text
    assert "boom" in log_text
    assert "furu manager finished with error" in log_text
    with pytest.raises(RuntimeError, match="failed jobs"):
        manager.raise_for_failure()


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
        return self.parent.load_or_create() + self.value


class DynamicCpuAfterGpu(Furu[int]):
    parent: DynamicGpuAfterSeed
    value: int
    create_calls: ClassVar[list[int]] = []

    @property
    def resource_requirements(self) -> ResourceRequirements | None:
        return ResourceRequirements(gpus=(0, 0))

    def create(self) -> int:
        type(self).create_calls.append(self.value)
        return self.parent.load_or_create() + self.value


class DynamicGpuAfterCpu(Furu[int]):
    parent: DynamicCpuAfterGpu
    value: int
    create_calls: ClassVar[list[int]] = []

    @property
    def resource_requirements(self) -> ResourceRequirements | None:
        return ResourceRequirements(gpus=(1, None))

    def create(self) -> int:
        type(self).create_calls.append(self.value)
        return self.parent.load_or_create() + self.value


def test_count_satisfiable_jobs_caps_at_max_workers_and_filters_by_requirements() -> (
    None
):
    manager = Manager([ManagerLeaf(value=1), ManagerLeaf(value=2), GpuLeaf(value=3)])

    assert (
        manager.count_satisfiable_jobs(resources=ResourceRequest(), max_workers=10) == 2
    )
    assert (
        manager.count_satisfiable_jobs(resources=ResourceRequest(), max_workers=1) == 1
    )
    assert (
        manager.count_satisfiable_jobs(
            resources=ResourceRequest(gpus=1), max_workers=10
        )
        == 3
    )


def test_lease_job_filters_by_worker_resources() -> None:
    cpu_leaf = CpuOnlyLeaf(value=1)
    gpu_leaf = GpuLeaf(value=2)
    manager = Manager([cpu_leaf, gpu_leaf])

    cpu_job = manager.lease_job(resources=ResourceRequest(gpus=0))
    assert isinstance(cpu_job, Job)
    assert cpu_job.artifact.object_id == cpu_leaf.object_id

    assert manager.lease_job(resources=ResourceRequest(gpus=0)) == "wait"

    gpu_job = manager.lease_job(resources=ResourceRequest(gpus=1))
    assert isinstance(gpu_job, Job)
    assert gpu_job.artifact.object_id == gpu_leaf.object_id


def test_manager_run_dynamically_allocates_local_workers_for_later_resource_stages() -> (
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

    Manager(final_gpus).run(
        worker_backends=(
            LocalThreadWorkerBackend(
                max_workers=1,
                resource_request=ResourceRequest(gpus=0),
            ),
            LocalThreadWorkerBackend(
                max_workers=3,
                resource_request=ResourceRequest(gpus=1),
            ),
        )
    )

    assert DynamicCpuSeed.create_calls == [seed_value]
    assert DynamicGpuAfterSeed.create_calls == [20]
    assert DynamicCpuAfterGpu.create_calls == [30]
    assert sorted(DynamicGpuAfterCpu.create_calls) == [0, 1, 2, 3]
    assert [obj.load_or_create() for obj in final_gpus] == [
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
        assert all(name.startswith("furu-worker-") for name in worker_names)
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


def test_manager_run_fails_when_worker_pool_reports_unhealthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def crashing_worker_loop(
        *,
        server_url: str,
        auth_token: str,
        resource_request: ResourceRequest,
        idle_timeout: float,
    ) -> None:
        raise RuntimeError("worker boom")

    monkeypatch.setattr(worker_loop_module, "worker_loop", crashing_worker_loop)

    leaf = ManagerLeaf(value=42)
    manager = Manager([leaf])

    with pytest.raises(RuntimeError, match="local worker pool became unhealthy"):
        manager.run(
            worker_backends=(
                LocalThreadWorkerBackend(
                    max_workers=1,
                    max_failed_restarts=0,
                    resource_request=ResourceRequest(),
                    scale_interval=0.05,
                ),
            )
        )


def test_manager_run_uses_worker_backend() -> None:
    class RecordingBackend:
        manager_listen_host = "0.0.0.0"

        def __init__(self) -> None:
            self.server_urls: list[str] = []
            self.auth_tokens: list[str] = []

        def start_pool(
            self,
            *,
            server_url: str,
            auth_token: str,
            executor_dir: Path,
        ) -> LocalThreadWorkerPool:
            self.server_urls.append(server_url)
            self.auth_tokens.append(auth_token)
            return LocalThreadWorkerBackend(
                max_workers=1,
                resource_request=ResourceRequest(),
                scale_interval=1.0,
            ).start_pool(
                server_url=server_url,
                auth_token=auth_token,
                executor_dir=executor_dir,
            )

    leaf = ManagerLeaf(value=11)
    backend = RecordingBackend()

    Manager([leaf]).run(worker_backends=(backend,))

    assert leaf.status() == "completed"
    assert leaf.load_or_create() == 11
    assert len(backend.server_urls) == 1
    assert backend.server_urls[0].startswith("http://0.0.0.0:")
    assert len(backend.auth_tokens) == 1
    assert backend.auth_tokens[0]


def test_manager_run_passes_executor_dir_to_worker_backend() -> None:
    class RecordingBackend:
        manager_listen_host = "127.0.0.1"

        def __init__(self) -> None:
            self.executor_dirs: list[Path] = []

        def start_pool(
            self,
            *,
            server_url: str,
            auth_token: str,
            executor_dir: Path,
        ) -> LocalThreadWorkerPool:
            self.executor_dirs.append(executor_dir)
            return LocalThreadWorkerBackend(
                max_workers=1,
                resource_request=ResourceRequest(),
                scale_interval=1.0,
            ).start_pool(
                server_url=server_url,
                auth_token=auth_token,
                executor_dir=executor_dir,
            )

    leaf = ManagerLeaf(value=12)
    manager = Manager([leaf])
    backend = RecordingBackend()

    manager.run(worker_backends=(backend,))

    assert backend.executor_dirs == [manager.executor_dir]


def test_manager_run_writes_log_to_executor_dir() -> None:
    leaf = ManagerLeaf(value=14)
    manager = Manager([leaf])

    manager.run(worker_backends=(LocalThreadWorkerBackend(),))

    log_path = manager_log_path_in(manager.executor_dir)
    assert manager_log_path_in(manager.executor_dir) == log_path
    assert log_path.parent == manager.executor_dir

    log_text = log_path.read_text(encoding="utf-8")
    assert "starting furu manager" in log_text
    assert "manager server listening" in log_text
    assert "leased job:" in log_text
    assert leaf.object_id in log_text
    assert "job completed:" in log_text
    assert (
        "manager progress: completed=1/1 failed=0 running=0 ready=0 blocked=0"
        in log_text
    )
    assert "furu manager finished successfully" in log_text


def test_manager_run_starts_backend_pool_and_stops_and_joins_when_done() -> None:
    class RecordingDone:
        def __init__(self) -> None:
            self.wait_calls = 0

        def wait(self, timeout: float | None = None) -> bool:
            self.wait_calls += 1
            return True

    class RecordingPool:
        def __init__(self) -> None:
            self.events: list[str] = []
            self.stop_timeouts: list[float] = []

        def stop(self, *, timeout: float) -> None:
            self.events.append("stop")
            self.stop_timeouts.append(timeout)

    class RecordingBackend:
        manager_listen_host = "127.0.0.1"

        def __init__(self, pool: RecordingPool) -> None:
            self.pool = pool

        def start_pool(
            self,
            *,
            server_url: str,
            auth_token: str,
            executor_dir: Path,
        ) -> RecordingPool:
            self.pool.events.append("start_pool")
            return self.pool

    manager = Manager([ManagerLeaf(value=13)])
    done = RecordingDone()
    pool = RecordingPool()
    cast(Any, manager).done = done

    _run_until_done(
        manager,
        worker_backends=(RecordingBackend(pool),),
        port=0,
    )

    assert done.wait_calls == 1
    assert pool.events == ["start_pool", "stop"]
    assert pool.stop_timeouts == [5]


def test_run_until_done_uses_worker_backend_manager_listen_host() -> None:
    class RecordingDone:
        def wait(self, timeout: float | None = None) -> bool:
            return True

    class RecordingPool:
        def stop(self, *, timeout: float) -> None:
            pass

    class RecordingBackend:
        manager_listen_host = "127.0.0.1"

        def __init__(self) -> None:
            self.server_urls: list[str] = []

        def start_pool(
            self,
            *,
            server_url: str,
            auth_token: str,
            executor_dir: Path,
        ) -> RecordingPool:
            self.server_urls.append(server_url)
            return RecordingPool()

    manager = Manager([ManagerLeaf(value=15)])
    backend = RecordingBackend()
    cast(Any, manager).done = RecordingDone()

    _run_until_done(manager, worker_backends=(backend,), port=0)

    assert len(backend.server_urls) == 1
    assert backend.server_urls[0].startswith("http://127.0.0.1:")


def test_manager_server_exposes_bound_host_and_port() -> None:
    manager = Manager([ManagerLeaf(value=12)])

    with manager_server(manager, bind_host="127.0.0.1", port=0) as server:
        assert server.bound_host == "127.0.0.1"
        assert server.bound_port > 0
        assert server.auth_token


def test_manager_server_rejects_requests_without_auth_token() -> None:
    manager = Manager([ManagerLeaf(value=12)])

    with manager_server(manager, bind_host="127.0.0.1", port=0) as server:
        response = httpx.post(f"{server.server_url}/worker/lease_job")
        assert response.status_code == 401
        assert response.json() == {"detail": "invalid furu manager auth token"}

        response = httpx.post(
            f"{server.server_url}/worker/lease_job",
            headers={"Authorization": "Bearer wrong"},
        )
        assert response.status_code == 401
        assert response.json() == {"detail": "invalid furu manager auth token"}

        response = httpx.post(
            f"{server.server_url}/worker/lease_job",
            headers={"Authorization": f"Bearer {server.auth_token}"},
            json={"resources": {"cpus": 1, "gpus": 0}},
        )
        assert response.status_code == 200


def test_manager_run_requires_explicit_worker_backends() -> None:
    manager = Manager([ManagerLeaf(value=12)])

    with pytest.raises(TypeError, match="worker_backends"):
        manager.run()  # ty: ignore[missing-argument]


def test_manager_run_rejects_empty_worker_backends() -> None:
    manager = Manager([ManagerLeaf(value=12)])

    with pytest.raises(ValueError, match="not enough values to unpack"):
        manager.run(worker_backends=())


def test_manager_run_rejects_conflicting_manager_listen_host() -> None:
    manager = Manager([ManagerLeaf(value=12)])

    with pytest.raises(ValueError, match="too many values to unpack"):
        manager.run(
            worker_backends=(
                LocalThreadWorkerBackend(manager_listen_host="127.0.0.1"),
                LocalThreadWorkerBackend(manager_listen_host="0.0.0.0"),
            )
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
    with pytest.raises(httpx.ConnectError):
        worker_loop(
            server_url="http://127.0.0.1:1",
            auth_token="test-token",
            resource_request=ResourceRequest(),
            idle_timeout=get_config().worker.idle_timeout_seconds,
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
    )

    assert test_client.lease_calls == 1


def test_worker_loop_exits_after_max_consecutive_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaf = ManagerLeaf(value=1)
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
    )

    assert test_client.lease_calls == 2
    assert [lease_id for lease_id, _request in test_client.results] == [
        "lease-1",
        "lease-2",
    ]
    assert all(
        isinstance(request, JobFailedResult)
        for _lease_id, request in test_client.results
    )


def test_worker_loop_resets_consecutive_failures_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaf = ManagerLeaf(value=1)
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
    leaf = ManagerLeaf(value=1)
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
    leaf = ManagerLeaf(value=1)
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
    ).lease_job(resources=ResourceRequest(cpus=2, gpus=1))

    assert isinstance(job, Job)
    assert requests == [
        (
            "POST",
            "http://worker.test/worker/lease_job",
            {"Authorization": "Bearer secret-token"},
            {"resources": {"cpus": 2, "gpus": 1}},
        )
    ]


def test_manager_api_rejects_missing_auth_token() -> None:
    app = create_manager_api_app(Manager([ManagerLeaf(value=1)]), auth_token="secret")
    client = TestClient(app)

    response = client.post("/worker/lease_job")

    assert response.status_code == 401
    assert response.json() == {"detail": "invalid furu manager auth token"}


def test_manager_api_rejects_wrong_auth_token() -> None:
    app = create_manager_api_app(Manager([ManagerLeaf(value=1)]), auth_token="secret")
    client = TestClient(app)

    response = client.post(
        "/worker/lease_job",
        headers={"Authorization": "Bearer wrong"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "invalid furu manager auth token"}


def test_manager_api_fail_endpoint_sets_finish_error_and_done() -> None:
    manager = Manager([ManagerLeaf(value=1)])
    app = create_manager_api_app(manager, auth_token="secret")
    client = TestClient(app)

    response = client.post(
        "/pool/fail",
        headers={"Authorization": "Bearer secret"},
        json={"message": "pool broke"},
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert manager.done.is_set()
    with pytest.raises(RuntimeError, match="pool broke"):
        manager.raise_for_failure()


def test_manager_api_accepts_matching_auth_token() -> None:
    app = create_manager_api_app(Manager([ManagerLeaf(value=1)]), auth_token="secret")
    client = TestClient(app)

    response = client.post(
        "/worker/lease_job",
        headers={"Authorization": "Bearer secret"},
        json={"resources": {"cpus": 1, "gpus": 0}},
    )

    assert response.status_code == 200
    assert response.json()["artifact"]["artifact_data"]["|fields"] == {"value": 1}
