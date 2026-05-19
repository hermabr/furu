from pathlib import Path
from typing import Any, cast
from uuid import UUID

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import TypeAdapter, ValidationError

import furu.worker.backends.local as local_backend_module
import furu.worker.loop as worker_loop_module
from furu import Furu, ResourceRequest, ResourceRequirements
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
from furu._storage_layout import manager_log_path_in
from furu.worker.backends.local import LocalThreadWorkerBackend, LocalThreadWorkerPool
from furu.worker.loop import worker_loop
from furu.worker.protocol import (
    CountSatisfiableReadyJobsResponse,
    Job,
    JobBlockedResult,
    JobCompletedResult,
    JobFailedResult,
    JobResultRequest,
    LeaseJobResponse,
)


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


class ManagerResourceLeaf(Furu[int]):
    value: int
    min_cpus: int | None = None
    max_cpus: int | None = None
    min_gpus: int | None = None
    max_gpus: int | None = None
    min_memory: int | None = None
    max_memory: int | None = None

    @property
    def resource_requirements(self) -> ResourceRequirements | None:
        return ResourceRequirements(
            cpus=(self.min_cpus, self.max_cpus),
            gpus=(self.min_gpus, self.max_gpus),
            memory=(self.min_memory, self.max_memory),
        )

    def create(self) -> int:
        return self.value


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

    job = manager.lease_job()
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

    job = manager.lease_job()
    assert isinstance(job, Job)

    assert manager.lease_job() == "wait"
    assert not manager.done.is_set()


def test_manager_counts_satisfiable_ready_jobs() -> None:
    default = ManagerLeaf(value=1)
    cpu_job = ManagerResourceLeaf(value=2, min_cpus=4)
    gpu_job = ManagerResourceLeaf(value=3, min_gpus=1)
    memory_job = ManagerResourceLeaf(value=4, min_memory=16)
    manager = Manager([default, cpu_job, gpu_job, memory_job])

    assert (
        manager.count_satisfiable_ready_jobs(
            ResourceRequest(cpus=4, gpus=0, memory=None),
            max_workers=10,
        )
        == 2
    )
    assert (
        manager.count_satisfiable_ready_jobs(
            ResourceRequest(cpus=8, gpus=2, memory=32),
            max_workers=3,
        )
        == 3
    )
    assert (
        manager.count_satisfiable_ready_jobs(
            ResourceRequest(cpus=8, gpus=2, memory=32),
            max_workers=2,
        )
        == 2
    )


def test_manager_count_satisfiable_ready_jobs_rejects_negative_max_workers() -> None:
    manager = Manager([ManagerLeaf(value=1)])

    with pytest.raises(ValueError, match="max_workers"):
        manager.count_satisfiable_ready_jobs(ResourceRequest(), max_workers=-1)


def test_manager_job_result_blocked_discovers_lazy_dependency_and_reruns_parent() -> (
    None
):
    parent = ManagerLazyParent(value=2)
    dependency = ManagerLeaf(value=2)
    manager = Manager([parent])

    parent_job = manager.lease_job()
    assert isinstance(parent_job, Job)
    assert parent_job.lease_id != parent.object_id

    manager.job_result(
        parent_job.lease_id,
        JobBlockedResult(dependencies=[ArtifactSpec.from_furu(dependency)]),
    )

    assert set(manager.ready) == {dependency.object_id}
    assert set(manager.blocked) == {parent.object_id}

    dependency_job = manager.lease_job()
    assert isinstance(dependency_job, Job)
    manager.job_result(dependency_job.lease_id, JobCompletedResult())

    assert set(manager.ready) == {parent.object_id}
    assert manager.blocked == {}


def test_manager_job_result_blocked_ignores_completed_lazy_dependency() -> None:
    parent = ManagerLazyParent(value=2)
    dependency = ManagerLeaf(value=2)
    dependency.load_or_create()
    manager = Manager([parent])

    parent_job = manager.lease_job()
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

    parent_job = manager.lease_job()
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

    first_parent_job = manager.lease_job()
    assert isinstance(first_parent_job, Job)

    manager.job_result(
        first_parent_job.lease_id,
        JobBlockedResult(dependencies=[ArtifactSpec.from_furu(dependency)]),
    )

    dependency_job = manager.lease_job()
    assert isinstance(dependency_job, Job)
    manager.job_result(dependency_job.lease_id, JobCompletedResult())

    second_parent_job = manager.lease_job()
    assert isinstance(second_parent_job, Job)
    assert second_parent_job.lease_id != first_parent_job.lease_id
    assert second_parent_job.artifact.object_id == parent.object_id

    assert set(manager.running) == {second_parent_job.lease_id}
    assert set(manager.completed) == {dependency.object_id}


def test_manager_job_result_failed_finishes_with_error() -> None:
    leaf = ManagerLeaf(value=1)
    manager = Manager([leaf])
    job = manager.lease_job()
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
            return LocalThreadWorkerPool(
                server_url=server_url,
                auth_token=auth_token,
                n_workers=1,
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


def test_local_worker_backend_uses_manager_worker_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client_calls: list[tuple[str, str, ResourceRequest, int]] = []
    worker_calls: list[tuple[str, str]] = []

    class RecordingClient:
        def __init__(self, server_url: str, *, auth_token: str) -> None:
            self.server_url = server_url
            self.auth_token = auth_token

        def count_satisfiable_ready_jobs(
            self,
            resource_request: ResourceRequest,
            *,
            max_workers: int,
        ) -> int:
            client_calls.append(
                (self.server_url, self.auth_token, resource_request, max_workers)
            )
            return 2

    def worker_loop(*, server_url: str, auth_token: str) -> None:
        worker_calls.append((server_url, auth_token))

    monkeypatch.setattr(local_backend_module, "ManagerApiClient", RecordingClient)
    monkeypatch.setattr(worker_loop_module, "worker_loop", worker_loop)

    resource_request = ResourceRequest(cpus=2)
    pool = LocalThreadWorkerBackend(
        n_workers=4,
        resource_request=resource_request,
    ).start_pool(
        server_url="http://manager.test",
        auth_token="secret-token",
        executor_dir=tmp_path,
    )
    pool.join(timeout=1)

    assert pool.n_workers == 2
    assert len(pool._threads) == 2
    assert client_calls == [
        ("http://manager.test", "secret-token", resource_request, 4)
    ]
    assert worker_calls == [
        ("http://manager.test", "secret-token"),
        ("http://manager.test", "secret-token"),
    ]


def test_manager_run_fails_when_no_backend_launches_workers() -> None:
    manager = Manager([ManagerResourceLeaf(value=1, min_cpus=2)])

    with pytest.raises(RuntimeError, match="no worker backend launched workers"):
        manager.run(worker_backends=(LocalThreadWorkerBackend(),))

    assert manager.done.is_set()


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
            return LocalThreadWorkerPool(
                server_url=server_url,
                auth_token=auth_token,
                n_workers=1,
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
    assert "furu manager finished successfully" in log_text


def test_manager_run_waits_using_worker_pool_health_check_interval() -> None:
    class RecordingDone:
        def __init__(self) -> None:
            self.timeouts: list[float | None] = []

        def wait(self, timeout: float | None = None) -> bool:
            self.timeouts.append(timeout)
            return True

        def is_set(self) -> bool:
            return False

    class RecordingPool:
        n_workers = 1
        health_check_interval = 2.5

        def __init__(self) -> None:
            self.health_checks = 0
            self.join_timeouts: list[float] = []

        def is_healthy(self) -> bool:
            self.health_checks += 1
            return True

        def join(self, *, timeout: float) -> None:
            self.join_timeouts.append(timeout)

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

    assert done.timeouts == [pytest.approx(2.5, abs=0.5)]
    assert pool.health_checks == 0
    assert pool.join_timeouts == [5]


def test_run_until_done_uses_worker_backend_manager_listen_host() -> None:
    class RecordingDone:
        def wait(self, timeout: float | None = None) -> bool:
            return True

        def is_set(self) -> bool:
            return False

    class RecordingPool:
        n_workers = 1
        health_check_interval = 1.0

        def is_healthy(self) -> bool:
            return True

        def join(self, *, timeout: float) -> None:
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
        response = httpx.post(f"{server.server_url}/lease_job")
        assert response.status_code == 401
        assert response.json() == {"detail": "invalid furu manager auth token"}

        response = httpx.post(
            f"{server.server_url}/lease_job",
            headers={"Authorization": "Bearer wrong"},
        )
        assert response.status_code == 401
        assert response.json() == {"detail": "invalid furu manager auth token"}

        response = httpx.post(
            f"{server.server_url}/lease_job",
            headers={"Authorization": f"Bearer {server.auth_token}"},
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
        worker_loop(server_url="http://127.0.0.1:1", auth_token="test-token")


def test_worker_loop_does_not_swallow_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaf = ManagerLeaf(value=1)
    job = Job(lease_id="lease-1", artifact=ArtifactSpec.from_furu(leaf))

    class TestClient:
        calls: list[str]

        def __init__(self, server_url: str, *, auth_token: str) -> None:
            self.calls = []

        def lease_job(self) -> LeaseJobResponse:
            self.calls.append("lease_job")
            return job

        def job_result(self, lease_id: str, request: JobResultRequest) -> None:
            self.calls.append("job_result")

    def ensure_single_result(obj: Furu[object]) -> None:
        raise KeyboardInterrupt

    test_client = TestClient("http://worker.test", auth_token="test-token")
    monkeypatch.setattr(
        api,
        "ManagerApiClient",
        lambda server_url, *, auth_token: test_client,
    )
    monkeypatch.setattr(
        worker_loop_module, "_ensure_single_result", ensure_single_result
    )

    with pytest.raises(KeyboardInterrupt):
        worker_loop(server_url="http://worker.test", auth_token="test-token")

    assert test_client.calls == ["lease_job"]


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

    api.ManagerApiClient("http://worker.test/", auth_token="secret-token").job_result(
        "lease-1", JobBlockedResult(dependencies=[])
    )

    assert requests == [
        (
            "POST",
            "http://worker.test/job_result/lease-1",
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
        api.ManagerApiClient(
            "http://worker.test", auth_token="secret-token"
        ).job_result(
            "lease-1",
            JobCompletedResult(),
        )


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
        api.ManagerApiClient(
            "http://worker.test", auth_token="secret-token"
        ).lease_job()


def test_client_lease_job_posts_to_lease_job_endpoint(
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

    job = api.ManagerApiClient(
        "http://worker.test/",
        auth_token="secret-token",
    ).lease_job()

    assert isinstance(job, Job)
    assert requests == [
        (
            "POST",
            "http://worker.test/lease_job",
            {"Authorization": "Bearer secret-token"},
            None,
        )
    ]


def test_client_count_satisfiable_ready_jobs_posts_to_endpoint(
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
            200,
            json=CountSatisfiableReadyJobsResponse(count=2).model_dump(mode="json"),
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr(httpx, "request", request)

    count = api.ManagerApiClient(
        "http://worker.test/",
        auth_token="secret-token",
    ).count_satisfiable_ready_jobs(
        ResourceRequest(cpus=4, gpus=1, memory=16),
        max_workers=3,
    )

    assert count == 2
    assert requests == [
        (
            "POST",
            "http://worker.test/count_satisfiable_ready_jobs",
            {"Authorization": "Bearer secret-token"},
            {
                "resource_request": {"cpus": 4, "gpus": 1, "memory": 16},
                "max_workers": 3,
            },
        )
    ]


def test_manager_api_rejects_missing_auth_token() -> None:
    app = create_manager_api_app(Manager([ManagerLeaf(value=1)]), auth_token="secret")
    client = TestClient(app)

    response = client.post("/lease_job")

    assert response.status_code == 401
    assert response.json() == {"detail": "invalid furu manager auth token"}


def test_manager_api_rejects_wrong_auth_token() -> None:
    app = create_manager_api_app(Manager([ManagerLeaf(value=1)]), auth_token="secret")
    client = TestClient(app)

    response = client.post(
        "/lease_job",
        headers={"Authorization": "Bearer wrong"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "invalid furu manager auth token"}


def test_manager_api_accepts_matching_auth_token() -> None:
    app = create_manager_api_app(Manager([ManagerLeaf(value=1)]), auth_token="secret")
    client = TestClient(app)

    response = client.post(
        "/lease_job",
        headers={"Authorization": "Bearer secret"},
    )

    assert response.status_code == 200
    assert response.json()["artifact"]["artifact_data"]["|fields"] == {"value": 1}


def test_manager_api_counts_satisfiable_ready_jobs() -> None:
    app = create_manager_api_app(
        Manager([ManagerResourceLeaf(value=1, min_cpus=2)]),
        auth_token="secret",
    )
    client = TestClient(app)

    response = client.post(
        "/count_satisfiable_ready_jobs",
        headers={"Authorization": "Bearer secret"},
        json={
            "resource_request": {"cpus": 4, "gpus": 0, "memory": None},
            "max_workers": 5,
        },
    )

    assert response.status_code == 200
    assert response.json() == {"count": 1}
