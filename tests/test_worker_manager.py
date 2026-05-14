from uuid import UUID
import os
from pathlib import Path
import stat

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import TypeAdapter, ValidationError

import furu.worker.loop as worker_loop_module
from furu import Furu
from furu.execution import api
from furu.execution.api import create_manager_api_app
from furu.execution.manager import FailedJob, Manager, RunningJob
from furu.execution.server import manager_server
from furu.metadata import ArtifactSpec
from furu.worker.backends.local import LocalThreadWorkerPool
from furu.worker.backends.slurm import SlurmResources, SlurmWorkerBackend
from furu.worker.loop import worker_loop
from furu.worker.protocol import (
    JobBlockedResult,
    JobCompletedResult,
    JobFailedResult,
    JobResultRequest,
)
from furu.worker.protocol import LeaseJobResponse, Job


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
    with pytest.raises(RuntimeError, match="failed jobs"):
        manager.raise_for_failure()


def test_manager_run_uses_worker_backend() -> None:
    class RecordingBackend:
        def __init__(self) -> None:
            self.server_urls: list[str] = []
            self.auth_tokens: list[str] = []

        def start_pool(
            self,
            *,
            server_url: str,
            auth_token: str,
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

    Manager([leaf]).run(worker_backend=backend)

    assert leaf.status() == "completed"
    assert leaf.load_or_create() == 11
    assert len(backend.server_urls) == 1
    assert backend.server_urls[0].startswith("http://127.0.0.1:")
    assert len(backend.auth_tokens) == 1
    assert backend.auth_tokens[0]


def test_manager_server_exposes_bound_host_and_port() -> None:
    manager = Manager([ManagerLeaf(value=12)])

    with manager_server(manager, bind_host="127.0.0.1", port=0) as server:
        assert server.bound_host == "127.0.0.1"
        assert server.bound_port > 0
        assert server.auth_token


def test_manager_server_uses_advertised_host_for_worker_url() -> None:
    manager = Manager([ManagerLeaf(value=12)])

    with manager_server(
        manager,
        bind_host="127.0.0.1",
        port=0,
        advertised_host="manager.cluster",
    ) as server:
        assert server.bound_host == "127.0.0.1"
        assert server.server_url == f"http://manager.cluster:{server.bound_port}"


def test_manager_run_passes_advertised_host_to_worker_backend() -> None:
    class RecordingBackend:
        server_url: str | None = None

        def start_pool(
            self,
            *,
            server_url: str,
            auth_token: str,
        ) -> LocalThreadWorkerPool:
            del auth_token
            self.server_url = server_url

            class CompletedPool:
                def is_healthy(self) -> bool:
                    return True

                def join(self, *, timeout: float) -> None:
                    del timeout

            manager.done.set()
            return CompletedPool()  # ty: ignore[invalid-return-type]

    manager = Manager([ManagerLeaf(value=12)])
    backend = RecordingBackend()

    manager.run(worker_backend=backend, advertised_host="manager.cluster")

    assert backend.server_url is not None
    assert backend.server_url.startswith("http://manager.cluster:")


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


def test_slurm_worker_backend_submits_workers_and_creates_log_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slurm_bin = _make_fake_slurm_bin(tmp_path)
    monkeypatch.setenv("PATH", f"{slurm_bin}{os.pathsep}{os.environ['PATH']}")

    log_dir = tmp_path / "logs"
    backend = SlurmWorkerBackend(
        n_workers=2,
        log_dir=log_dir,
        chdir=tmp_path,
        resources=SlurmResources(
            cpus_per_task=4,
            memory="8G",
            time="00:10:00",
            partition="debug",
        ),
    )

    pool = backend.start_pool(
        server_url="http://manager.cluster:39001",
        auth_token="secret-token",
    )

    assert log_dir.is_dir()
    assert pool.job_ids == ("1001", "1002")
    assert pool.is_healthy()

    sbatch_calls = (tmp_path / "sbatch.calls").read_text().splitlines()
    assert len(sbatch_calls) == 2
    first_call = sbatch_calls[0]
    assert "--parsable" in first_call
    assert f"--chdir {tmp_path}" in first_call
    assert f"--output {log_dir / 'furu-worker-%j.out'}" in first_call
    assert f"--error {log_dir / 'furu-worker-%j.err'}" in first_call
    assert "--cpus-per-task 4" in first_call
    assert "--mem 8G" in first_call
    assert "--time 00:10:00" in first_call
    assert "--partition debug" in first_call
    assert "-m furu.worker.cli" in first_call
    assert "--server-url http://manager.cluster:39001" in first_call
    assert "--auth-token secret-token" in first_call

    pool.join(timeout=0)

    assert (tmp_path / "squeue.calls").read_text().strip() == (
        "--noheader --jobs 1001,1002"
    )
    assert (tmp_path / "scancel.calls").read_text().strip() == "1001 1002"


def test_slurm_worker_backend_rejects_local_manager_urls(tmp_path: Path) -> None:
    backend = SlurmWorkerBackend(
        log_dir=tmp_path / "logs",
        allow_local_manager_url=False,
    )

    with pytest.raises(ValueError, match="advertised_host"):
        backend.start_pool(
            server_url="http://127.0.0.1:39001",
            auth_token="secret-token",
        )


def _make_fake_slurm_bin(tmp_path: Path) -> Path:
    slurm_bin = tmp_path / "bin"
    slurm_bin.mkdir()
    job_counter = tmp_path / "job-counter"
    job_counter.write_text("1000")

    _write_executable(
        slurm_bin / "sbatch",
        f"""#!/bin/sh
printf '%s\\n' "$*" >> {tmp_path / "sbatch.calls"}
job_id=$(cat {job_counter})
job_id=$((job_id + 1))
printf '%s\\n' "$job_id" > {job_counter}
printf '%s\\n' "$job_id"
""",
    )
    _write_executable(
        slurm_bin / "squeue",
        f"""#!/bin/sh
printf '%s\\n' "$*" >> {tmp_path / "squeue.calls"}
printf '1001 RUNNING\\n1002 RUNNING\\n'
""",
    )
    _write_executable(
        slurm_bin / "scancel",
        f"""#!/bin/sh
printf '%s\\n' "$*" >> {tmp_path / "scancel.calls"}
""",
    )
    return slurm_bin


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_manager_run_requires_explicit_worker_backend() -> None:
    manager = Manager([ManagerLeaf(value=12)])

    with pytest.raises(TypeError, match="worker_backend"):
        manager.run()  # ty: ignore[missing-argument]


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
