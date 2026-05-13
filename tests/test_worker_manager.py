import threading
from uuid import UUID

import httpx
import pytest
from pydantic import TypeAdapter, ValidationError

import furu.worker.loop as worker_loop_module
from furu import Furu
from furu.execution import api
from furu.execution.manager import FailedJob, Manager, RunningJob
from furu.execution.server import _run_until_done, manager_server
from furu.metadata import ArtifactSpec
from furu.worker.backends.local import LocalThreadWorkerBackend, LocalThreadWorkerPool
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

        def start_pool(self, *, server_url: str) -> LocalThreadWorkerPool:
            self.server_urls.append(server_url)
            return LocalThreadWorkerPool(server_url=server_url, n_workers=1)

    leaf = ManagerLeaf(value=11)
    backend = RecordingBackend()

    Manager([leaf]).run(worker_backend=backend)

    assert leaf.status() == "completed"
    assert leaf.load_or_create() == 11
    assert len(backend.server_urls) == 1
    assert backend.server_urls[0].startswith("http://127.0.0.1:")


def test_manager_server_exposes_bound_host_and_port() -> None:
    manager = Manager([ManagerLeaf(value=12)])

    with manager_server(manager, bind_host="127.0.0.1", port=0) as server:
        assert server.bound_host == "127.0.0.1"
        assert server.bound_port > 0


def test_manager_run_requires_explicit_worker_backend() -> None:
    manager = Manager([ManagerLeaf(value=12)])

    with pytest.raises(TypeError, match="worker_backend"):
        manager.run()  # ty: ignore[missing-argument]


def test_local_thread_worker_backend_requires_at_least_one_worker() -> None:
    with pytest.raises(ValueError, match="at least one worker"):
        LocalThreadWorkerBackend(n_workers=0)


def test_local_thread_worker_pool_stop_requests_worker_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    seen_stop_events: list[threading.Event] = []

    def fake_worker_loop(
        *,
        server_url: str,
        stop_event: threading.Event | None = None,
    ) -> None:
        assert server_url == "http://worker.test"
        assert stop_event is not None
        seen_stop_events.append(stop_event)
        started.set()
        stop_event.wait(timeout=10)

    monkeypatch.setattr(worker_loop_module, "worker_loop", fake_worker_loop)

    pool = LocalThreadWorkerPool(server_url="http://worker.test", n_workers=1)
    try:
        assert started.wait(timeout=1)
        assert not pool.join(timeout=0.01)

        pool.stop()

        assert pool.join(timeout=1)
        assert [event.is_set() for event in seen_stop_events] == [True]
    finally:
        pool.stop()
        pool.join(timeout=1)


def test_run_until_done_stops_worker_pool_before_join() -> None:
    events: list[str] = []

    class TestWorkerPool:
        def is_healthy(self) -> bool:
            return True

        def stop(self) -> None:
            events.append("stop")

        def join(self, *, timeout: float) -> bool:
            events.append(f"join:{timeout:g}")
            return True

    class TestWorkerBackend:
        def start_pool(self, *, server_url: str) -> TestWorkerPool:
            assert server_url.startswith("http://127.0.0.1:")
            return TestWorkerPool()

    manager = Manager([ManagerLeaf(value=1)])
    manager.done.set()

    _run_until_done(
        manager,
        worker_backend=TestWorkerBackend(),
        host="127.0.0.1",
        port=0,
    )

    assert events == ["stop", "join:5"]


def test_run_until_done_raises_when_worker_pool_does_not_stop() -> None:
    class TestWorkerPool:
        def is_healthy(self) -> bool:
            return True

        def stop(self) -> None:
            pass

        def join(self, *, timeout: float) -> bool:
            return False

    class TestWorkerBackend:
        def start_pool(self, *, server_url: str) -> TestWorkerPool:
            return TestWorkerPool()

    manager = Manager([ManagerLeaf(value=1)])
    manager.done.set()

    with pytest.raises(RuntimeError, match="worker backend did not stop"):
        _run_until_done(
            manager,
            worker_backend=TestWorkerBackend(),
            host="127.0.0.1",
            port=0,
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
        worker_loop(server_url="http://127.0.0.1:1")


def test_worker_loop_exits_when_stop_event_is_already_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TestClient:
        def __init__(self, server_url: str) -> None:
            pass

        def lease_job(self) -> LeaseJobResponse:
            raise AssertionError("worker should stop before polling for a job")

    monkeypatch.setattr(api, "ManagerApiClient", TestClient)
    stop_event = threading.Event()
    stop_event.set()

    worker_loop(server_url="http://worker.test", stop_event=stop_event)


def test_worker_loop_does_not_swallow_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaf = ManagerLeaf(value=1)
    job = Job(lease_id="lease-1", artifact=ArtifactSpec.from_furu(leaf))

    class TestClient:
        calls: list[str]

        def __init__(self, server_url: str) -> None:
            self.calls = []

        def lease_job(self) -> LeaseJobResponse:
            self.calls.append("lease_job")
            return job

        def job_result(self, lease_id: str, request: JobResultRequest) -> None:
            self.calls.append("job_result")

    def ensure_single_result(obj: Furu[object]) -> None:
        raise KeyboardInterrupt

    test_client = TestClient("http://worker.test")
    monkeypatch.setattr(api, "ManagerApiClient", lambda server_url: test_client)
    monkeypatch.setattr(
        worker_loop_module, "_ensure_single_result", ensure_single_result
    )

    with pytest.raises(KeyboardInterrupt):
        worker_loop(server_url="http://worker.test")

    assert test_client.calls == ["lease_job"]


def test_client_job_result_uses_job_result_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[tuple[str, str, object | None]] = []

    def request(
        method: str,
        url: str,
        *,
        json: object | None,
        timeout: float,
    ) -> httpx.Response:
        requests.append((method, url, json))
        return httpx.Response(
            200, json={"ok": True}, request=httpx.Request(method, url)
        )

    monkeypatch.setattr(httpx, "request", request)

    api.ManagerApiClient("http://worker.test/").job_result(
        "lease-1", JobBlockedResult(dependencies=[])
    )

    assert requests == [
        (
            "POST",
            "http://worker.test/job_result/lease-1",
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
        json: object | None,
        timeout: float,
    ) -> httpx.Response:
        return httpx.Response(
            200, json={"ok": False}, request=httpx.Request(method, url)
        )

    monkeypatch.setattr(httpx, "request", request)

    with pytest.raises(ValidationError, match="Input should be True"):
        api.ManagerApiClient("http://worker.test").job_result(
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
        json: object | None,
        timeout: float,
    ) -> httpx.Response:
        return httpx.Response(200, request=httpx.Request(method, url))

    monkeypatch.setattr(httpx, "request", request)

    with pytest.raises(RuntimeError, match="returned an empty response"):
        api.ManagerApiClient("http://worker.test").lease_job()


def test_client_lease_job_posts_to_lease_job_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaf = ManagerLeaf(value=1)
    requests: list[tuple[str, str, object | None]] = []

    def request(
        method: str,
        url: str,
        *,
        json: object | None,
        timeout: float,
    ) -> httpx.Response:
        requests.append((method, url, json))
        return httpx.Response(
            200,
            json={
                "lease_id": "lease-1",
                "artifact": ArtifactSpec.from_furu(leaf).model_dump(mode="json"),
            },
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr(httpx, "request", request)

    job = api.ManagerApiClient("http://worker.test/").lease_job()

    assert isinstance(job, Job)
    assert requests == [("POST", "http://worker.test/lease_job", None)]
