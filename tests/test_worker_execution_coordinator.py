import hashlib
import logging
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

import pytest
from pydantic import TypeAdapter, ValidationError
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK, InvalidStatus
from websockets.headers import build_authorization_basic
from websockets.sync.client import ClientConnection, connect
from websockets.sync.server import ServerConnection, serve

import furu
import furu.worker.loop as worker_loop_module
from furu import GiB, Metadata, Requires, Spec, Throttle, at_least
from furu.config import get_config
from furu.dag import _add_to_dag
from furu.execution.execution_coordinator import (
    ExecutionCoordinator,
    FailedJob,
    RunningJob,
)
from furu.execution.server import execution_coordinator_server
from furu.logging import _scoped_log_files
from furu.metadata import ArtifactSpec
from furu.provenance import (
    EnvironmentIdentity,
    GitIdentity,
    SubmitContext,
    SubmitProvenance,
)
from furu.resources import ResourceRequest
from furu.storage._layout import execution_coordinator_log_path_in
from furu.worker.backends.local import LocalThreadWorkerBackend, LocalThreadWorkerPool
from furu.worker.loop import worker_loop
from furu.worker.protocol import (
    HelloMessage,
    Job,
    JobBlockedResult,
    JobCompletedResult,
    JobFailedResult,
    JobResult,
    job_result_adapter,
)

ANY_RESOURCES = ResourceRequest()


def _submit_provenance() -> SubmitProvenance:
    # Real environment identity so worker-side lock-hash verification passes;
    # the git half is a stub since these tests never read it back.
    return SubmitProvenance(
        git=GitIdentity(
            commit="0" * 40,
            branch=None,
            remote=None,
            repo_root=".",
            dirty=False,
            diff_stats=None,
        ),
        environment=EnvironmentIdentity.capture(),
        snapshot_id=None,
        submitted=SubmitContext.capture(),
    )


def _job(obj: Spec[Any]) -> Job:
    return Job(
        artifacts=[ArtifactSpec.from_furu(obj)],
        provenance=_submit_provenance(),
    )


def _artifact(job: Job | None) -> ArtifactSpec:
    assert isinstance(job, Job)
    (artifact,) = job.artifacts
    return artifact


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
    objs: Sequence[furu.Spec[Any]],
    *,
    max_retries_per_object: int | None = None,
) -> ExecutionCoordinator:
    if max_retries_per_object is None:
        max_retries_per_object = get_config().worker.max_retries_per_object
    coordinator = ExecutionCoordinator(max_retries_per_object=max_retries_per_object)
    coordinator.submit_provenance = _submit_provenance()
    _add_to_dag(coordinator, objs)
    digest = hashlib.blake2s(digest_size=16)
    for obj in objs:
        digest.update(obj.object_id.encode("utf-8"))
        digest.update(b"\0")
    coordinator.executor_id = digest.hexdigest()
    return coordinator


def _lease_job(
    coordinator: ExecutionCoordinator, *, resources: ResourceRequest = ANY_RESOURCES
) -> Job | None:
    return coordinator.lease_job(resources=resources, worker=f"test-worker-{uuid4()}")


def _no_satisfiable_job(
    coordinator: ExecutionCoordinator, *, resources: ResourceRequest = ANY_RESOURCES
) -> bool:
    return coordinator.count_satisfiable_jobs(resources=resources, max_workers=1) == 0


@dataclass(slots=True)
class _ScriptedServer:
    """A hand-rolled coordinator that plays a fixed sequence of messages."""

    server_url: str
    hellos: list[HelloMessage] = field(default_factory=list)
    results: list[JobResult] = field(default_factory=list)


@contextmanager
def _scripted_worker_server(
    jobs: Sequence[Job],
    *,
    hold_open: bool = False,
) -> Iterator[_ScriptedServer]:
    record = _ScriptedServer(server_url="")

    def handler(connection: ServerConnection) -> None:
        hello = HelloMessage.model_validate_json(connection.recv(timeout=5))
        record.hellos.append(hello)
        try:
            for job in jobs:
                connection.send(job.model_dump_json())
                record.results.append(
                    job_result_adapter.validate_json(connection.recv(timeout=5))
                )
            if hold_open:
                # Linger until the worker hangs up (idle timeout, crash, ...).
                connection.recv(timeout=5)
        except (TimeoutError, ConnectionClosed):
            pass

    server = serve(handler, "127.0.0.1", 0)
    record.server_url = f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield record
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _connect_worker(
    server: Any,
    *,
    auth_token: str | None = None,
    worker: str = "raw-test-worker",
    resources: ResourceRequest | None = None,
) -> ClientConnection:
    token = server.auth_token if auth_token is None else auth_token
    connection = connect(
        server.server_url,
        additional_headers={"Authorization": build_authorization_basic("furu", token)},
    )
    connection.send(
        HelloMessage(
            worker=worker,
            backend="test",
            resources=resources or ResourceRequest(),
        ).model_dump_json()
    )
    return connection


def _wait_until(condition: Callable[[], bool], *, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not met in time")
        time.sleep(0.01)


def _complete_one_job_over_ws(server_url: str, auth_token: str) -> None:
    connection = connect(
        server_url,
        additional_headers={
            "Authorization": build_authorization_basic("furu", auth_token)
        },
    )
    with connection:
        connection.send(
            HelloMessage(
                worker="recording-worker",
                backend="test",
                resources=ResourceRequest(),
            ).model_dump_json()
        )
        while True:
            try:
                message = connection.recv(timeout=10)
            except ConnectionClosed:
                return
            Job.model_validate_json(message)
            connection.send(JobCompletedResult().model_dump_json())


class ExecutionCoordinatorLeaf(Spec[int]):
    value: int

    def create(self) -> int:
        return self.value


class FlakyExecutionCoordinatorLeaf(Spec[int]):
    value: int
    attempts_by_value: ClassVar[dict[int, int]] = {}

    def create(self) -> int:
        attempts = type(self).attempts_by_value.get(self.value, 0) + 1
        type(self).attempts_by_value[self.value] = attempts
        if attempts == 1:
            raise RuntimeError(f"temporary failure: {self.value}")
        return self.value


class LimitedExecutionCoordinatorLeaf(Spec[int]):
    value: int
    throttle = Throttle(max_running=2)

    def create(self) -> int:
        return self.value


class BatchedCoordinatorLeaf(furu.Spec[int]):
    value: int
    group: str = "g"
    cap: int = 10
    batch_sizes: ClassVar[list[int]] = []

    def batch_key(self) -> tuple[str, int]:
        return (self.group, self.cap)

    @furu.batched(batch_key)
    def create(objs: list["BatchedCoordinatorLeaf"]) -> list[int]:
        BatchedCoordinatorLeaf.batch_sizes.append(len(objs))
        return [obj.value for obj in objs]


class ThrottledBatchedCoordinatorLeaf(furu.Spec[int]):
    value: int
    throttle = Throttle(max_running=2)

    def batch_key(self) -> tuple[None, int]:
        return (None, 3)

    @furu.batched(batch_key)
    def create(objs: list["ThrottledBatchedCoordinatorLeaf"]) -> list[int]:
        return [obj.value for obj in objs]


class ExecutionCoordinatorParent(Spec[int]):
    child: ExecutionCoordinatorLeaf

    def create(self) -> int:
        return self.child.create() + 1


class ExecutionCoordinatorLazyParent(Spec[int]):
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

    job = coordinator.lease_job(resources=ANY_RESOURCES, worker="test-worker")
    assert isinstance(job, Job)
    assert set(coordinator.running) == {leaf.object_id}
    running_job = coordinator.running[leaf.object_id]
    assert isinstance(running_job, RunningJob)
    assert running_job.node.obj is leaf

    coordinator.job_result(leaf.object_id, JobCompletedResult())

    assert coordinator.running == {}
    assert set(coordinator.completed) == {leaf.object_id}
    assert set(coordinator.ready) == {parent.object_id}
    assert coordinator.blocked == {}


def test_execution_coordinator_has_no_lease_when_only_running_jobs_can_unblock_work() -> (
    None
):
    leaf = ExecutionCoordinatorLeaf(value=1)
    parent = ExecutionCoordinatorParent(child=leaf)
    coordinator = _new_execution_coordinator([parent])

    job = _lease_job(coordinator)
    assert isinstance(job, Job)

    assert _no_satisfiable_job(coordinator)
    assert not coordinator.done.is_set()


def test_execution_coordinator_job_result_blocked_discovers_lazy_dependency_and_reruns_parent() -> (
    None
):
    parent = ExecutionCoordinatorLazyParent(value=2)
    dependency = ExecutionCoordinatorLeaf(value=2)
    coordinator = _new_execution_coordinator([parent])

    parent_job = coordinator.lease_job(resources=ANY_RESOURCES, worker="test-worker")
    assert isinstance(parent_job, Job)

    coordinator.job_result(
        parent.object_id,
        JobBlockedResult(dependencies=[ArtifactSpec.from_furu(dependency)]),
    )

    assert set(coordinator.ready) == {dependency.object_id}
    assert set(coordinator.blocked) == {parent.object_id}

    dependency_job = coordinator.lease_job(
        resources=ANY_RESOURCES, worker="test-worker"
    )
    assert isinstance(dependency_job, Job)
    coordinator.job_result(dependency.object_id, JobCompletedResult())

    assert set(coordinator.ready) == {parent.object_id}
    assert coordinator.blocked == {}


def test_execution_coordinator_job_result_blocked_ignores_completed_lazy_dependency() -> (
    None
):
    parent = ExecutionCoordinatorLazyParent(value=2)
    dependency = ExecutionCoordinatorLeaf(value=2)
    dependency.create()
    coordinator = _new_execution_coordinator([parent])

    parent_job = coordinator.lease_job(resources=ANY_RESOURCES, worker="test-worker")
    assert isinstance(parent_job, Job)

    coordinator.job_result(
        parent.object_id,
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

    parent_job = coordinator.lease_job(resources=ANY_RESOURCES, worker="test-worker")
    assert isinstance(parent_job, Job)

    coordinator.job_result(
        parent.object_id,
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


def test_execution_coordinator_re_leases_blocked_job_after_dependency_completes() -> (
    None
):
    parent = ExecutionCoordinatorLazyParent(value=2)
    dependency = ExecutionCoordinatorLeaf(value=2)
    coordinator = _new_execution_coordinator([parent])

    first_parent_job = coordinator.lease_job(
        resources=ANY_RESOURCES, worker="test-worker"
    )
    assert isinstance(first_parent_job, Job)

    coordinator.job_result(
        parent.object_id,
        JobBlockedResult(dependencies=[ArtifactSpec.from_furu(dependency)]),
    )

    dependency_job = coordinator.lease_job(
        resources=ANY_RESOURCES, worker="test-worker"
    )
    assert isinstance(dependency_job, Job)
    coordinator.job_result(dependency.object_id, JobCompletedResult())

    second_parent_job = coordinator.lease_job(
        resources=ANY_RESOURCES, worker="test-worker"
    )
    assert isinstance(second_parent_job, Job)
    assert _artifact(second_parent_job).object_id == parent.object_id

    assert set(coordinator.running) == {parent.object_id}
    assert set(coordinator.completed) == {dependency.object_id}


def test_execution_coordinator_job_result_failed_finishes_with_error() -> None:
    leaf = ExecutionCoordinatorLeaf(value=1)
    coordinator = _new_execution_coordinator([leaf], max_retries_per_object=0)
    job = coordinator.lease_job(resources=ANY_RESOURCES, worker="test-worker")
    assert isinstance(job, Job)

    coordinator.job_result(leaf.object_id, JobFailedResult(error="boom"))

    assert coordinator.running == {}
    assert set(coordinator.failed) == {leaf.object_id}
    failed_job = coordinator.failed[leaf.object_id]
    assert failed_job.failed_attempts == 1
    assert isinstance(failed_job, FailedJob)
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

    first_job = coordinator.lease_job(resources=ANY_RESOURCES, worker="test-worker")
    assert isinstance(first_job, Job)
    with _captured_furu_logs(caplog):
        coordinator.job_result(leaf.object_id, JobFailedResult(error="boom 1"))

    assert set(coordinator.failed) == {leaf.object_id}
    failed_job = coordinator.failed[leaf.object_id]
    assert failed_job.failed_attempts == 1
    assert failed_job.error == "boom 1"
    assert set(coordinator.ready) == {leaf.object_id}
    assert not coordinator.done.is_set()

    second_job = coordinator.lease_job(resources=ANY_RESOURCES, worker="test-worker")
    assert isinstance(second_job, Job)
    with _captured_furu_logs(caplog):
        coordinator.job_result(leaf.object_id, JobFailedResult(error="boom 2"))

    assert set(coordinator.failed) == {leaf.object_id}
    failed_job = coordinator.failed[leaf.object_id]
    assert failed_job.failed_attempts == 2
    assert failed_job.error == "boom 2"
    assert set(coordinator.ready) == {leaf.object_id}
    assert not coordinator.done.is_set()

    third_job = coordinator.lease_job(resources=ANY_RESOURCES, worker="test-worker")
    assert isinstance(third_job, Job)
    coordinator.job_result(leaf.object_id, JobFailedResult(error="boom 3"))

    assert coordinator.running == {}
    assert coordinator.ready == {}
    assert set(coordinator.failed) == {leaf.object_id}
    failed_job = coordinator.failed[leaf.object_id]
    assert failed_job.failed_attempts == 3
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
    assert f"object_id={leaf.object_id}" in log_text
    assert "failed_retry=1 failed=0" in log_text
    assert "failed_retry=0 failed=1" in log_text


def test_execution_coordinator_job_result_failed_retry_can_later_complete() -> None:
    leaf = ExecutionCoordinatorLeaf(value=1)
    coordinator = _new_execution_coordinator([leaf], max_retries_per_object=1)

    first_job = coordinator.lease_job(resources=ANY_RESOURCES, worker="test-worker")
    assert isinstance(first_job, Job)
    coordinator.job_result(leaf.object_id, JobFailedResult(error="boom"))

    failed_job = coordinator.failed[leaf.object_id]
    assert failed_job.failed_attempts == 1

    retry_job = coordinator.lease_job(resources=ANY_RESOURCES, worker="test-worker")
    assert isinstance(retry_job, Job)
    coordinator.job_result(leaf.object_id, JobCompletedResult())

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


class GpuLeaf(Spec[int]):
    value: int

    def metadata(self) -> Metadata:
        return Metadata(requires=Requires(gpus=at_least(1)))

    def create(self) -> int:
        return self.value


class CpuOnlyLeaf(Spec[int]):
    value: int

    def create(self) -> int:
        return self.value


class MemoryLeaf(Spec[int]):
    value: int

    def metadata(self) -> Metadata:
        return Metadata(requires=Requires(ram=GiB(8)))

    def create(self) -> int:
        return self.value


class DynamicCpuSeed(Spec[int]):
    value: int
    create_calls: ClassVar[list[int]] = []

    def create(self) -> int:
        type(self).create_calls.append(self.value)
        return self.value


class DynamicGpuAfterSeed(Spec[int]):
    parent: DynamicCpuSeed
    value: int
    create_calls: ClassVar[list[int]] = []

    def metadata(self) -> Metadata:
        return Metadata(requires=Requires(gpus=at_least(1)))

    def create(self) -> int:
        type(self).create_calls.append(self.value)
        return self.parent.create() + self.value


class DynamicCpuAfterGpu(Spec[int]):
    parent: DynamicGpuAfterSeed
    value: int
    create_calls: ClassVar[list[int]] = []

    def create(self) -> int:
        type(self).create_calls.append(self.value)
        return self.parent.create() + self.value


class DynamicGpuAfterCpu(Spec[int]):
    parent: DynamicCpuAfterGpu
    value: int
    create_calls: ClassVar[list[int]] = []

    def metadata(self) -> Metadata:
        return Metadata(requires=Requires(gpus=at_least(1)))

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


def test_count_satisfiable_jobs_returns_zero_when_coordinator_is_done() -> None:
    coordinator = _new_execution_coordinator([ExecutionCoordinatorLeaf(value=1)])
    assert (
        coordinator.count_satisfiable_jobs(resources=ResourceRequest(), max_workers=10)
        == 1
    )

    coordinator.fail("execution interrupted")

    assert (
        coordinator.count_satisfiable_jobs(resources=ResourceRequest(), max_workers=10)
        == 0
    )


def test_worker_cap_limits_satisfiable_jobs_and_leases() -> None:
    limited = [LimitedExecutionCoordinatorLeaf(value=value) for value in range(3)]
    uncapped = ExecutionCoordinatorLeaf(value=10)
    coordinator = _new_execution_coordinator([*limited, uncapped])

    assert (
        coordinator.count_satisfiable_jobs(resources=ResourceRequest(), max_workers=10)
        == 3
    )

    first = _lease_job(coordinator, resources=ResourceRequest())
    second = _lease_job(coordinator, resources=ResourceRequest())
    third = _lease_job(coordinator, resources=ResourceRequest())

    assert isinstance(first, Job)
    assert isinstance(second, Job)
    assert isinstance(third, Job)
    limited_ids = {obj.object_id for obj in limited}
    leased_limited_ids = {_artifact(first).object_id, _artifact(second).object_id}
    assert leased_limited_ids < limited_ids
    assert _artifact(third).object_id == uncapped.object_id
    assert (
        coordinator.count_satisfiable_jobs(resources=ResourceRequest(), max_workers=10)
        == 0
    )
    assert _no_satisfiable_job(coordinator, resources=ResourceRequest())

    coordinator.job_result(_artifact(first).object_id, JobCompletedResult())
    fourth = _lease_job(coordinator, resources=ResourceRequest())

    assert isinstance(fourth, Job)
    assert _artifact(fourth).object_id in limited_ids - leased_limited_ids


def test_lease_job_assembles_same_key_batched_group_into_one_job(
    caplog: pytest.LogCaptureFixture,
) -> None:
    objs = [BatchedCoordinatorLeaf(value=value) for value in range(3)]
    coordinator = _new_execution_coordinator(objs)

    with _captured_furu_logs(caplog):
        job = _lease_job(coordinator)

    assert isinstance(job, Job)
    assert len(job.artifacts) == 3
    assert {artifact.object_id for artifact in job.artifacts} == set(
        coordinator.running
    )
    assert coordinator.ready == {}
    detail = caplog.records[-1].__dict__["_furu_detail"]
    assert detail["object_ids"] == ",".join(obj.object_id for obj in objs)

    for artifact in job.artifacts:
        coordinator.job_result(artifact.object_id, JobCompletedResult())

    assert set(coordinator.completed) == {obj.object_id for obj in objs}
    assert coordinator.done.is_set()


def test_lease_job_groups_only_matching_batch_keys() -> None:
    first_x = BatchedCoordinatorLeaf(value=1, group="x")
    only_y = BatchedCoordinatorLeaf(value=2, group="y")
    second_x = BatchedCoordinatorLeaf(value=3, group="x")
    coordinator = _new_execution_coordinator([first_x, only_y, second_x])

    x_job = _lease_job(coordinator)
    y_job = _lease_job(coordinator)

    assert isinstance(x_job, Job) and isinstance(y_job, Job)
    assert {artifact.object_id for artifact in x_job.artifacts} == {
        first_x.object_id,
        second_x.object_id,
    }
    assert [artifact.object_id for artifact in y_job.artifacts] == [only_y.object_id]


def test_lease_job_chunks_batched_group_to_the_cap() -> None:
    objs = [BatchedCoordinatorLeaf(value=value, cap=2) for value in range(5)]
    coordinator = _new_execution_coordinator(objs)

    jobs = [_lease_job(coordinator) for _ in range(3)]

    member_counts = [len(job.artifacts) for job in jobs if isinstance(job, Job)]
    assert member_counts == [2, 2, 1]
    assert coordinator.ready == {}


def test_throttle_limits_concurrent_batches_not_members() -> None:
    objs = [ThrottledBatchedCoordinatorLeaf(value=value) for value in range(8)]
    coordinator = _new_execution_coordinator(objs)

    first = _lease_job(coordinator)
    second = _lease_job(coordinator)

    assert isinstance(first, Job) and len(first.artifacts) == 3
    assert isinstance(second, Job) and len(second.artifacts) == 3
    assert _no_satisfiable_job(coordinator)

    for artifact in first.artifacts:
        coordinator.job_result(artifact.object_id, JobCompletedResult())
    third = _lease_job(coordinator)
    assert isinstance(third, Job) and len(third.artifacts) == 2


def test_count_satisfiable_jobs_counts_throttled_batches() -> None:
    objs = [ThrottledBatchedCoordinatorLeaf(value=value) for value in range(9)]
    coordinator = _new_execution_coordinator(objs)

    assert (
        coordinator.count_satisfiable_jobs(resources=ANY_RESOURCES, max_workers=10) == 2
    )


def test_count_satisfiable_jobs_counts_batched_groups() -> None:
    objs = [BatchedCoordinatorLeaf(value=value) for value in range(4)]
    coordinator = _new_execution_coordinator(objs)

    assert (
        coordinator.count_satisfiable_jobs(resources=ANY_RESOURCES, max_workers=10) == 1
    )


def test_batched_group_failure_retries_each_member() -> None:
    objs = [BatchedCoordinatorLeaf(value=value) for value in range(2)]
    coordinator = _new_execution_coordinator(objs, max_retries_per_object=1)

    job = _lease_job(coordinator)
    assert isinstance(job, Job)
    for artifact in job.artifacts:
        coordinator.job_result(artifact.object_id, JobFailedResult(error="boom"))

    assert set(coordinator.failed) == {obj.object_id for obj in objs}
    assert all(record.failed_attempts == 1 for record in coordinator.failed.values())
    assert set(coordinator.ready) == {obj.object_id for obj in objs}


def test_execution_coordinator_runs_batched_specs_as_one_batch() -> None:
    BatchedCoordinatorLeaf.batch_sizes.clear()
    objs = [BatchedCoordinatorLeaf(value=value, group="e2e") for value in range(3)]

    results = furu.create(objs, on=[LocalThreadWorkerBackend()])

    assert results == [0, 1, 2]
    assert BatchedCoordinatorLeaf.batch_sizes == [3]


def test_worker_lost_requeues_running_lease_without_counting_failure() -> None:
    objs = [LimitedExecutionCoordinatorLeaf(value=value) for value in range(3)]
    coordinator = _new_execution_coordinator(objs)

    first = coordinator.lease_job(resources=ResourceRequest(), worker="worker-1")
    second = coordinator.lease_job(resources=ResourceRequest(), worker="worker-2")

    assert isinstance(first, Job)
    assert isinstance(second, Job)
    assert (
        coordinator.count_satisfiable_jobs(resources=ResourceRequest(), max_workers=10)
        == 0
    )

    coordinator.worker_lost("worker-1")

    assert set(coordinator.running) == {_artifact(second).object_id}
    assert _artifact(first).object_id in coordinator.ready
    assert coordinator.failed == {}
    assert (
        coordinator.count_satisfiable_jobs(resources=ResourceRequest(), max_workers=10)
        == 1
    )


def test_job_result_after_worker_lost_is_ignored() -> None:
    leaf = ExecutionCoordinatorLeaf(value=1)
    coordinator = _new_execution_coordinator([leaf])
    job = coordinator.lease_job(resources=ResourceRequest(), worker="worker-1")
    assert isinstance(job, Job)

    coordinator.worker_lost("worker-1")
    coordinator.job_result(leaf.object_id, JobCompletedResult())

    assert coordinator.running == {}
    assert set(coordinator.ready) == {leaf.object_id}
    assert coordinator.completed == {}


def test_lease_job_filters_by_worker_resources() -> None:
    cpu_leaf = CpuOnlyLeaf(value=1)
    gpu_leaf = GpuLeaf(value=2)
    coordinator = _new_execution_coordinator([cpu_leaf, gpu_leaf])

    cpu_job = _lease_job(coordinator, resources=ResourceRequest(gpus=0))
    assert isinstance(cpu_job, Job)
    assert _artifact(cpu_job).object_id == cpu_leaf.object_id

    assert _no_satisfiable_job(coordinator, resources=ResourceRequest(gpus=0))

    gpu_job = _lease_job(coordinator, resources=ResourceRequest(gpus=1))
    assert isinstance(gpu_job, Job)
    assert _artifact(gpu_job).object_id == gpu_leaf.object_id


def test_lease_job_filters_by_worker_memory_gib() -> None:
    memory_leaf = MemoryLeaf(value=1)
    coordinator = _new_execution_coordinator([memory_leaf])

    assert _no_satisfiable_job(coordinator, resources=ResourceRequest(memory_gib=7))

    memory_job = coordinator.lease_job(
        resources=ResourceRequest(memory_gib=8), worker="test-worker"
    )
    assert isinstance(memory_job, Job)
    assert _artifact(memory_job).object_id == memory_leaf.object_id


def test_execution_coordinator_run_completes_later_resource_stages_on_local_workers() -> (
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


def test_execution_coordinator_run_fails_when_local_worker_crashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def crashing_worker_loop(
        *,
        server_url: str,
        auth_token: str,
        resource_request: ResourceRequest,
        idle_timeout: float | None,
        component: str,
        backend: str,
    ) -> None:
        raise RuntimeError("worker boom")

    monkeypatch.setattr(worker_loop_module, "worker_loop", crashing_worker_loop)

    with pytest.raises(
        RuntimeError,
        match="local worker thread crashed: RuntimeError: worker boom",
    ):
        ExecutionCoordinator.run(
            [ExecutionCoordinatorLeaf(value=42)],
            worker_backends=(LocalThreadWorkerBackend(max_workers=1),),
        )


def test_execution_coordinator_run_uses_worker_backend() -> None:
    class RecordingBackend:
        execution_coordinator_listen_host = "0.0.0.0"

        def __init__(self) -> None:
            self.bound_ports: list[int] = []
            self.auth_tokens: list[str] = []
            self.provenances: list[SubmitProvenance] = []

        def start_pool(
            self,
            *,
            coordinator: ExecutionCoordinator,
            bound_port: int,
            auth_token: str,
            executor_dir: Path,
            provenance: SubmitProvenance,
        ) -> LocalThreadWorkerPool:
            self.bound_ports.append(bound_port)
            self.auth_tokens.append(auth_token)
            self.provenances.append(provenance)
            return LocalThreadWorkerBackend(
                max_workers=1,
                resource_request=ResourceRequest(),
            ).start_pool(
                coordinator=coordinator,
                bound_port=bound_port,
                auth_token=auth_token,
                executor_dir=executor_dir,
                provenance=provenance,
            )

    leaf = ExecutionCoordinatorLeaf(value=11)
    objs = [leaf]
    backend = RecordingBackend()

    returned = ExecutionCoordinator.run(objs, worker_backends=(backend,))

    assert returned is objs
    assert leaf.status == "done"
    assert leaf.create() == 11
    assert len(backend.bound_ports) == 1
    (pool_provenance,) = backend.provenances
    assert isinstance(pool_provenance, SubmitProvenance)
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
            coordinator: ExecutionCoordinator,
            bound_port: int,
            auth_token: str,
            executor_dir: Path,
            provenance: SubmitProvenance,
        ) -> LocalThreadWorkerPool:
            self.executor_dirs.append(executor_dir)
            return LocalThreadWorkerBackend(
                max_workers=1,
                resource_request=ResourceRequest(),
            ).start_pool(
                coordinator=coordinator,
                bound_port=bound_port,
                auth_token=auth_token,
                executor_dir=executor_dir,
                provenance=provenance,
            )

    leaf = ExecutionCoordinatorLeaf(value=12)
    expected_executor_dir = _new_execution_coordinator([leaf]).executor_dir
    backend = RecordingBackend()

    ExecutionCoordinator.run([leaf], worker_backends=(backend,))

    assert backend.executor_dirs == [expected_executor_dir]


def test_top_level_create_runs_dag_on_worker_backends() -> None:
    leaf = ExecutionCoordinatorLeaf(value=21)

    assert furu.create(leaf, on=(LocalThreadWorkerBackend(),)) == 21
    assert leaf.status == "done"
    assert furu.create([leaf], on=(LocalThreadWorkerBackend(),)) == [21]


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
    assert f"leased {leaf._log_label} ×1 to local-worker-0" in log_text
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
            coordinator: ExecutionCoordinator,
            bound_port: int,
            auth_token: str,
            executor_dir: Path,
            provenance: SubmitProvenance,
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
            coordinator: ExecutionCoordinator,
            bound_port: int,
            auth_token: str,
            executor_dir: Path,
            provenance: SubmitProvenance,
        ) -> RecordingPool:
            self.pool.events.append("start_pool")
            server_url = f"ws://127.0.0.1:{bound_port}"

            def complete_job() -> None:
                try:
                    _complete_one_job_over_ws(server_url, auth_token)
                except BaseException as exc:
                    coordinator.fail(f"recording backend failed: {exc!r}")

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
            coordinator: ExecutionCoordinator,
            bound_port: int,
            auth_token: str,
            executor_dir: Path,
            provenance: SubmitProvenance,
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
            coordinator: ExecutionCoordinator,
            bound_port: int,
            auth_token: str,
            executor_dir: Path,
            provenance: SubmitProvenance,
        ) -> RecordingPool:
            server_url = f"ws://{self.execution_coordinator_listen_host}:{bound_port}"
            self.server_urls.append(server_url)
            pool = RecordingPool()

            def complete_job() -> None:
                try:
                    _complete_one_job_over_ws(server_url, auth_token)
                except BaseException as exc:
                    coordinator.fail(f"recording backend failed: {exc!r}")

            pool.worker_thread = threading.Thread(target=complete_job)
            pool.worker_thread.start()
            return pool

    backend = RecordingBackend()

    ExecutionCoordinator.run(
        [ExecutionCoordinatorLeaf(value=15)], worker_backends=(backend,), port=0
    )

    assert len(backend.server_urls) == 1
    assert backend.server_urls[0].startswith("ws://127.0.0.1:")


def test_execution_coordinator_server_exposes_bound_host_and_port() -> None:
    coordinator = _new_execution_coordinator([ExecutionCoordinatorLeaf(value=12)])

    with execution_coordinator_server(
        coordinator, bind_host="127.0.0.1", port=0
    ) as server:
        assert server.bound_host == "127.0.0.1"
        assert server.bound_port > 0
        assert server.auth_token


def test_execution_coordinator_server_rejects_connections_without_auth_token() -> None:
    coordinator = _new_execution_coordinator([ExecutionCoordinatorLeaf(value=12)])

    with execution_coordinator_server(
        coordinator, bind_host="127.0.0.1", port=0
    ) as server:
        with pytest.raises(InvalidStatus) as no_token:
            connect(server.server_url)
        assert no_token.value.response.status_code == 401

        with pytest.raises(InvalidStatus) as wrong_token:
            connect(
                server.server_url,
                additional_headers={
                    "Authorization": build_authorization_basic("furu", "wrong")
                },
            )
        assert wrong_token.value.response.status_code == 401

        connection = _connect_worker(server)
        connection.close()


def test_worker_protocol_round_trip_over_server() -> None:
    leaf = ExecutionCoordinatorLeaf(value=1)
    coordinator = _new_execution_coordinator([leaf])

    with (
        execution_coordinator_server(
            coordinator, bind_host="127.0.0.1", port=0
        ) as server,
        _connect_worker(server) as connection,
    ):
        job = Job.model_validate_json(connection.recv(timeout=5))
        (artifact,) = job.artifacts
        assert artifact.object_id == leaf.object_id
        assert artifact.artifact_data["|fields"] == {"value": 1}
        assert artifact.object_id in coordinator.running

        connection.send(JobCompletedResult().model_dump_json())
        # The server hanging up cleanly is the stop signal.
        with pytest.raises(ConnectionClosedOK):
            connection.recv(timeout=5)

    assert set(coordinator.completed) == {leaf.object_id}
    assert coordinator.done.is_set()


def test_worker_disconnect_requeues_leased_job() -> None:
    leaf = ExecutionCoordinatorLeaf(value=1)
    coordinator = _new_execution_coordinator([leaf])

    with execution_coordinator_server(
        coordinator, bind_host="127.0.0.1", port=0
    ) as server:
        connection = _connect_worker(server, worker="doomed-worker")
        Job.model_validate_json(connection.recv(timeout=5))
        connection.close()

        # The dropped connection releases the lease; nothing is lost or failed.
        _wait_until(lambda: leaf.object_id in coordinator.ready)
        assert coordinator.running == {}
        assert coordinator.failed == {}

        with _connect_worker(server, worker="replacement-worker") as replacement:
            reassign = Job.model_validate_json(replacement.recv(timeout=5))
            (artifact,) = reassign.artifacts
            assert artifact.object_id == leaf.object_id
            replacement.send(JobCompletedResult().model_dump_json())
            with pytest.raises(ConnectionClosedOK):
                replacement.recv(timeout=5)

    assert set(coordinator.completed) == {leaf.object_id}


def test_execution_coordinator_server_closes_active_workers() -> None:
    coordinator = _new_execution_coordinator([ExecutionCoordinatorLeaf(value=1)])

    with execution_coordinator_server(
        coordinator, bind_host="127.0.0.1", port=0
    ) as server:
        connection = _connect_worker(server)
        Job.model_validate_json(connection.recv(timeout=5))

    with pytest.raises(ConnectionClosed):
        connection.recv(timeout=5)


def test_execution_coordinator_server_shutdown_wakes_idle_worker_handlers() -> None:
    coordinator = _new_execution_coordinator([GpuLeaf(value=1)])

    started = time.monotonic()
    with execution_coordinator_server(
        coordinator, bind_host="127.0.0.1", port=0
    ) as server:
        # No leasable job for a CPU-only worker, so its handler waits inside
        # lease_job without touching the socket.
        connection = _connect_worker(server, resources=ResourceRequest(gpus=0))

    assert time.monotonic() - started < 5
    assert coordinator.finish_error is not None
    with pytest.raises(ConnectionClosed):
        connection.recv(timeout=5)


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


def test_job_result_requires_error_for_failed_status() -> None:
    with pytest.raises(ValidationError, match="Field required"):
        JobFailedResult.model_validate({"status": "failed"})


def test_job_result_uses_status_discriminator() -> None:
    adapter = TypeAdapter(JobResult)

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
    with pytest.raises(OSError):
        worker_loop(
            server_url="ws://127.0.0.1:1",
            auth_token="test-token",
            resource_request=ResourceRequest(),
            idle_timeout=get_config().worker.idle_timeout_seconds,
            component="test-worker",
            backend="test",
        )


def test_worker_loop_exits_after_idle_timeout() -> None:
    with _scripted_worker_server([], hold_open=True) as server:
        worker_loop(
            server_url=server.server_url,
            auth_token="test-token",
            resource_request=ResourceRequest(),
            idle_timeout=0.05,
            component="test-worker",
            backend="test",
        )

        assert len(server.hellos) == 1
        assert server.results == []


def test_worker_loop_logs_received_task_and_result(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    leaf = ExecutionCoordinatorLeaf(value=1)
    other_leaf = ExecutionCoordinatorLeaf(value=2)
    job = Job(
        artifacts=[ArtifactSpec.from_furu(leaf), ArtifactSpec.from_furu(other_leaf)],
        provenance=_submit_provenance(),
    )
    log_path = tmp_path / "worker.log"

    monkeypatch.setattr(
        worker_loop_module,
        "execute_job",
        lambda obj, *, job: JobCompletedResult(),
    )

    with (
        _scripted_worker_server([job]) as server,
        _captured_furu_logs(caplog),
        _scoped_log_files((log_path,)),
    ):
        worker_loop(
            server_url=server.server_url,
            auth_token="test-token",
            resource_request=ResourceRequest(),
            idle_timeout=get_config().worker.idle_timeout_seconds,
            component="test-worker",
            backend="test",
        )

        assert server.results == [JobCompletedResult()]

    assert f"received {leaf._log_label} ×2" in caplog.messages
    assert any(
        message.startswith(f"finished {leaf._log_label} ×2 ok ·")
        for message in caplog.messages
    )
    assert "server closed the connection; worker exiting" in caplog.messages
    received_line = next(
        line for line in log_path.read_text().splitlines() if 'msg="received ' in line
    )
    assert "artifacts=2" in received_line


def test_worker_loop_exits_after_exceeding_max_consecutive_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaf = ExecutionCoordinatorLeaf(value=1)
    jobs = [_job(leaf) for _ in range(4)]

    monkeypatch.setattr(
        worker_loop_module,
        "execute_job",
        lambda obj, *, job: JobFailedResult(error="worker task failed"),
    )

    with _scripted_worker_server(jobs) as server:
        worker_loop(
            server_url=server.server_url,
            auth_token="test-token",
            resource_request=ResourceRequest(),
            idle_timeout=get_config().worker.idle_timeout_seconds,
            max_consecutive_failures=2,
            component="test-worker",
            backend="test",
        )

        # The worker gives up after its third consecutive failure and never
        # reports a result for the fourth assignment.
        assert len(server.results) == 3
        assert all(isinstance(result, JobFailedResult) for result in server.results)


def test_worker_loop_resets_consecutive_failures_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaf = ExecutionCoordinatorLeaf(value=1)
    jobs = [_job(leaf) for _ in range(3)]

    calls = 0

    def execute_job(obj: Spec[object], *, job: Job) -> JobResult:
        nonlocal calls
        calls += 1
        if calls in (1, 3):
            return JobFailedResult(error="worker task failed")
        return JobCompletedResult()

    monkeypatch.setattr(worker_loop_module, "execute_job", execute_job)

    with _scripted_worker_server(jobs) as server:
        worker_loop(
            server_url=server.server_url,
            auth_token="test-token",
            resource_request=ResourceRequest(),
            idle_timeout=get_config().worker.idle_timeout_seconds,
            max_consecutive_failures=2,
            component="test-worker",
            backend="test",
        )

        assert [result.status for result in server.results] == [
            "failed",
            "completed",
            "failed",
        ]


def test_worker_loop_does_not_swallow_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaf = ExecutionCoordinatorLeaf(value=1)

    def execute_job(obj: Spec[object], *, job: Job) -> JobResult:
        raise KeyboardInterrupt

    monkeypatch.setattr(worker_loop_module, "execute_job", execute_job)

    with _scripted_worker_server([_job(leaf)]) as server:
        with pytest.raises(KeyboardInterrupt):
            worker_loop(
                server_url=server.server_url,
                auth_token="test-token",
                resource_request=ResourceRequest(gpus=1),
                idle_timeout=get_config().worker.idle_timeout_seconds,
                component="test-worker",
                backend="test",
            )

        assert server.results == []
        (hello,) = server.hellos
        assert hello.resources == ResourceRequest(gpus=1)
        assert hello.worker == "test-worker"
        assert hello.backend == "test"


def test_execution_coordinator_fail_sets_finish_error_and_done() -> None:
    coordinator = _new_execution_coordinator([ExecutionCoordinatorLeaf(value=1)])

    coordinator.fail("pool broke")

    assert coordinator.done.is_set()
    with pytest.raises(RuntimeError, match="pool broke"):
        coordinator.raise_for_failure()
