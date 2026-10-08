import os
import threading
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest import mock
from uuid import uuid4

import pytest
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK, InvalidStatus
from websockets.headers import build_authorization_basic
from websockets.sync.client import ClientConnection, connect
from websockets.sync.server import ServerConnection, serve

import furu
import furu.worker.loop as worker_loop_module
from furu import Spec, Throttle, Worker
from furu.config import _Config, _dump_worker_json_config, get_config
from furu.dag import _add_to_dag
from furu.execution.execution_coordinator import (
    ExecutionCoordinator,
    _resolve_takeover,
)
from furu.execution.server import execution_coordinator_server, request_takeover
from furu.locking import lock
from furu.metadata import ArtifactSpec
from furu.provenance import (
    EnvironmentIdentity,
    GitIdentity,
    SubmitContext,
    SubmitProvenance,
)
from furu.storage._layout import (
    compute_lock_path_in,
)
from furu.worker.backends.local import LocalThreadWorkerBackend, LocalThreadWorkerPool
from furu.worker.execute import ChildSlot
from furu.worker.loop import worker_loop
from furu.worker.protocol import (
    CancelMessage,
    HelloMessage,
    Job,
    JobBlockedResult,
    JobCompletedResult,
    JobFailedResult,
    JobResult,
    PoolHandoff,
    ProcessSettings,
    coordinator_url,
    job_result_adapter,
    server_message_adapter,
)

CPU_POOL = LocalThreadWorkerBackend(worker=Worker())
GPU_POOL = LocalThreadWorkerBackend(worker=Worker(gpus=1, memory_gib=16))


def _pool(worker: Worker) -> LocalThreadWorkerBackend:
    return LocalThreadWorkerBackend(worker=worker)


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


def _job(obj: Spec) -> Job:
    return Job(
        artifacts=[ArtifactSpec.from_furu(obj)],
        provenance=_submit_provenance(),
        process=ProcessSettings.from_metadata(obj._metadata),
    )


def _artifact(job: Job | None) -> ArtifactSpec:
    assert isinstance(job, Job)
    (artifact,) = job.artifacts
    return artifact


def _new_execution_coordinator(
    objs: Sequence[furu.Spec],
    *,
    max_retries_per_object: int | None = None,
) -> ExecutionCoordinator:
    if max_retries_per_object is None:
        max_retries_per_object = get_config().worker.max_retries_per_object
    coordinator = ExecutionCoordinator(
        max_retries_per_object=max_retries_per_object,
        backends={"cpu": CPU_POOL, "gpu": GPU_POOL},
        submit_provenance=_submit_provenance(),
    )
    _add_to_dag(coordinator, objs)
    return coordinator


def _lease_job(
    coordinator: ExecutionCoordinator,
    *,
    backend: LocalThreadWorkerBackend = CPU_POOL,
) -> Job | None:
    return coordinator.lease_job(backend=backend, worker=f"test-worker-{uuid4()}")


def _no_satisfiable_job(
    coordinator: ExecutionCoordinator,
    *,
    backend: LocalThreadWorkerBackend = CPU_POOL,
) -> bool:
    return coordinator.count_satisfiable_jobs(backend=backend, max_workers=1) == 0


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
) -> Generator[_ScriptedServer]:
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


@contextmanager
def _serve(handler: Callable[[ServerConnection], None]) -> Generator[str]:
    server = serve(handler, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _connect_worker(
    server: Any,
    *,
    auth_token: str | None = None,
    worker: str = "raw-test-worker",
    pool: str = "cpu",
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
            pool=pool,
        ).model_dump_json()
    )
    return connection


@contextmanager
def _mark_running(obj: Spec) -> Generator[None]:
    obj._base_dir.mkdir(parents=True, exist_ok=True)
    with lock([compute_lock_path_in(obj._base_dir)]):
        yield


@contextmanager
def _taking_over(prefix: str) -> Generator[None]:
    with mock.patch.dict(os.environ, {"FURU_TAKEOVER": prefix}):
        yield


def _pool_worker_file(executor_dir: Path, pool: str) -> Path:
    return executor_dir / "workers" / pool / "worker.config.json"


def _write_worker_config(
    path: Path, *, url: str, config: _Config | None = None
) -> None:
    path.write_text(
        _dump_worker_json_config(
            get_config() if config is None else config,
            coordinator_url=url,
        ),
        encoding="utf-8",
    )


def _wait_until(condition: Callable[[], bool], *, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not met in time")
        time.sleep(0.01)


def _complete_one_job_over_ws(server_url: str, auth_token: str, pool: str) -> None:
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
                pool=pool,
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


class GatedExecutionCoordinatorLeaf(Spec[int]):
    value: int
    release_file: str

    def create(self) -> int:
        deadline = time.monotonic() + 30
        while not Path(self.release_file).exists():
            if time.monotonic() > deadline:
                raise TimeoutError("never released")
            time.sleep(0.01)
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

    def batch_key(self, worker: furu.Worker) -> tuple[str, int]:
        return (self.group, self.cap)

    @furu.batched(batch_key)
    def create(objs: list["BatchedCoordinatorLeaf"]) -> list[int]:
        return [obj.value for obj in objs]


class BatchSizeCoordinatorLeaf(furu.Spec[int]):
    value: int

    @furu.batched(lambda _, __: (None, 10))
    def create(objs: list["BatchSizeCoordinatorLeaf"]) -> list[int]:
        return [len(objs)] * len(objs)


class OptionalLoadLeaf(furu.Spec[str]):
    name: str

    def create(self) -> str:
        return self.name


class OptionalLoadParent(furu.Spec[str]):
    name: str

    def create(self) -> str:
        try:
            return OptionalLoadLeaf(name=self.name).load_existing()
        except furu.Missing:
            return "missing"


class GpuBatchedLeaf(furu.Spec[int]):
    value: int

    def runs_on(self, worker: Worker) -> bool:
        return worker.gpus in (1, 8)

    @furu.batched(lambda _, worker: (None, worker.gpus))
    def create(objs: list["GpuBatchedLeaf"]) -> list[int]:
        return [obj.value for obj in objs]


class PerGpuBatchedLeaf(furu.Spec[int]):
    value: int

    @furu.batched(lambda _, worker: (None, worker.gpus))
    def create(objs: list["PerGpuBatchedLeaf"]) -> list[int]:
        return [obj.value for obj in objs]


class ThrottledBatchedCoordinatorLeaf(furu.Spec[int]):
    value: int
    throttle = Throttle(max_running=2)

    def batch_key(self, worker: furu.Worker) -> tuple[None, int]:
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


def test_execution_coordinator_job_result_completed_moves_dependents_to_ready() -> None:
    leaf = ExecutionCoordinatorLeaf(value=1)
    parent = ExecutionCoordinatorParent(child=leaf)
    coordinator = _new_execution_coordinator([parent])
    assert set(coordinator.blocked) == {parent.object_id}

    job = coordinator.lease_job(backend=CPU_POOL, worker="test-worker")
    assert _artifact(job).object_id == leaf.object_id

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

    parent_job = coordinator.lease_job(backend=CPU_POOL, worker="test-worker")
    assert isinstance(parent_job, Job)

    coordinator.job_result(
        parent.object_id,
        JobBlockedResult(dependencies=[ArtifactSpec.from_furu(dependency)]),
    )

    assert set(coordinator.ready) == {dependency.object_id}
    assert set(coordinator.blocked) == {parent.object_id}

    dependency_job = coordinator.lease_job(backend=CPU_POOL, worker="test-worker")
    assert isinstance(dependency_job, Job)
    coordinator.job_result(dependency.object_id, JobCompletedResult())

    assert set(coordinator.ready) == {parent.object_id}
    assert coordinator.blocked == {}
    assert _artifact(_lease_job(coordinator)).object_id == parent.object_id


def test_execution_coordinator_job_result_blocked_ignores_completed_lazy_dependency() -> (
    None
):
    parent = ExecutionCoordinatorLazyParent(value=2)
    dependency = ExecutionCoordinatorLeaf(value=2)
    dependency.create()
    coordinator = _new_execution_coordinator([parent])

    parent_job = coordinator.lease_job(backend=CPU_POOL, worker="test-worker")
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

    parent_job = coordinator.lease_job(backend=CPU_POOL, worker="test-worker")
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

    # The parent waits for every discovered dependency, not just the first.
    for _ in dependencies:
        assert set(coordinator.blocked) == {parent.object_id}
        leased = _artifact(_lease_job(coordinator)).object_id
        coordinator.job_result(leased, JobCompletedResult())
    assert set(coordinator.ready) == {parent.object_id}


def test_execution_coordinator_job_result_failed_retries_before_finishing() -> None:
    leaf = ExecutionCoordinatorLeaf(value=1)
    coordinator = _new_execution_coordinator([leaf], max_retries_per_object=2)

    for attempt in (1, 2, 3):
        assert not coordinator.done.is_set()
        assert isinstance(_lease_job(coordinator), Job)
        coordinator.job_result(leaf.object_id, JobFailedResult(error=f"boom {attempt}"))

        failed_job = coordinator.failed[leaf.object_id]
        assert failed_job.failed_attempts == attempt
        assert failed_job.error == f"boom {attempt}"
        # Retries go back to ready; the last failure is final.
        assert set(coordinator.ready) == ({leaf.object_id} if attempt < 3 else set())

    assert coordinator.running == {}
    assert coordinator.done.is_set()
    with pytest.raises(RuntimeError, match="failed jobs"):
        coordinator.raise_for_failure()


def test_execution_coordinator_job_result_failed_retry_can_later_complete() -> None:
    leaf = ExecutionCoordinatorLeaf(value=1)
    coordinator = _new_execution_coordinator([leaf], max_retries_per_object=1)

    first_job = coordinator.lease_job(backend=CPU_POOL, worker="test-worker")
    assert isinstance(first_job, Job)
    coordinator.job_result(leaf.object_id, JobFailedResult(error="boom"))

    failed_job = coordinator.failed[leaf.object_id]
    assert failed_job.failed_attempts == 1

    retry_job = coordinator.lease_job(backend=CPU_POOL, worker="test-worker")
    assert isinstance(retry_job, Job)
    coordinator.job_result(leaf.object_id, JobCompletedResult())

    assert coordinator.failed == {}
    assert set(coordinator.completed) == {leaf.object_id}
    assert coordinator.done.is_set()


class GpuLeaf(Spec[int]):
    value: int

    def runs_on(self, worker: Worker) -> bool:
        return worker.gpus >= 1

    def create(self) -> int:
        return self.value


class CpuOnlyLeaf(Spec[int]):
    value: int

    def create(self) -> int:
        return self.value


class MemoryLeaf(Spec[int]):
    value: int

    def runs_on(self, worker: Worker) -> bool:
        return worker.memory_gib >= 8

    def create(self) -> int:
        return self.value


class DynamicCpuSeed(Spec[int]):
    value: int

    def create(self) -> int:
        return self.value


class DynamicGpuAfterSeed(Spec[int]):
    parent: DynamicCpuSeed
    value: int

    def runs_on(self, worker: Worker) -> bool:
        return worker.gpus >= 1

    def create(self) -> int:
        return self.parent.create() + self.value


class DynamicCpuAfterGpu(Spec[int]):
    parent: DynamicGpuAfterSeed
    value: int

    def create(self) -> int:
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

    assert coordinator.count_satisfiable_jobs(backend=CPU_POOL, max_workers=10) == 2
    assert coordinator.count_satisfiable_jobs(backend=CPU_POOL, max_workers=1) == 1
    assert coordinator.count_satisfiable_jobs(backend=CPU_POOL, max_workers=0) == 0
    with pytest.raises(ValueError):
        coordinator.count_satisfiable_jobs(backend=CPU_POOL, max_workers=-1)
    assert (
        coordinator.count_satisfiable_jobs(
            backend=_pool(Worker(gpus=1)), max_workers=10
        )
        == 3
    )


def test_count_satisfiable_jobs_returns_zero_when_coordinator_is_done() -> None:
    coordinator = _new_execution_coordinator([ExecutionCoordinatorLeaf(value=1)])
    assert coordinator.count_satisfiable_jobs(backend=CPU_POOL, max_workers=10) == 1

    coordinator.fail("execution interrupted")

    assert coordinator.count_satisfiable_jobs(backend=CPU_POOL, max_workers=10) == 0


@pytest.mark.parametrize(
    ("leaf_factory", "initial_demand"),
    [
        (lambda value: ExecutionCoordinatorLeaf(value=value), 2),
        (lambda value: BatchedCoordinatorLeaf(value=value), 1),
    ],
)
def test_only_discovered_external_computations_are_polled(
    leaf_factory: Callable[[int], Spec],
    initial_demand: int,
) -> None:
    leaves = [leaf_factory(value) for value in range(2)]
    coordinator = _new_execution_coordinator(leaves)
    leased: list[Job | None] = []

    with mock.patch(
        "furu.execution.execution_coordinator._RUNNING_ELSEWHERE_POLL_INTERVAL_S",
        0.01,
    ):
        with _mark_running(leaves[0]), _mark_running(leaves[1]):
            assert (
                coordinator.count_satisfiable_jobs(backend=CPU_POOL, max_workers=10)
                == initial_demand
            )
            thread = threading.Thread(
                target=lambda: leased.append(_lease_job(coordinator))
            )
            thread.start()
            _wait_until(
                lambda: (
                    set(coordinator.running_elsewhere)
                    == {leaf.object_id for leaf in leaves}
                )
            )
            assert (
                coordinator.count_satisfiable_jobs(backend=CPU_POOL, max_workers=10)
                == 0
            )

        thread.join(timeout=10)
        assert not thread.is_alive()

    assert isinstance(leased[0], Job)


def test_worker_cap_limits_satisfiable_jobs_and_leases() -> None:
    limited = [LimitedExecutionCoordinatorLeaf(value=value) for value in range(3)]
    uncapped = ExecutionCoordinatorLeaf(value=10)
    coordinator = _new_execution_coordinator([*limited, uncapped])

    assert coordinator.count_satisfiable_jobs(backend=CPU_POOL, max_workers=10) == 3

    first = _lease_job(coordinator, backend=CPU_POOL)
    second = _lease_job(coordinator, backend=CPU_POOL)
    third = _lease_job(coordinator, backend=CPU_POOL)

    assert isinstance(first, Job)
    assert isinstance(second, Job)
    assert isinstance(third, Job)
    limited_ids = {obj.object_id for obj in limited}
    leased_limited_ids = {_artifact(first).object_id, _artifact(second).object_id}
    assert leased_limited_ids < limited_ids
    assert _artifact(third).object_id == uncapped.object_id
    assert coordinator.count_satisfiable_jobs(backend=CPU_POOL, max_workers=10) == 0
    assert _no_satisfiable_job(coordinator, backend=CPU_POOL)

    coordinator.job_result(_artifact(first).object_id, JobCompletedResult())
    fourth = _lease_job(coordinator, backend=CPU_POOL)

    assert isinstance(fourth, Job)
    assert _artifact(fourth).object_id in limited_ids - leased_limited_ids


def test_lease_job_assembles_same_key_batched_group_into_one_job() -> None:
    objs = [BatchedCoordinatorLeaf(value=value) for value in range(3)]
    coordinator = _new_execution_coordinator(objs)

    job = _lease_job(coordinator)

    assert isinstance(job, Job)
    assert len(job.artifacts) == 3
    assert {artifact.object_id for artifact in job.artifacts} == set(
        coordinator.running
    )
    assert coordinator.ready == {}

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


def test_batch_cap_follows_the_leasing_worker() -> None:
    objs = [GpuBatchedLeaf(value=value) for value in range(10)]
    coordinator = _new_execution_coordinator(objs)
    one_gpu, eight_gpus = _pool(Worker(gpus=1)), _pool(Worker(gpus=8))

    assert coordinator.count_satisfiable_jobs(backend=one_gpu, max_workers=20) == 10
    assert coordinator.count_satisfiable_jobs(backend=eight_gpus, max_workers=20) == 2
    assert _no_satisfiable_job(coordinator, backend=_pool(Worker(gpus=4)))

    big = _lease_job(coordinator, backend=eight_gpus)
    small = _lease_job(coordinator, backend=one_gpu)
    assert isinstance(big, Job) and len(big.artifacts) == 8
    assert isinstance(small, Job) and len(small.artifacts) == 1


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

    assert coordinator.count_satisfiable_jobs(backend=CPU_POOL, max_workers=10) == 2


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
    objs = [BatchSizeCoordinatorLeaf(value=value) for value in range(3)]

    assert furu.create(objs, on=[LocalThreadWorkerBackend()]) == [3, 3, 3]


def test_build_on_workers_leaves_results_on_disk() -> None:
    objs = [BatchSizeCoordinatorLeaf(value=value) for value in range(3)]

    furu.build(objs, on=[LocalThreadWorkerBackend()])

    assert [obj.status for obj in objs] == ["done", "done", "done"]


def test_load_existing_in_worker_does_not_build_missing_spec() -> None:
    parent = OptionalLoadParent(name="optional")

    assert furu.create(parent, on=[LocalThreadWorkerBackend()]) == "missing"
    assert OptionalLoadLeaf(name="optional").status == "missing"


def test_worker_lost_requeues_running_lease_without_counting_failure() -> None:
    objs = [LimitedExecutionCoordinatorLeaf(value=value) for value in range(3)]
    coordinator = _new_execution_coordinator(objs)

    first = coordinator.lease_job(backend=CPU_POOL, worker="worker-1")
    second = coordinator.lease_job(backend=CPU_POOL, worker="worker-2")

    assert isinstance(first, Job)
    assert isinstance(second, Job)
    assert coordinator.count_satisfiable_jobs(backend=CPU_POOL, max_workers=10) == 0

    coordinator.worker_lost("worker-1")

    assert set(coordinator.running) == {_artifact(second).object_id}
    assert _artifact(first).object_id in coordinator.ready
    assert coordinator.failed == {}
    assert coordinator.count_satisfiable_jobs(backend=CPU_POOL, max_workers=10) == 1


def test_job_result_after_worker_lost_is_ignored() -> None:
    leaf = ExecutionCoordinatorLeaf(value=1)
    coordinator = _new_execution_coordinator([leaf])
    job = coordinator.lease_job(backend=CPU_POOL, worker="worker-1")
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

    cpu_job = _lease_job(coordinator, backend=CPU_POOL)
    assert isinstance(cpu_job, Job)
    assert _artifact(cpu_job).object_id == cpu_leaf.object_id

    assert _no_satisfiable_job(coordinator, backend=CPU_POOL)

    gpu_job = _lease_job(coordinator, backend=_pool(Worker(gpus=1)))
    assert isinstance(gpu_job, Job)
    assert _artifact(gpu_job).object_id == gpu_leaf.object_id


def test_lease_job_honors_pool_accepts() -> None:
    cpu_leaf = CpuOnlyLeaf(value=1)
    memory_leaf = MemoryLeaf(value=2)
    coordinator = _new_execution_coordinator([cpu_leaf, memory_leaf])
    reserved = LocalThreadWorkerBackend(
        worker=Worker(memory_gib=16),
        accepts=lambda spec: isinstance(spec, MemoryLeaf),
    )

    memory_job = _lease_job(coordinator, backend=reserved)
    assert isinstance(memory_job, Job)
    assert _artifact(memory_job).object_id == memory_leaf.object_id
    assert _no_satisfiable_job(coordinator, backend=reserved)

    cpu_job = _lease_job(coordinator, backend=_pool(Worker(memory_gib=16)))
    assert isinstance(cpu_job, Job)
    assert _artifact(cpu_job).object_id == cpu_leaf.object_id


def test_execution_coordinator_run_fails_fast_when_no_pool_can_run_a_job() -> None:
    with pytest.raises(RuntimeError, match=r"no worker pool can run.*workers: Worker"):
        ExecutionCoordinator.run(
            [MemoryLeaf(value=uuid4().int)],
            worker_backends=(LocalThreadWorkerBackend(worker=Worker()),),
        )


def test_execution_coordinator_fails_when_no_pool_can_run_a_lazy_dependency() -> None:
    parent = ExecutionCoordinatorLazyParent(value=uuid4().int)
    coordinator = _new_execution_coordinator([parent])
    coordinator.backends = {"cpu": CPU_POOL}
    assert isinstance(_lease_job(coordinator), Job)

    coordinator.job_result(
        parent.object_id,
        JobBlockedResult(
            dependencies=[ArtifactSpec.from_furu(MemoryLeaf(value=uuid4().int))]
        ),
    )

    assert coordinator.done.is_set()
    assert coordinator.finish_error is not None
    assert "no worker pool can run" in coordinator.finish_error


def test_execution_coordinator_run_rejects_a_zero_batch_cap_before_starting_pools() -> (
    None
):
    with pytest.raises(TypeError, match=r"cap must be a positive int, got 0 on Worker"):
        ExecutionCoordinator.run(
            [PerGpuBatchedLeaf(value=uuid4().int)],
            worker_backends=(
                LocalThreadWorkerBackend(worker=Worker(gpus=1)),
                LocalThreadWorkerBackend(worker=Worker()),
            ),
        )


def test_execution_coordinator_fails_on_a_zero_batch_cap_in_a_lazy_dependency() -> None:
    parent = ExecutionCoordinatorLazyParent(value=uuid4().int)
    coordinator = _new_execution_coordinator([parent])
    job = _lease_job(coordinator)
    assert isinstance(job, Job)

    coordinator.job_result(
        parent.object_id,
        JobBlockedResult(
            dependencies=[ArtifactSpec.from_furu(PerGpuBatchedLeaf(value=1))]
        ),
    )

    assert coordinator.done.is_set()
    assert coordinator.finish_error is not None
    assert "cap must be a positive int, got 0" in coordinator.finish_error


def test_execution_coordinator_run_completes_later_resource_stages_on_local_workers() -> (
    None
):
    seed_value = uuid4().int
    seed = DynamicCpuSeed(value=seed_value)
    first_gpu = DynamicGpuAfterSeed(parent=seed, value=20)
    second_cpu = DynamicCpuAfterGpu(parent=first_gpu, value=30)

    ExecutionCoordinator.run(
        [second_cpu],
        worker_backends=(
            LocalThreadWorkerBackend(worker=Worker(gpus=0)),
            LocalThreadWorkerBackend(worker=Worker(gpus=1)),
        ),
    )

    for obj in (seed, first_gpu, second_cpu):
        assert obj.status == "done"
    assert second_cpu.create() == seed_value + 50


def test_execution_coordinator_run_fails_when_local_worker_crashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def crashing_worker_loop(
        *,
        coordinator: str | Path,
        pool: str,
        idle_timeout: float | None,
        max_failures: int,
        component: str,
        backend: str,
        materialize_snapshot: bool,
        log_file: Path,
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


def test_execution_coordinator_run_drives_worker_backend_pool() -> None:
    class RecordingPool:
        def __init__(self) -> None:
            self.stop_timeouts: list[float] = []
            self.worker_thread: threading.Thread | None = None

        def handoff(self) -> PoolHandoff:
            return PoolHandoff()

        def stop(self, *, timeout: float) -> None:
            self.stop_timeouts.append(timeout)
            assert self.worker_thread is not None
            self.worker_thread.join(timeout=timeout)

    class RecordingBackend:
        execution_coordinator_listen_host = "127.0.0.1"
        worker = Worker()
        accepts = None
        pool_key = "test-pool"

        def __init__(self) -> None:
            self.pool = RecordingPool()
            self.starts: list[tuple[ExecutionCoordinator, int, str, Path]] = []

        def start_pool(
            self,
            *,
            coordinator: ExecutionCoordinator,
            bound_port: int,
            auth_token: str,
            executor_dir: Path,
            handoff: PoolHandoff,
        ) -> RecordingPool:
            self.starts.append((coordinator, bound_port, auth_token, executor_dir))
            # The server listens on the backend's host.
            server_url = f"ws://{self.execution_coordinator_listen_host}:{bound_port}"

            def complete_job() -> None:
                try:
                    _complete_one_job_over_ws(server_url, auth_token, self.pool_key)
                except BaseException as exc:
                    coordinator.fail(f"recording backend failed: {exc!r}")

            self.pool.worker_thread = threading.Thread(target=complete_job)
            self.pool.worker_thread.start()
            return self.pool

    leaf = ExecutionCoordinatorLeaf(value=11)
    objs = [leaf]
    backend = RecordingBackend()

    assert ExecutionCoordinator.run(objs, worker_backends=(backend,)) is objs

    ((coordinator, bound_port, auth_token, executor_dir),) = backend.starts
    assert bound_port > 0
    assert auth_token
    assert isinstance(coordinator.submit_provenance, SubmitProvenance)
    assert coordinator.pools == {backend.pool_key: backend.pool}
    assert backend.pool.stop_timeouts == [5]
    assert executor_dir == coordinator.executor_dir
    assert executor_dir.parent == get_config().run_directories.executions
    assert set(coordinator.completed) == {leaf.object_id}


def test_execution_coordinator_run_returns_when_all_objects_are_already_completed() -> (
    None
):
    class UnexpectedBackend:
        execution_coordinator_listen_host = "127.0.0.1"
        worker = Worker()
        accepts = None
        pool_key = "test-pool"

        def start_pool(
            self,
            *,
            coordinator: ExecutionCoordinator,
            bound_port: int,
            auth_token: str,
            executor_dir: Path,
            handoff: PoolHandoff,
        ) -> LocalThreadWorkerPool:
            raise AssertionError("coordinator started workers with no runnable objects")

    leaf = ExecutionCoordinatorLeaf(value=15)
    leaf.create()
    objs = [leaf]
    coordinator = _new_execution_coordinator(objs)
    executions_dir = get_config().run_directories.executions

    assert coordinator.nodes_by_id == {}

    returned = ExecutionCoordinator.run(objs, worker_backends=(UnexpectedBackend(),))

    assert returned is objs
    # A no-op run returns before creating an executor dir or capturing
    # provenance; nothing appears under executions/.
    assert not executions_dir.exists() or list(executions_dir.iterdir()) == []


def test_execution_coordinator_run_stops_backend_pool_when_interrupted() -> None:
    class InterruptingEvent(threading.Event):
        def wait(self, timeout: float | None = None) -> bool:
            raise KeyboardInterrupt

    class InterruptingCoordinator(ExecutionCoordinator):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.done = InterruptingEvent()

    class RecordingPool:
        def __init__(self) -> None:
            self.events: list[str] = []
            self.stop_timeouts: list[float] = []

        def handoff(self) -> PoolHandoff:
            return PoolHandoff()

        def stop(self, *, timeout: float) -> None:
            self.events.append("stop")
            self.stop_timeouts.append(timeout)

    class RecordingBackend:
        execution_coordinator_listen_host = "127.0.0.1"
        worker = Worker()
        accepts = None
        pool_key = "test-pool"

        def __init__(self, pool: RecordingPool) -> None:
            self.pool = pool

        def start_pool(
            self,
            *,
            coordinator: ExecutionCoordinator,
            bound_port: int,
            auth_token: str,
            executor_dir: Path,
            handoff: PoolHandoff,
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


def test_execution_coordinator_server_accepts_token_in_url() -> None:
    coordinator = _new_execution_coordinator([ExecutionCoordinatorLeaf(value=12)])

    with execution_coordinator_server(
        coordinator, bind_host="127.0.0.1", port=0
    ) as server:
        with connect(
            coordinator_url(
                host="127.0.0.1", port=server.bound_port, auth_token=server.auth_token
            )
        ) as connection:
            connection.send(
                HelloMessage(
                    worker="url-auth-worker",
                    backend="test",
                    pool="cpu",
                ).model_dump_json()
            )
            Job.model_validate_json(connection.recv(timeout=5))

        with pytest.raises(InvalidStatus) as wrong_token:
            connect(
                coordinator_url(
                    host="127.0.0.1", port=server.bound_port, auth_token="wrong"
                )
            )
        assert wrong_token.value.response.status_code == 401


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
        connection = _connect_worker(server, pool="cpu")

    assert time.monotonic() - started < 5
    assert coordinator.finish_error is not None
    with pytest.raises(ConnectionClosed):
        connection.recv(timeout=5)


def test_local_pool_key_distinguishes_resources_not_worker_count() -> None:
    backend = LocalThreadWorkerBackend(max_workers=2)

    assert backend.pool_key.startswith("local:")
    assert LocalThreadWorkerBackend(max_workers=2).pool_key == backend.pool_key
    assert LocalThreadWorkerBackend(max_workers=3).pool_key == backend.pool_key
    assert (
        LocalThreadWorkerBackend(max_workers=2, worker=Worker(gpus=1)).pool_key
        != backend.pool_key
    )


def test_execution_coordinator_run_rejects_identical_worker_backends() -> None:
    with pytest.raises(ValueError, match="identical configuration"):
        ExecutionCoordinator.run(
            [ExecutionCoordinatorLeaf(value=12)],
            worker_backends=(LocalThreadWorkerBackend(), LocalThreadWorkerBackend()),
        )


def test_execution_coordinator_run_rejects_conflicting_execution_coordinator_listen_host() -> (
    None
):
    with pytest.raises(ValueError):
        ExecutionCoordinator.run(
            [ExecutionCoordinatorLeaf(value=12)],
            worker_backends=(
                LocalThreadWorkerBackend(
                    worker=Worker(), execution_coordinator_listen_host="127.0.0.1"
                ),
                LocalThreadWorkerBackend(
                    worker=Worker(gpus=1), execution_coordinator_listen_host="0.0.0.0"
                ),
            ),
        )


def test_worker_loop_raises_when_server_is_unavailable(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        worker_loop(
            coordinator="ws://127.0.0.1:1",
            pool="cpu",
            idle_timeout=get_config().worker.idle_timeout_seconds,
            max_failures=get_config().worker.max_failures_per_worker,
            component="test-worker",
            backend="test",
            materialize_snapshot=False,
            log_file=tmp_path / "worker.log",
        )


def test_worker_loop_exits_after_idle_timeout(tmp_path: Path) -> None:
    with _scripted_worker_server([], hold_open=True) as server:
        worker_loop(
            coordinator=server.server_url,
            pool="cpu",
            idle_timeout=0.05,
            max_failures=get_config().worker.max_failures_per_worker,
            component="test-worker",
            backend="test",
            materialize_snapshot=False,
            log_file=tmp_path / "worker.log",
        )

        assert len(server.hellos) == 1
        assert server.results == []


def test_worker_loop_exits_non_zero_after_consecutive_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    failed = JobFailedResult(error="boom")
    # A success in between resets the count.
    outcomes = iter([failed, JobCompletedResult(), failed, failed, failed])
    monkeypatch.setattr(ChildSlot, "run", lambda *_, **__: next(outcomes))
    jobs = [_job(ExecutionCoordinatorLeaf(value=value)) for value in range(5)]

    with (
        _scripted_worker_server(jobs, hold_open=True) as server,
        pytest.raises(SystemExit) as exc_info,
    ):
        worker_loop(
            coordinator=server.server_url,
            pool="cpu",
            idle_timeout=get_config().worker.idle_timeout_seconds,
            max_failures=2,
            component="test-worker",
            backend="test",
            materialize_snapshot=False,
            log_file=tmp_path / "worker.log",
        )

    assert "2 jobs failed in a row" in str(exc_info.value.code)
    # Every result reaches the coordinator before the worker gives up.
    assert [result.status for result in server.results] == [
        "failed",
        "completed",
        "failed",
        "failed",
    ]


def test_worker_loop_does_not_swallow_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    leaf = ExecutionCoordinatorLeaf(value=1)

    def run(self: ChildSlot, job: Job, *, cancelled: threading.Event) -> JobResult:
        raise KeyboardInterrupt

    monkeypatch.setattr(ChildSlot, "run", run)

    with _scripted_worker_server([_job(leaf)]) as server:
        with pytest.raises(KeyboardInterrupt):
            worker_loop(
                coordinator=server.server_url,
                pool="gpu",
                idle_timeout=get_config().worker.idle_timeout_seconds,
                max_failures=get_config().worker.max_failures_per_worker,
                component="test-worker",
                backend="test",
                materialize_snapshot=False,
                log_file=tmp_path / "worker.log",
            )

        assert server.results == []
        (hello,) = server.hellos
        assert hello.pool == "gpu"
        assert hello.worker == "test-worker"
        assert hello.backend == "test"


def test_hello_running_adopts_job_this_run_still_wants() -> None:
    leaf = ExecutionCoordinatorLeaf(value=1)
    coordinator = _new_execution_coordinator([leaf])

    with execution_coordinator_server(
        coordinator, bind_host="127.0.0.1", port=0
    ) as server:
        connection = connect(
            coordinator_url(
                host="127.0.0.1", port=server.bound_port, auth_token=server.auth_token
            )
        )
        with connection:
            connection.send(
                HelloMessage(
                    worker="inherited-worker",
                    backend="test",
                    pool="cpu",
                    running=[ArtifactSpec.from_furu(leaf)],
                ).model_dump_json()
            )
            _wait_until(lambda: leaf.object_id in coordinator.running)
            assert coordinator.running[leaf.object_id].worker == "inherited-worker"
            assert coordinator.ready == {}

            connection.send(JobCompletedResult().model_dump_json())
            with pytest.raises(ConnectionClosedOK):
                connection.recv(timeout=5)

    assert set(coordinator.completed) == {leaf.object_id}


def test_hello_running_cancels_job_not_in_this_run() -> None:
    leaf = ExecutionCoordinatorLeaf(value=1)
    stranger = ExecutionCoordinatorLeaf(value=2)
    coordinator = _new_execution_coordinator([leaf])

    with execution_coordinator_server(
        coordinator, bind_host="127.0.0.1", port=0
    ) as server:
        connection = connect(
            coordinator_url(
                host="127.0.0.1", port=server.bound_port, auth_token=server.auth_token
            )
        )
        with connection:
            connection.send(
                HelloMessage(
                    worker="inherited-worker",
                    backend="test",
                    pool="cpu",
                    running=[ArtifactSpec.from_furu(stranger)],
                ).model_dump_json()
            )
            assert (
                server_message_adapter.validate_json(connection.recv(timeout=5))
                == CancelMessage()
            )
            connection.send(
                JobFailedResult(error="subprocess died: signal 9").model_dump_json()
            )

            job = Job.model_validate_json(connection.recv(timeout=5))
            assert _artifact(job).object_id == leaf.object_id
            connection.send(JobCompletedResult().model_dump_json())
            with pytest.raises(ConnectionClosedOK):
                connection.recv(timeout=5)

    assert coordinator.failed == {}
    assert set(coordinator.completed) == {leaf.object_id}


def test_worker_loop_cancel_kills_running_job(tmp_path: Path) -> None:
    leaf = GatedExecutionCoordinatorLeaf(value=1, release_file=str(tmp_path / "go"))
    results: list[JobResult] = []

    def handler(connection: ServerConnection) -> None:
        HelloMessage.model_validate_json(connection.recv(timeout=5))
        connection.send(_job(leaf).model_dump_json())
        connection.send(CancelMessage().model_dump_json())
        results.append(job_result_adapter.validate_json(connection.recv(timeout=10)))

    with _serve(handler) as url:
        worker_loop(
            coordinator=url,
            pool="cpu",
            idle_timeout=5,
            max_failures=get_config().worker.max_failures_per_worker,
            component="test-worker",
            backend="test",
            materialize_snapshot=False,
            log_file=tmp_path / "worker.log",
        )

    (result,) = results
    assert isinstance(result, JobFailedResult)
    assert result.error.startswith("subprocess died: signal 9 (SIGKILL)")
    assert leaf.status != "done"


def test_worker_loop_reads_unchanged_worker_config_only_after_disconnect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = tmp_path / "worker.config.json"
    read_target = worker_loop_module._read_target
    reads = 0

    def count_control_reads(coordinator: str | Path) -> Any:
        nonlocal reads
        reads += 1
        if reads > 2:
            raise AssertionError("worker polled an unchanged config file")
        return read_target(coordinator)

    monkeypatch.setattr(worker_loop_module, "_read_target", count_control_reads)

    def handler(connection: ServerConnection) -> None:
        HelloMessage.model_validate_json(connection.recv(timeout=5))

    with _serve(handler) as url:
        _write_worker_config(config_file, url=url)
        worker_loop(
            coordinator=config_file,
            pool="cpu",
            idle_timeout=5,
            max_failures=get_config().worker.max_failures_per_worker,
            component="test-worker",
            backend="test",
            materialize_snapshot=False,
            log_file=tmp_path / "worker.log",
            disconnect_grace=0,
        )

    assert reads == 2


def test_worker_loop_fails_when_worker_config_disappears(tmp_path: Path) -> None:
    leaf = GatedExecutionCoordinatorLeaf(value=1, release_file=str(tmp_path / "go"))
    config_file = tmp_path / "worker.config.json"

    def handler(connection: ServerConnection) -> None:
        HelloMessage.model_validate_json(connection.recv(timeout=5))
        connection.send(_job(leaf).model_dump_json())
        config_file.unlink()

    with _serve(handler) as url, pytest.raises(OSError):
        _write_worker_config(config_file, url=url)
        worker_loop(
            coordinator=config_file,
            pool="cpu",
            idle_timeout=5,
            max_failures=get_config().worker.max_failures_per_worker,
            component="test-worker",
            backend="test",
            materialize_snapshot=False,
            log_file=tmp_path / "worker.log",
        )

    assert leaf.status != "done"


def test_worker_loop_carries_running_job_to_moved_coordinator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_file = tmp_path / "go"
    leaf = GatedExecutionCoordinatorLeaf(value=7, release_file=str(release_file))
    job = _job(leaf)
    config_file = tmp_path / "worker.config.json"
    old_hellos: list[HelloMessage] = []
    new_hellos: list[HelloMessage] = []
    results: list[JobResult] = []
    read_target = worker_loop_module._read_target
    read_truncated = threading.Event()

    def reading(coordinator: str | Path) -> Any:
        try:
            return read_target(coordinator)
        except ValueError:
            read_truncated.set()
            raise

    monkeypatch.setattr(worker_loop_module, "_read_target", reading)
    monkeypatch.setattr(worker_loop_module, "_WORKER_CONFIG_POLL_INTERVAL_S", 0.01)
    config = get_config()
    next_run_config = config.model_copy(
        update={
            "worker": config.worker.model_copy(update={"max_retries_per_object": 0})
        }
    )

    def new_handler(connection: ServerConnection) -> None:
        new_hellos.append(HelloMessage.model_validate_json(connection.recv(timeout=5)))
        release_file.touch()
        results.append(job_result_adapter.validate_json(connection.recv(timeout=10)))
        # A differently configured run ends the worker without a grace wait.
        _write_worker_config(config_file, url=new_url, config=next_run_config)

    def old_handler(connection: ServerConnection) -> None:
        old_hellos.append(HelloMessage.model_validate_json(connection.recv(timeout=5)))
        connection.send(job.model_dump_json())
        # Hang up mid-job, mid-rewrite of the worker config; finish the rewrite
        # only once the worker has seen the truncated file.
        config_file.write_text("")

        def finish_rewrite() -> None:
            assert read_truncated.wait(timeout=10)
            _write_worker_config(config_file, url=new_url)

        threading.Thread(target=finish_rewrite).start()

    with _serve(new_handler) as new_url, _serve(old_handler) as old_url:
        _write_worker_config(config_file, url=old_url)
        worker_loop(
            coordinator=config_file,
            pool="cpu",
            idle_timeout=5,
            max_failures=get_config().worker.max_failures_per_worker,
            component="test-worker",
            backend="test",
            materialize_snapshot=False,
            log_file=tmp_path / "worker.log",
            disconnect_grace=30,
        )

    assert [hello.running for hello in old_hellos] == [[]]
    assert [hello.running for hello in new_hellos] == [job.artifacts]
    assert results == [JobCompletedResult()]
    assert leaf.status == "done"


def test_worker_loop_exits_when_worker_config_changes(
    tmp_path: Path,
) -> None:
    release_file = tmp_path / "go"
    old_leaf = GatedExecutionCoordinatorLeaf(value=7, release_file=str(release_file))
    old_job = _job(old_leaf)
    config_file = tmp_path / "worker.config.json"
    old_config = get_config()
    new_config = old_config.model_copy(
        update={
            "directories": old_config.directories.model_copy(
                update={
                    "objects": tmp_path / "new-objects",
                    "snapshots": tmp_path / "new-snapshots",
                }
            )
        }
    )
    new_hellos: list[HelloMessage] = []

    def new_handler(connection: ServerConnection) -> None:
        new_hellos.append(HelloMessage.model_validate_json(connection.recv(timeout=5)))

    with _serve(new_handler) as new_url:

        def old_handler(connection: ServerConnection) -> None:
            HelloMessage.model_validate_json(connection.recv(timeout=5))
            connection.send(old_job.model_dump_json())
            _write_worker_config(
                config_file,
                url=new_url,
                config=new_config,
            )

        with _serve(old_handler) as old_url:
            _write_worker_config(
                config_file,
                url=old_url,
                config=old_config,
            )
            worker_loop(
                coordinator=config_file,
                pool="cpu",
                idle_timeout=5,
                max_failures=get_config().worker.max_failures_per_worker,
                component="test-worker",
                backend="test",
                materialize_snapshot=False,
                log_file=tmp_path / "worker.log",
            )

    assert new_hellos == []
    assert get_config() == old_config
    assert old_leaf.status != "done"


def test_worker_loop_kills_job_when_coordinator_disappears(tmp_path: Path) -> None:
    leaf = GatedExecutionCoordinatorLeaf(value=1, release_file=str(tmp_path / "go"))

    def handler(connection: ServerConnection) -> None:
        HelloMessage.model_validate_json(connection.recv(timeout=5))
        connection.send(_job(leaf).model_dump_json())

    started = time.monotonic()
    with _serve(handler) as url:
        worker_loop(
            coordinator=url,
            pool="cpu",
            idle_timeout=5,
            max_failures=get_config().worker.max_failures_per_worker,
            component="test-worker",
            backend="test",
            materialize_snapshot=False,
            log_file=tmp_path / "worker.log",
        )

    assert time.monotonic() - started < 10
    assert leaf.status != "done"


def test_worker_loop_kills_job_when_worker_config_never_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(worker_loop_module, "_WORKER_CONFIG_POLL_INTERVAL_S", 0.01)
    leaf = GatedExecutionCoordinatorLeaf(value=1, release_file=str(tmp_path / "go"))
    config_file = tmp_path / "worker.config.json"

    def handler(connection: ServerConnection) -> None:
        HelloMessage.model_validate_json(connection.recv(timeout=5))
        connection.send(_job(leaf).model_dump_json())

    with _serve(handler) as url:
        _write_worker_config(config_file, url=url)
        started = time.monotonic()
        worker_loop(
            coordinator=config_file,
            pool="cpu",
            idle_timeout=5,
            max_failures=get_config().worker.max_failures_per_worker,
            component="test-worker",
            backend="test",
            materialize_snapshot=False,
            log_file=tmp_path / "worker.log",
            disconnect_grace=0.2,
        )

    assert 0.2 <= time.monotonic() - started < 10
    assert leaf.status != "done"


def test_lease_job_checks_for_locks_acquired_after_dag_build() -> None:
    held = ExecutionCoordinatorLeaf(value=1)
    free = ExecutionCoordinatorLeaf(value=2)
    coordinator = _new_execution_coordinator([held, free])
    leased: list[Job | None] = []

    with mock.patch(
        "furu.execution.execution_coordinator._RUNNING_ELSEWHERE_POLL_INTERVAL_S",
        0.01,
    ):
        with _mark_running(held):
            assert set(coordinator.ready) == {held.object_id, free.object_id}
            assert _artifact(_lease_job(coordinator)).object_id == free.object_id
            assert set(coordinator.running_elsewhere) == {held.object_id}
            thread = threading.Thread(
                target=lambda: leased.append(_lease_job(coordinator))
            )
            thread.start()
            time.sleep(0.1)
            assert leased == []

        thread.join(timeout=10)
        assert not thread.is_alive()
    assert _artifact(leased[0]).object_id == held.object_id


def test_adopt_accepts_job_started_after_dag_build() -> None:
    held = ExecutionCoordinatorLeaf(value=1)
    coordinator = _new_execution_coordinator([held])

    with _mark_running(held):
        assert coordinator.adopt([ArtifactSpec.from_furu(held)], worker="w") is True

    assert coordinator.running[held.object_id].worker == "w"


def test_lease_job_checks_every_batch_member_for_active_locks() -> None:
    held = BatchedCoordinatorLeaf(value=1)
    free = BatchedCoordinatorLeaf(value=2)
    coordinator = _new_execution_coordinator([held, free])

    with _mark_running(held):
        job = _lease_job(coordinator)

    assert isinstance(job, Job)
    assert [artifact.object_id for artifact in job.artifacts] == [free.object_id]


class _InertPool:
    def __init__(self, job_ids: list[str]) -> None:
        self.job_ids = job_ids
        self.handoffs = 0
        self.stops = 0

    def handoff(self) -> PoolHandoff:
        self.handoffs += 1
        return PoolHandoff(job_ids=self.job_ids, worker_files=[])

    def stop(self, *, timeout: float) -> None:
        self.stops += 1


def _url_for(server: Any) -> str:
    return coordinator_url(
        host="127.0.0.1", port=server.bound_port, auth_token=server.auth_token
    )


def test_request_takeover_hands_off_matching_pools_and_closing_ends_old_run() -> None:
    leaf = ExecutionCoordinatorLeaf(value=1)
    old = _new_execution_coordinator([leaf])
    matched, unmatched = _InertPool(["100_0"]), _InertPool(["200_0"])
    old.pools = {"k": matched, "n": unmatched}
    new = _new_execution_coordinator([leaf])

    with execution_coordinator_server(old, bind_host="127.0.0.1", port=0) as server:
        with request_takeover(
            executor_id=new.executor_id,
            source_id=old.executor_id,
            url=_url_for(server),
            pool_keys=["k", "m"],
        ) as handoffs:
            assert handoffs == {"k": PoolHandoff(job_ids=["100_0"])}
            assert (matched.handoffs, unmatched.handoffs) == (1, 0)
            assert _lease_job(old) is None
            assert not old.done.is_set()
        _wait_until(old.done.is_set)

    assert old.finish_error == f"execution taken over by exec={new.executor_id[:5]}"


def test_request_takeover_hands_off_pools_outside_coordinator_lock() -> None:
    leaf = ExecutionCoordinatorLeaf(value=1)
    old = _new_execution_coordinator([leaf])

    class LockCheckingPool(_InertPool):
        def handoff(self) -> PoolHandoff:
            with pytest.raises(RuntimeError, match="un-acquired lock"):
                old.lock.notify()
            return super().handoff()

    old.pools = {"k": LockCheckingPool(["100_0"])}

    with execution_coordinator_server(old, bind_host="127.0.0.1", port=0) as server:
        with request_takeover(
            executor_id="b" * 32,
            source_id=old.executor_id,
            url=_url_for(server),
            pool_keys=["k"],
        ):
            pass
        _wait_until(old.done.is_set)


def test_request_takeover_without_matching_pool_is_refused() -> None:
    leaf = ExecutionCoordinatorLeaf(value=1)
    old = _new_execution_coordinator([leaf])
    pool = _InertPool(["100_0"])
    old.pools = {"k": pool}

    with execution_coordinator_server(old, bind_host="127.0.0.1", port=0) as server:
        with (
            pytest.raises(
                RuntimeError,
                match="refused the takeover: no worker pool with a matching",
            ),
            request_takeover(
                executor_id="b" * 32,
                source_id=old.executor_id,
                url=_url_for(server),
                pool_keys=["m"],
            ),
        ):
            raise AssertionError("unreachable")
        assert pool.handoffs == 0
        assert not old.done.is_set()


def test_request_takeover_refuses_second_concurrent_takeover() -> None:
    leaf = ExecutionCoordinatorLeaf(value=1)
    old = _new_execution_coordinator([leaf])
    old.pools = {"k": _InertPool(["100_0"])}

    with execution_coordinator_server(old, bind_host="127.0.0.1", port=0) as server:
        with (
            request_takeover(
                executor_id="b" * 32,
                source_id=old.executor_id,
                url=_url_for(server),
                pool_keys=["k"],
            ),
            pytest.raises(
                RuntimeError, match="refused the takeover: already being taken over"
            ),
            request_takeover(
                executor_id="c" * 32,
                source_id=old.executor_id,
                url=_url_for(server),
                pool_keys=["k"],
            ),
        ):
            raise AssertionError("unreachable")
        _wait_until(old.done.is_set)

    assert old.finish_error == f"execution taken over by exec={'b' * 5}"


def test_request_takeover_failing_midway_still_ends_old_run() -> None:
    leaf = ExecutionCoordinatorLeaf(value=1)
    old = _new_execution_coordinator([leaf])
    old.pools = {"k": _InertPool(["100_0"])}
    new = _new_execution_coordinator([leaf])

    with execution_coordinator_server(old, bind_host="127.0.0.1", port=0) as server:
        with (
            pytest.raises(OSError, match="sbatch script would not write"),
            request_takeover(
                executor_id=new.executor_id,
                source_id=old.executor_id,
                url=_url_for(server),
                pool_keys=["k"],
            ),
        ):
            raise OSError("sbatch script would not write")
        _wait_until(old.done.is_set)

    assert old.finish_error == f"execution taken over by exec={new.executor_id[:5]}"


def test_request_takeover_reports_unreachable_coordinator() -> None:
    with (
        pytest.raises(RuntimeError, match="cannot reach exec=7f3a1"),
        request_takeover(
            executor_id="b" * 32,
            source_id="7f3a1" + "0" * 27,
            url="ws://furu:token@127.0.0.1:1",
            pool_keys=[],
        ),
    ):
        raise AssertionError("unreachable")


def test_resolve_takeover_matches_unique_prefix() -> None:
    executions = get_config().run_directories.executions
    for executor_id in ("7f3a1" + "0" * 27, "7f3b2" + "0" * 27, "c09e4" + "0" * 27):
        (executions / executor_id).mkdir(parents=True)
    worker_file = _pool_worker_file(executions / ("7f3a1" + "0" * 27), "abc")
    worker_file.parent.mkdir(parents=True)
    _write_worker_config(
        worker_file,
        url="ws://furu:token@login01:41233",
    )

    assert _resolve_takeover("7f3a") == (
        "7f3a1" + "0" * 27,
        "ws://furu:token@login01:41233",
    )
    with pytest.raises(RuntimeError, match=r"matches 2 executions; candidates: 7f3a1"):
        _resolve_takeover("7f3")
    with pytest.raises(RuntimeError, match="matches 0 executions"):
        _resolve_takeover("zzz")
    with pytest.raises(RuntimeError, match="exec=c09e4 has no worker pools"):
        _resolve_takeover("c09e4")


def test_execution_coordinator_run_inherits_pools_on_takeover() -> None:
    class InertBackend:
        execution_coordinator_listen_host = "127.0.0.1"
        worker = Worker(gpus=1)
        accepts = None
        pool_key = "inert"

        def __init__(self) -> None:
            self.pool = _InertPool(["100_0", "100_1"])
            self.coordinators: list[ExecutionCoordinator] = []
            self.handoffs: list[PoolHandoff] = []

        def start_pool(
            self,
            *,
            coordinator: ExecutionCoordinator,
            bound_port: int,
            auth_token: str,
            executor_dir: Path,
            handoff: PoolHandoff,
        ) -> _InertPool:
            self.coordinators.append(coordinator)
            self.handoffs.append(handoff)
            worker_file = _pool_worker_file(executor_dir, self.pool_key)
            worker_file.parent.mkdir(parents=True)
            _write_worker_config(
                worker_file,
                url=coordinator_url(
                    host="127.0.0.1", port=bound_port, auth_token=auth_token
                ),
            )
            return self.pool

    leaf = ExecutionCoordinatorLeaf(value=uuid4().int)
    old_backend = InertBackend()
    old_errors: list[BaseException] = []

    def run_old() -> None:
        try:
            ExecutionCoordinator.run([leaf], worker_backends=(old_backend,))
        except BaseException as exc:
            old_errors.append(exc)

    old_thread = threading.Thread(target=run_old)
    old_thread.start()
    _wait_until(lambda: len(old_backend.handoffs) == 1)
    old = old_backend.coordinators[0]
    assert old_backend.handoffs == [PoolHandoff()]

    new_backend = InertBackend()
    with _taking_over(old.executor_id[:5]):
        ExecutionCoordinator.run(
            [leaf], worker_backends=(LocalThreadWorkerBackend(), new_backend)
        )
        assert old.executor_dir.is_dir()
        assert "FURU_TAKEOVER" not in os.environ
    old_thread.join(timeout=10)

    assert leaf.status == "done"
    (new,) = new_backend.coordinators
    (error,) = old_errors
    assert str(error) == f"execution taken over by exec={new.executor_id[:5]}"
    assert (old_backend.pool.handoffs, old_backend.pool.stops) == (1, 1)
    assert new_backend.handoffs == [PoolHandoff(job_ids=["100_0", "100_1"])]
