import logging
import re
import threading
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest
from pydantic import TypeAdapter, ValidationError
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.sync.client import ClientConnection, connect
from websockets.sync.server import ServerConnection, serve

import furu
import furu.worker.loop as worker_loop_module
from furu import Spec, Throttle, Worker
from furu.config import get_config
from furu.execution.manager import Manager, _Lease, _Queued, _Script
from furu.execution.run import Run
from furu.execution.run_on_workers import run_on_workers
from furu.locking import lock
from furu.logging import _display_path
from furu.metadata import ArtifactSpec
from furu.provenance import (
    EnvironmentIdentity,
    GitIdentity,
    SubmitContext,
    SubmitProvenance,
)
from furu.snapshot import CodeLocation
from furu.storage._layout import (
    compute_lock_path_in,
    execution_log_path_in,
    run_log_path_in,
)
from furu.worker.backends.local import LocalThreadWorkerBackend
from furu.worker.backends.slurm.backend import SlurmWorkerBackend
from furu.worker.backends.slurm.resources import SlurmResources
from furu.worker.execute import ChildSlot
from furu.worker.loop import worker_loop
from furu.worker.messages import (
    CancelMessage,
    HelloMessage,
    Job,
    JobBlockedResult,
    JobCompletedResult,
    JobFailedResult,
    JobResult,
    LocalPool,
    ProcessSettings,
    ResultMessage,
    SubmitMessage,
    job_result_adapter,
    manager_message_adapter,
    server_message_adapter,
)

CPU_POOL = LocalThreadWorkerBackend(worker=Worker())
GPU_POOL = LocalThreadWorkerBackend(worker=Worker(gpus=1, memory_gib=16))


def _pool(worker: Worker) -> LocalThreadWorkerBackend:
    return LocalThreadWorkerBackend(worker=worker)


def _submit_provenance() -> SubmitProvenance:
    # The git half is a stub since these tests never read it back.
    return SubmitProvenance(
        git=GitIdentity(commit="0" * 40, branch=None, remote=None, repo_root="."),
        environment=EnvironmentIdentity.capture(),
        submitted=SubmitContext.capture(),
    )


def _job(*objs: Spec) -> Job:
    return Job(
        artifacts=[ArtifactSpec.from_furu(obj) for obj in objs],
        run_logs=[run_log_path_in(obj._base_dir) for obj in objs],
        attempt=1,
        execution_log=get_config().directories.anchored().executions
        / "e1"
        / "execution.log",
        provenance=_submit_provenance(),
        process=ProcessSettings.from_metadata(objs[0]._metadata),
        config=get_config().anchored(),
        code=CodeLocation.here(),
    )


@contextmanager
def _captured_furu_logs(caplog: pytest.LogCaptureFixture) -> Generator[None]:
    furu_logger = logging.getLogger("furu")
    furu_logger.addHandler(caplog.handler)
    try:
        caplog.set_level(logging.DEBUG, logger="furu")
        yield
    finally:
        furu_logger.removeHandler(caplog.handler)


def _new_run(
    objs: Sequence[Spec],
    *,
    backends: Sequence[LocalThreadWorkerBackend] = (CPU_POOL, GPU_POOL),
    max_retries_per_object: int | None = None,
) -> Run:
    run = Run(
        backends={backend.pool_key: backend for backend in backends},
        provenance=_submit_provenance(),
        code=CodeLocation.here(),
        config=get_config().anchored(),
        execution_log=get_config().directories.anchored().executions
        / "e1"
        / "execution.log",
        max_retries_per_object=get_config().worker.max_retries_per_object
        if max_retries_per_object is None
        else max_retries_per_object,
    )
    run.add(objs)
    return run


def _assert_one_state(run: Run) -> None:
    """Every node is in exactly one state dict."""
    states = (
        run.ready,
        run.blocked,
        run.completed,
        run.running,
        run.waiting,
        run.failed,
    )
    ids = [object_id for state in states for object_id in state]
    assert sorted(ids) == sorted(run.nodes_by_id)


def _submitted_ids(run: Run) -> list[str]:
    return [
        submission.job.artifacts[0].object_id for submission in run.take_submissions()
    ]


# --- manager-side leasing, without sockets -------------------------------------


def _submit(
    manager: Manager,
    objs: Sequence[Spec],
    *,
    backends: Sequence[LocalThreadWorkerBackend] = (CPU_POOL, GPU_POOL),
    max_workers: int = 10,
) -> _Script:
    """Queue ``objs`` as one more script would, planned by its own Run."""
    run = _new_run(objs, backends=backends)
    script = _Script(
        connection=cast(ServerConnection, None),
        pools={key: LocalPool(max_workers=max_workers) for key in run.backends},
    )
    manager.scripts.append(script)
    manager.queue += [_Queued(script, job) for job in run.take_submissions()]
    return script


def _manager(
    objs: Sequence[Spec],
    *,
    backends: Sequence[LocalThreadWorkerBackend] = (CPU_POOL, GPU_POOL),
    max_workers: int = 10,
) -> Manager:
    manager = Manager(url="ws://unused", manager_dir=None)
    _submit(manager, objs, backends=backends, max_workers=max_workers)
    return manager


def _lease(
    manager: Manager, backend: LocalThreadWorkerBackend = CPU_POOL
) -> Job | None:
    queued = next(manager._leasable_locked(backend.pool_key), None)
    if queued is None:
        return None
    manager.queue = [q for q in manager.queue if q not in queued]
    worker = cast(ServerConnection, object())
    lease = manager.leases[worker] = _Lease(
        script=queued[0].script, queued=queued, pool=backend.pool_key
    )
    return lease.job


def _release(manager: Manager, job: Job) -> None:
    (worker,) = [
        worker
        for worker, lease in manager.leases.items()
        if lease.job.artifacts == job.artifacts
    ]
    del manager.leases[worker]


def _ids(job: Job | None) -> list[str]:
    assert isinstance(job, Job)
    return [artifact.object_id for artifact in job.artifacts]


@contextmanager
def _mark_running(obj: Spec) -> Generator[None]:
    obj._base_dir.mkdir(parents=True, exist_ok=True)
    with lock([compute_lock_path_in(obj._base_dir)]):
        yield


@dataclass(slots=True)
class _ScriptedServer:
    """A hand-rolled manager that plays a fixed sequence of jobs."""

    server_url: str
    hellos: list[HelloMessage] = field(default_factory=list)
    results: list[JobResult] = field(default_factory=list)


@contextmanager
def _scripted_worker_server(
    jobs: Sequence[Job], *, hold_open: bool = False
) -> Generator[_ScriptedServer]:
    record = _ScriptedServer(server_url="")

    def handler(connection: ServerConnection) -> None:
        record.hellos.append(
            HelloMessage.model_validate_json(connection.recv(timeout=5))
        )
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

    with _serve(handler) as url:
        record.server_url = url
        yield record


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


class RunLeaf(Spec[int]):
    value: int

    def create(self) -> int:
        return self.value


class FlakyRunLeaf(Spec[int]):
    value: int
    attempts_file: str

    def create(self) -> int:
        # Workers run create() in a child process; count attempts on disk.
        with open(self.attempts_file, "a", encoding="utf-8") as f:
            f.write("x")
        if Path(self.attempts_file).read_text(encoding="utf-8") == "x":
            raise RuntimeError(f"temporary failure: {self.value}")
        return self.value


class GatedRunLeaf(Spec[int]):
    value: int
    release_file: str

    def create(self) -> int:
        deadline = time.monotonic() + 30
        while not Path(self.release_file).exists():
            if time.monotonic() > deadline:
                raise TimeoutError("never released")
            time.sleep(0.01)
        return self.value


class LimitedRunLeaf(Spec[int]):
    value: int
    throttle = Throttle(max_running=2)

    def create(self) -> int:
        return self.value


class BatchedRunLeaf(furu.Spec[int]):
    value: int
    group: str = "g"
    cap: int = 10

    def batch_key(self, worker: furu.Worker) -> tuple[str, int]:
        return (self.group, self.cap)

    @furu.batched(batch_key)
    def create(objs: list["BatchedRunLeaf"]) -> list[int]:
        return [obj.value for obj in objs]


class BatchSizeRunLeaf(furu.Spec[int]):
    value: int

    @furu.batched(lambda _, __: (None, 10))
    def create(objs: list["BatchSizeRunLeaf"]) -> list[int]:
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


class ThrottledBatchedRunLeaf(furu.Spec[int]):
    value: int
    throttle = Throttle(max_running=2)

    def batch_key(self, worker: furu.Worker) -> tuple[None, int]:
        return (None, 3)

    @furu.batched(batch_key)
    def create(objs: list["ThrottledBatchedRunLeaf"]) -> list[int]:
        return [obj.value for obj in objs]


class RunParent(Spec[int]):
    child: RunLeaf

    def create(self) -> int:
        return self.child.create() + 1


class RunLazyParent(Spec[int]):
    value: int

    def create(self) -> int:
        return RunLeaf(value=self.value).create() + 1


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


class DynamicGpuAfterCpu(Spec[int]):
    parent: DynamicCpuAfterGpu
    value: int

    def runs_on(self, worker: Worker) -> bool:
        return worker.gpus >= 1

    def create(self) -> int:
        return self.parent.create() + self.value


# --- the script's run ----------------------------------------------------------


def test_run_partitions_ready_and_blocked() -> None:
    leaf = RunLeaf(value=1)
    parent = RunParent(child=leaf)

    run = _new_run([parent])

    assert set(run.ready) == {leaf.object_id}
    assert set(run.blocked) == {parent.object_id}
    assert run.running == {}


def test_run_max_retries_per_object_defaults_to_config() -> None:
    run = Run(
        backends={},
        provenance=_submit_provenance(),
        code=CodeLocation.here(),
        config=get_config(),
        execution_log=Path("execution.log"),
    )

    assert run.max_retries_per_object == get_config().worker.max_retries_per_object


def test_run_completed_result_moves_dependents_to_ready() -> None:
    leaf = RunLeaf(value=1)
    parent = RunParent(child=leaf)
    run = _new_run([parent])

    assert _submitted_ids(run) == [leaf.object_id]
    assert set(run.running) == {leaf.object_id}
    # Nothing else can go until the running job finishes.
    assert run.take_submissions() == []

    run.job_result(leaf.object_id, JobCompletedResult())

    assert run.running == {}
    assert set(run.completed) == {leaf.object_id}
    assert set(run.ready) == {parent.object_id}
    assert run.blocked == {}


def test_run_submission_carries_the_runs_settings_and_code() -> None:
    leaf = RunLeaf(value=1)
    run = _new_run([leaf], backends=(CPU_POOL,))

    (submission,) = run.take_submissions()

    assert submission.job.config == get_config().anchored()
    assert submission.job.code == CodeLocation.here()
    assert submission.job.run_logs == [run_log_path_in(leaf._base_dir)]
    assert submission.pools == {CPU_POOL.pool_key: None}
    assert submission.throttle is None


def test_run_blocked_result_discovers_lazy_dependency_and_reruns_parent() -> None:
    parent = RunLazyParent(value=2)
    dependency = RunLeaf(value=2)
    run = _new_run([parent])

    assert _submitted_ids(run) == [parent.object_id]
    run.job_result(
        parent.object_id,
        JobBlockedResult(dependencies=[ArtifactSpec.from_furu(dependency)]),
    )

    assert set(run.ready) == {dependency.object_id}
    assert set(run.blocked) == {parent.object_id}

    assert _submitted_ids(run) == [dependency.object_id]
    run.job_result(dependency.object_id, JobCompletedResult())

    assert _submitted_ids(run) == [parent.object_id]
    assert set(run.running) == {parent.object_id}
    assert set(run.completed) == {dependency.object_id}


def _blocked(*dependencies: Spec) -> JobBlockedResult:
    return JobBlockedResult(
        dependencies=[ArtifactSpec.from_furu(dependency) for dependency in dependencies]
    )


def _answer_until_idle(
    run: Run, answer: Callable[[str, int], JobResult], *, max_rounds: int = 50
) -> None:
    """Answer each round of submissions, polling in between, until none come."""
    for round_ in range(max_rounds):
        run.poll_waiting()
        object_ids = _submitted_ids(run)
        if not object_ids:
            return
        for object_id in object_ids:
            run.job_result(object_id, answer(object_id, round_))
            _assert_one_state(run)
    raise AssertionError(f"still submitting after {max_rounds} rounds")


def test_run_reruns_a_spec_blocked_on_a_built_dependency_after_a_poll() -> None:
    parent = RunLazyParent(value=2)
    dependency = RunLeaf(value=2)
    dependency.create()
    run = _new_run([parent])

    run.take_submissions()
    run.job_result(parent.object_id, _blocked(dependency))

    assert dependency.object_id not in run.nodes_by_id
    assert set(run.waiting) == {parent.object_id}
    assert run.take_submissions() == []
    _assert_one_state(run)
    run.poll_waiting()
    assert _submitted_ids(run) == [parent.object_id]


def test_run_keeps_rerunning_specs_whose_worker_lags_within_the_grace() -> None:
    dependency = RunLeaf(value=uuid4().int)
    parents = [BatchedRunLeaf(value=value) for value in range(2)]
    run = _new_run([dependency, *parents])

    def answer(object_id: str, round_: int) -> JobResult:
        if object_id == dependency.object_id:
            dependency.create()
            return JobCompletedResult()
        # Its worker misses the dependency for a few reruns, then sees it.
        return _blocked(dependency) if round_ < 4 else JobCompletedResult()

    _answer_until_idle(run, answer)

    assert set(run.completed) == {dependency.object_id} | {
        parent.object_id for parent in parents
    }
    run.finish()


def test_run_reruns_a_spec_once_per_dependency_published_meanwhile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Run, "unseen_grace_s", 0.0)
    parent = RunLazyParent(value=uuid4().int)
    dependencies = [RunLeaf(value=uuid4().int) for _ in range(3)]
    for dependency in dependencies:
        dependency.create()
    run = _new_run([parent])

    def answer(object_id: str, round_: int) -> JobResult:
        # Each lookup races another run publishing the next dependency.
        if round_ < len(dependencies):
            return _blocked(dependencies[round_])
        return JobCompletedResult()

    _answer_until_idle(run, answer)

    assert set(run.completed) == {parent.object_id}


@pytest.mark.parametrize(
    "missing",
    [
        pytest.param(lambda a, b, c, round_: [a], id="same"),
        pytest.param(lambda a, b, c, round_: [(a, b)[round_ % 2]], id="alternating"),
        pytest.param(
            lambda a, b, c, round_: [(a, b), (b, c), (c, a)][round_ % 3],
            id="overlapping",
        ),
        pytest.param(lambda a, b, c, round_: [a, b, c][: round_ + 1], id="growing"),
    ],
)
def test_run_fails_a_spec_that_keeps_missing_built_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    missing: Callable[[Spec, Spec, Spec, int], Sequence[Spec]],
) -> None:
    monkeypatch.setattr(Run, "unseen_grace_s", 0.0)
    parents = [BatchedRunLeaf(value=value) for value in range(2)]
    built_here = RunLeaf(value=uuid4().int)
    built_elsewhere = [RunLeaf(value=uuid4().int) for _ in range(2)]
    for dependency in built_elsewhere:
        dependency.create()
    other = RunLeaf(value=uuid4().int)
    run = _new_run([built_here, *parents, other], max_retries_per_object=1)

    def answer(object_id: str, round_: int) -> JobResult:
        if object_id == built_here.object_id:
            built_here.create()
            return JobCompletedResult()
        if object_id == other.object_id:
            return (
                JobFailedResult(error="boom") if round_ == 0 else JobCompletedResult()
            )
        return _blocked(*missing(built_here, *built_elsewhere, round_))

    _answer_until_idle(run, answer)

    assert set(run.failed) == {parent.object_id for parent in parents}
    with pytest.raises(RuntimeError) as excinfo:
        run.finish()
    lines = str(excinfo.value).splitlines()
    assert lines[0] == "2 specs failed, 0 blocked"
    for line in lines[1:]:
        assert re.search(
            r" · has found \S+ missing for \S+, though this run sees it built → ", line
        )


@pytest.mark.parametrize("built_here", [True, False])
def test_run_does_not_wait_on_a_built_dependency_of_a_new_spec(
    built_here: bool,
) -> None:
    leaf = RunLeaf(value=uuid4().int)
    lazy = RunLazyParent(value=uuid4().int)
    run = _new_run([leaf, lazy], max_retries_per_object=0)

    run.take_submissions()
    if built_here:
        run.job_result(leaf.object_id, JobCompletedResult())
    else:
        run.job_result(leaf.object_id, JobFailedResult(error="boom"))
        leaf.create()  # another run builds it after this one gave up
    parent = RunParent(child=leaf)
    run.job_result(lazy.object_id, _blocked(parent))

    assert set(run.ready) == {parent.object_id}
    assert run.nodes_by_id[parent.object_id].dependencies == []
    _assert_one_state(run)


def test_run_blocked_result_discovers_multiple_lazy_dependencies_together() -> None:
    parent = RunLazyParent(value=2)
    dependencies = [RunLeaf(value=2), RunLeaf(value=3)]
    run = _new_run([parent])

    run.take_submissions()
    run.job_result(
        parent.object_id,
        JobBlockedResult(
            dependencies=[
                ArtifactSpec.from_furu(dependency) for dependency in dependencies
            ]
        ),
    )

    assert set(run.ready) == {dependency.object_id for dependency in dependencies}
    assert set(run.blocked) == {parent.object_id}
    parent_node = run.nodes_by_id[parent.object_id]
    assert {node.obj.object_id for node in parent_node.dependencies} == {
        dependency.object_id for dependency in dependencies
    }
    for dependency in dependencies:
        assert parent_node in run.nodes_by_id[dependency.object_id].dependents


def test_run_fails_on_a_dependency_cycle() -> None:
    parent = RunLazyParent(value=2)
    dependency = RunLeaf(value=2)
    run = _new_run([parent])

    run.take_submissions()
    run.job_result(
        parent.object_id,
        JobBlockedResult(dependencies=[ArtifactSpec.from_furu(dependency)]),
    )
    run.take_submissions()
    with pytest.raises(RuntimeError) as excinfo:
        run.job_result(
            dependency.object_id,
            JobBlockedResult(dependencies=[ArtifactSpec.from_furu(parent)]),
        )

    assert str(excinfo.value) == (
        f"RuntimeError: dependency cycle: {parent._log_label} → "
        f"{dependency._log_label} → {parent._log_label}"
    )


def test_run_failed_result_finishes_with_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    leaf = RunLeaf(value=1)
    run = _new_run([leaf], max_retries_per_object=0)
    run.take_submissions()

    with _captured_furu_logs(caplog):
        run.job_result(leaf.object_id, JobFailedResult(error="boom"))
        with pytest.raises(RuntimeError, match="1 spec failed, 0 blocked"):
            run.finish()

    assert run.running == {}
    assert run.failed[leaf.object_id].obj is leaf
    assert run.failures[leaf.object_id] == ["boom"]
    _assert_one_state(run)
    failed_record = next(
        record for record in caplog.records if record.message.startswith("failed ")
    )
    assert failed_record.levelno == logging.ERROR
    assert failed_record.message == f"failed {leaf._log_label} · attempt 1/0 · boom"
    assert failed_record.__dict__["path"] == run_log_path_in(leaf._base_dir)
    assert (
        caplog.messages[-1] == f"run failed · 1 spec failed · {leaf._log_label} · boom"
    )


def test_run_failed_result_retries_before_finishing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    leaf = RunLeaf(value=1)
    run = _new_run([leaf], max_retries_per_object=2)

    with _captured_furu_logs(caplog):
        for attempt in (1, 2, 3):
            (submission,) = run.take_submissions()
            assert submission.job.attempt == attempt
            run.job_result(leaf.object_id, JobFailedResult(error=f"boom {attempt}"))
            assert run.failures[leaf.object_id] == [
                f"boom {n}" for n in range(1, attempt + 1)
            ]
            assert set(run.ready) == ({leaf.object_id} if attempt < 3 else set())
            assert set(run.failed) == ({leaf.object_id} if attempt == 3 else set())
            _assert_one_state(run)

    assert run.running == {}
    assert f"failed {leaf._log_label} · attempt 1/2 · boom 1" in caplog.messages
    assert f"failed {leaf._log_label} · attempt 2/2 · boom 2" in caplog.messages
    assert "progress 0/1 · 0 running · 1 ready · 1 retrying" in caplog.messages
    assert caplog.messages[-1] == "progress 0/1 · 0 running · 1 failed"
    with pytest.raises(RuntimeError, match="1 spec failed"):
        run.finish()


def test_run_retries_with_the_same_error_say_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    leaf = RunLeaf(value=1)
    run = _new_run([leaf], max_retries_per_object=3)
    errors = [
        "Traceback (most recent call last):\n  File a\nValueError: bad shape\n",
        "Traceback (most recent call last):\n  File b\nValueError: bad shape\n",
        "Traceback (most recent call last):\nOSError: disk full\n\n",
        "subprocess died: killed by signal 9 (SIGKILL)",
    ]

    with _captured_furu_logs(caplog):
        for error in errors:
            assert _submitted_ids(run) == [leaf.object_id]
            run.job_result(leaf.object_id, JobFailedResult(error=error))
        with pytest.raises(RuntimeError):
            run.finish()

    label = leaf._log_label
    assert [
        message for message in caplog.messages if message.startswith("failed ")
    ] == [
        f"failed {label} · attempt 1/3 · ValueError: bad shape",
        f"failed {label} · attempt 2/3 · same error",
        f"failed {label} · attempt 3/3 · OSError: disk full",
        f"failed {label} · attempt 4/3 · subprocess died: killed by signal 9 (SIGKILL)",
    ]
    assert all("Traceback" not in message for message in caplog.messages)
    assert caplog.messages[-1] == (
        f"run failed · 1 spec failed · {label} · "
        "subprocess died: killed by signal 9 (SIGKILL)"
    )


def test_run_error_lists_one_line_per_failed_and_blocked_spec() -> None:
    leaf = RunLeaf(value=1)
    other = RunLeaf(value=2)
    parent = RunParent(child=leaf)
    run = _new_run([parent, other], max_retries_per_object=1)

    for _ in range(2):
        run.take_submissions()
        for obj in (leaf, other):
            run.job_result(
                obj.object_id,
                JobFailedResult(error=f"Traceback\n  File x\nValueError: {obj.value}"),
            )

    _assert_one_state(run)
    with pytest.raises(RuntimeError) as excinfo:
        run.finish()
    lines = str(excinfo.value).splitlines()
    assert lines[0] == "2 specs failed, 1 blocked"
    assert sorted(lines[1:3]) == sorted(
        f"  {obj._log_label} · 2 attempts · ValueError: {obj.value} → "
        f"{_display_path(run_log_path_in(obj._base_dir))}"
        for obj in (leaf, other)
    )
    assert lines[3:] == [f"  {parent._log_label} · blocked"]


def test_run_failed_retry_can_later_complete() -> None:
    leaf = RunLeaf(value=1)
    run = _new_run([leaf], max_retries_per_object=1)

    run.take_submissions()
    run.job_result(leaf.object_id, JobFailedResult(error="boom"))
    assert run.failures[leaf.object_id] == ["boom"]
    assert run.failed == {}

    run.take_submissions()
    run.job_result(leaf.object_id, JobCompletedResult())

    assert run.failures == {}
    _assert_one_state(run)
    assert set(run.completed) == {leaf.object_id}
    run.finish()


def test_run_failed_retry_found_done_elsewhere_stops_retrying(
    caplog: pytest.LogCaptureFixture,
) -> None:
    leaf = RunLeaf(value=1)
    other = RunLeaf(value=2)
    run = _new_run([leaf, other], max_retries_per_object=1)

    run.take_submissions()
    run.job_result(leaf.object_id, JobFailedResult(error="boom"))
    leaf.create()  # another run builds it
    run.take_submissions()
    assert run.failures == {}
    _assert_one_state(run)

    caplog.set_level(logging.INFO, logger="furu.run")
    run.job_result(other.object_id, JobCompletedResult())
    assert caplog.messages[-1] == "progress 2/2 · 0 running"


def test_run_error_lists_failed_specs_in_order_of_first_failure() -> None:
    first, second = RunLeaf(value=1), RunLeaf(value=2)
    run = _new_run([first, second], max_retries_per_object=1)

    run.take_submissions()
    for obj in (first, second):
        run.job_result(obj.object_id, JobFailedResult(error=f"boom {obj.value}"))
    run.take_submissions()
    for obj in (second, first):  # second runs out of retries first
        run.job_result(obj.object_id, JobFailedResult(error=f"boom {obj.value}"))

    with pytest.raises(RuntimeError) as excinfo:
        run.finish()
    lines = str(excinfo.value).splitlines()
    assert [line.split(" · ")[0].strip() for line in lines[1:]] == [
        first._log_label,
        second._log_label,
    ]


def test_run_ignores_a_result_for_a_spec_it_is_not_running() -> None:
    leaf = RunLeaf(value=1)
    run = _new_run([leaf])

    run.job_result(leaf.object_id, JobCompletedResult())

    assert set(run.ready) == {leaf.object_id}
    assert run.completed == {}


def test_batched_group_failure_retries_each_member() -> None:
    objs = [BatchedRunLeaf(value=value) for value in range(2)]
    run = _new_run(objs, max_retries_per_object=1)

    run.take_submissions()
    for obj in objs:
        run.job_result(obj.object_id, JobFailedResult(error="boom"))

    assert run.failures == {obj.object_id: ["boom"] for obj in objs}
    assert set(run.ready) == {obj.object_id for obj in objs}
    _assert_one_state(run)


@pytest.mark.parametrize(
    "leaf_factory",
    [
        lambda value: RunLeaf(value=value),
        lambda value: BatchedRunLeaf(value=value),
    ],
)
def test_run_waits_on_specs_locked_elsewhere_until_their_lock_goes(
    caplog: pytest.LogCaptureFixture,
    leaf_factory: Callable[[int], Spec],
) -> None:
    leaves = [leaf_factory(value) for value in range(2)]
    run = _new_run(leaves)

    with _mark_running(leaves[0]), _mark_running(leaves[1]):
        with _captured_furu_logs(caplog):
            assert run.take_submissions() == []
        assert set(run.waiting) == {leaf.object_id for leaf in leaves}
        run.poll_waiting()
        assert run.take_submissions() == []

    run.poll_waiting()
    assert run.waiting == {}
    assert sorted(_submitted_ids(run)) == sorted(leaf.object_id for leaf in leaves)
    assert "waiting on 2 specs running elsewhere" in caplog.messages


def test_run_checks_every_batch_member_for_active_locks() -> None:
    held = BatchedRunLeaf(value=1)
    free = BatchedRunLeaf(value=2)
    run = _new_run([held, free])

    with _mark_running(held):
        assert _submitted_ids(run) == [free.object_id]


def test_run_completes_specs_finished_elsewhere_without_submitting() -> None:
    leaf = RunLeaf(value=1)
    parent = RunParent(child=leaf)
    run = _new_run([parent])
    leaf.create()

    assert _submitted_ids(run) == [parent.object_id]
    assert set(run.completed) == {leaf.object_id}


def test_run_fails_when_no_pool_can_run_a_lazy_dependency() -> None:
    parent = RunLazyParent(value=uuid4().int)
    run = _new_run([parent], backends=(CPU_POOL,))
    run.take_submissions()

    with pytest.raises(RuntimeError, match="no worker pool can run"):
        run.job_result(
            parent.object_id,
            JobBlockedResult(
                dependencies=[ArtifactSpec.from_furu(MemoryLeaf(value=uuid4().int))]
            ),
        )


def test_run_fails_on_a_zero_batch_cap_in_a_lazy_dependency() -> None:
    parent = RunLazyParent(value=uuid4().int)
    run = _new_run([parent])
    run.take_submissions()

    with pytest.raises(RuntimeError, match="cap must be a positive int, got 0"):
        run.job_result(
            parent.object_id,
            JobBlockedResult(
                dependencies=[ArtifactSpec.from_furu(PerGpuBatchedLeaf(value=1))]
            ),
        )


# --- the manager's queue -------------------------------------------------------


def test_demand_caps_at_max_workers_and_filters_by_requirements() -> None:
    objs = [RunLeaf(value=1), RunLeaf(value=2), GpuLeaf(value=3)]

    for max_workers, cpu_demand in ((10, 2), (1, 1), (0, 0)):
        manager = _manager(objs, max_workers=max_workers)
        assert manager.demand(CPU_POOL.pool_key) == cpu_demand
    assert _manager(objs).demand(GPU_POOL.pool_key) == 3


def test_demand_counts_busy_workers_and_drops_to_zero_without_scripts() -> None:
    manager = _manager([RunLeaf(value=1), RunLeaf(value=2)])
    assert _lease(manager) is not None

    assert manager.demand(CPU_POOL.pool_key) == 2

    manager.scripts.clear()
    assert manager.demand(CPU_POOL.pool_key) == 0


def test_throttle_limits_leases_and_demand() -> None:
    limited = [LimitedRunLeaf(value=value) for value in range(3)]
    uncapped = RunLeaf(value=10)
    manager = _manager([*limited, uncapped])

    assert manager.demand(CPU_POOL.pool_key) == 3

    first, second, third = (_lease(manager) for _ in range(3))
    limited_ids = {obj.object_id for obj in limited}
    leased_limited_ids = {*_ids(first), *_ids(second)}
    assert leased_limited_ids < limited_ids
    assert _ids(third) == [uncapped.object_id]
    # Busy workers still count; none more can start.
    assert manager.demand(CPU_POOL.pool_key) == 3
    assert _lease(manager) is None

    assert isinstance(first, Job)
    _release(manager, first)
    assert set(_ids(_lease(manager))) == limited_ids - leased_limited_ids


def test_lease_assembles_same_key_batched_group_into_one_job() -> None:
    objs = [BatchedRunLeaf(value=value) for value in range(3)]
    manager = _manager(objs)

    job = _lease(manager)

    assert _ids(job) == [obj.object_id for obj in objs]
    assert isinstance(job, Job)
    assert job.run_logs == [run_log_path_in(obj._base_dir) for obj in objs]
    assert manager.queue == []


def test_lease_groups_only_matching_batch_keys() -> None:
    first_x = BatchedRunLeaf(value=1, group="x")
    only_y = BatchedRunLeaf(value=2, group="y")
    second_x = BatchedRunLeaf(value=3, group="x")
    manager = _manager([first_x, only_y, second_x])

    assert _ids(_lease(manager)) == [first_x.object_id, second_x.object_id]
    assert _ids(_lease(manager)) == [only_y.object_id]


def test_lease_chunks_batched_group_to_the_cap() -> None:
    manager = _manager([BatchedRunLeaf(value=value, cap=2) for value in range(5)])

    assert [len(_ids(_lease(manager))) for _ in range(3)] == [2, 2, 1]
    assert manager.queue == []


def test_batch_cap_follows_the_leasing_worker() -> None:
    one_gpu, four_gpus, eight_gpus = (_pool(Worker(gpus=gpus)) for gpus in (1, 4, 8))
    manager = _manager(
        [GpuBatchedLeaf(value=value) for value in range(10)],
        backends=(one_gpu, four_gpus, eight_gpus),
        max_workers=20,
    )

    assert manager.demand(one_gpu.pool_key) == 10
    assert manager.demand(eight_gpus.pool_key) == 2
    assert manager.demand(four_gpus.pool_key) == 0

    assert len(_ids(_lease(manager, eight_gpus))) == 8
    assert len(_ids(_lease(manager, one_gpu))) == 1


def test_throttle_limits_concurrent_batches_not_members() -> None:
    manager = _manager([ThrottledBatchedRunLeaf(value=value) for value in range(8)])

    assert manager.demand(CPU_POOL.pool_key) == 2
    first = _lease(manager)
    assert len(_ids(first)) == 3
    assert len(_ids(_lease(manager))) == 3
    assert _lease(manager) is None

    assert isinstance(first, Job)
    _release(manager, first)
    assert len(_ids(_lease(manager))) == 2


def test_demand_counts_batched_groups() -> None:
    manager = _manager([BatchedRunLeaf(value=value) for value in range(4)])

    assert manager.demand(CPU_POOL.pool_key) == 1


def test_manager_throttles_across_scripts_and_batches_within_one() -> None:
    manager = Manager(url="ws://unused", manager_dir=None)
    scripts = [
        _submit(manager, [ThrottledBatchedRunLeaf(value=value) for value in values])
        for values in ((0, 1), (2, 3), (4,))
    ]

    first, second = _lease(manager), _lease(manager)

    # Both scripts' batches are called "b0", yet batch-mates never cross scripts.
    assert [lease.script for lease in manager.leases.values()] == scripts[:2]
    assert len(_ids(first)) == len(_ids(second)) == 2
    # The throttle counts the two running batches across both scripts.
    assert _lease(manager) is None
    assert isinstance(first, Job)
    _release(manager, first)
    assert len(_ids(_lease(manager))) == 1
    assert [lease.script for lease in manager.leases.values()] == scripts[1:]


def test_lease_filters_by_worker_resources() -> None:
    cpu_leaf = CpuOnlyLeaf(value=1)
    gpu_leaf = GpuLeaf(value=2)
    gpu_pool = _pool(Worker(gpus=1))
    manager = _manager([cpu_leaf, gpu_leaf], backends=(CPU_POOL, gpu_pool))

    assert _ids(_lease(manager, CPU_POOL)) == [cpu_leaf.object_id]
    assert _lease(manager, CPU_POOL) is None
    assert _ids(_lease(manager, gpu_pool)) == [gpu_leaf.object_id]


def test_lease_filters_by_worker_memory_gib() -> None:
    memory_leaf = MemoryLeaf(value=1)
    small, large = _pool(Worker(memory_gib=7)), _pool(Worker(memory_gib=8))
    manager = _manager([memory_leaf], backends=(small, large))

    assert _lease(manager, small) is None
    assert _ids(_lease(manager, large)) == [memory_leaf.object_id]


def test_lease_honors_pool_accepts() -> None:
    cpu_leaf = CpuOnlyLeaf(value=1)
    memory_leaf = MemoryLeaf(value=2)
    reserved = LocalThreadWorkerBackend(
        worker=Worker(memory_gib=16),
        accepts=lambda spec: isinstance(spec, MemoryLeaf),
    )
    open_pool = _pool(Worker(memory_gib=32))
    manager = _manager([cpu_leaf, memory_leaf], backends=(reserved, open_pool))

    assert _ids(_lease(manager, reserved)) == [memory_leaf.object_id]
    assert _lease(manager, reserved) is None
    assert _ids(_lease(manager, open_pool)) == [cpu_leaf.object_id]


# --- the manager's server ------------------------------------------------------


def _hello(
    url: str, *, worker: str = "raw-test-worker", pool: str = "cpu"
) -> ClientConnection:
    connection = connect(url, legacy=True)
    connection.send(
        HelloMessage(worker=worker, backend="test", pool=pool).model_dump_json()
    )
    return connection


def _submit_over(manager: Manager, *objs: Spec) -> ClientConnection:
    """Connect as a script whose jobs only the raw "cpu" pool runs."""
    run = _new_run(objs, backends=(CPU_POOL,))
    jobs = [
        submission.model_copy(update={"pools": {"cpu": None}})
        for submission in run.take_submissions()
    ]
    connection = connect(manager.url, legacy=True)
    connection.send(
        SubmitMessage(
            pools={"cpu": LocalPool(max_workers=0)}, jobs=jobs
        ).model_dump_json()
    )
    # Workers of a pool the manager has not started yet are turned away.
    deadline = time.monotonic() + 5
    while len(manager.queue) < len(jobs) and time.monotonic() < deadline:
        time.sleep(0.01)
    return connection


def test_manager_url_carries_a_token_and_rejects_others() -> None:
    with Manager.serve(
        listen_host="127.0.0.1", connect_host="127.0.0.1", manager_dir=None
    ) as manager:
        token, _, address = manager.url.removeprefix("ws://furu:").partition("@")
        assert token
        assert int(address.rpartition(":")[2]) > 0

        for url in (f"ws://{address}", f"ws://furu:wrong@{address}"):
            with pytest.raises(InvalidStatus) as rejected:
                connect(url)
            assert rejected.value.response.status_code == 401

        with connect(manager.url):
            pass


def test_job_round_trip_from_script_through_manager_to_worker() -> None:
    leaf = RunLeaf(value=1)

    with (
        Manager.serve(
            listen_host="127.0.0.1", connect_host="127.0.0.1", manager_dir=None
        ) as manager,
        _submit_over(manager, leaf) as script,
        _hello(manager.url) as worker,
    ):
        job = Job.model_validate_json(worker.recv(timeout=5))
        (artifact,) = job.artifacts
        assert artifact.object_id == leaf.object_id
        assert artifact.artifact_data["|fields"] == {"value": 1}

        worker.send(JobCompletedResult().model_dump_json())
        assert manager_message_adapter.validate_json(
            script.recv(timeout=5)
        ) == ResultMessage(object_ids=[leaf.object_id], result=JobCompletedResult())


def test_worker_disconnect_fails_its_job_for_the_script() -> None:
    leaf = RunLeaf(value=1)

    with (
        Manager.serve(
            listen_host="127.0.0.1", connect_host="127.0.0.1", manager_dir=None
        ) as manager,
        _submit_over(manager, leaf) as script,
    ):
        worker = _hello(manager.url, worker="doomed-worker")
        Job.model_validate_json(worker.recv(timeout=5))
        worker.close()

        assert manager_message_adapter.validate_json(
            script.recv(timeout=5)
        ) == ResultMessage(
            object_ids=[leaf.object_id],
            result=JobFailedResult(error="worker doomed-worker disconnected"),
        )


def test_worker_dropped_by_manager_shutdown_does_not_fail_its_job() -> None:
    # Shutdown closes connections in any order; a worker carries on alone.
    with (
        Manager.serve(
            listen_host="127.0.0.1", connect_host="127.0.0.1", manager_dir=None
        ) as manager,
        _submit_over(manager, RunLeaf(value=1)) as script,
    ):
        worker = _hello(manager.url)
        Job.model_validate_json(worker.recv(timeout=5))
        manager.stopping = True
        worker.close()

        with pytest.raises(TimeoutError):
            script.recv(timeout=0.5)


def test_manager_shutdown_closes_busy_and_idle_workers_promptly() -> None:
    started = time.monotonic()
    with Manager.serve(
        listen_host="127.0.0.1", connect_host="127.0.0.1", manager_dir=None
    ) as manager:
        script = _submit_over(manager, RunLeaf(value=1))
        busy = _hello(manager.url, worker="busy")
        Job.model_validate_json(busy.recv(timeout=5))
        # No job left, so this handler waits for one without touching the socket.
        idle = _hello(manager.url, worker="idle")

    assert time.monotonic() - started < 5
    for connection in (script, busy, idle):
        with pytest.raises(ConnectionClosed):
            connection.recv(timeout=5)


def test_manager_shutdown_forwards_the_cancel_of_a_script_that_just_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A Ctrl-C'd script cancels, then shuts its in-process manager down while
    # the script's handler may still be forwarding; make it slow to force that.
    cancel = Manager._cancel

    def slow_cancel(self: Manager, script: _Script) -> None:
        time.sleep(0.5)
        cancel(self, script)

    monkeypatch.setattr(Manager, "_cancel", slow_cancel)
    with Manager.serve(
        listen_host="127.0.0.1", connect_host="127.0.0.1", manager_dir=None
    ) as manager:
        script = _submit_over(manager, RunLeaf(value=1))
        worker = _hello(manager.url)
        Job.model_validate_json(worker.recv(timeout=5))
        script.send(CancelMessage().model_dump_json())
        script.close()

    assert server_message_adapter.validate_json(worker.recv(timeout=5)) == (
        CancelMessage()
    )


# --- create(..., on=[...]) end to end ------------------------------------------


def test_run_retries_failed_worker_result(tmp_path: Path) -> None:
    attempts_file = tmp_path / "attempts"
    leaf = FlakyRunLeaf(value=1, attempts_file=str(attempts_file))

    assert furu.create(leaf, on=[LocalThreadWorkerBackend()]) == 1
    assert attempts_file.read_text(encoding="utf-8") == "xx"


def test_runs_batched_specs_as_one_batch() -> None:
    objs = [BatchSizeRunLeaf(value=value) for value in range(3)]

    assert furu.create(objs, on=[LocalThreadWorkerBackend()]) == [3, 3, 3]


def test_build_on_workers_leaves_results_on_disk() -> None:
    objs = [BatchSizeRunLeaf(value=value) for value in range(3)]

    furu.build(objs, on=[LocalThreadWorkerBackend()])

    assert [obj.status for obj in objs] == ["done", "done", "done"]


def test_load_existing_in_worker_does_not_build_missing_spec() -> None:
    parent = OptionalLoadParent(name="optional")

    assert furu.create(parent, on=[LocalThreadWorkerBackend()]) == "missing"
    assert OptionalLoadLeaf(name="optional").status == "missing"


def test_top_level_create_runs_dag_on_worker_backends() -> None:
    leaf = RunLeaf(value=21)

    assert furu.create(leaf, on=(LocalThreadWorkerBackend(),)) == 21
    assert leaf.status == "done"
    assert furu.create([leaf], on=(LocalThreadWorkerBackend(),)) == [21]


def test_run_fails_fast_when_no_pool_can_run_a_job() -> None:
    with pytest.raises(RuntimeError, match=r"no worker pool can run.*workers: Worker"):
        furu.create(
            MemoryLeaf(value=uuid4().int),
            on=[LocalThreadWorkerBackend(worker=Worker())],
        )


def test_run_rejects_a_zero_batch_cap_before_submitting() -> None:
    with pytest.raises(TypeError, match=r"cap must be a positive int, got 0 on Worker"):
        furu.create(
            PerGpuBatchedLeaf(value=uuid4().int),
            on=[
                LocalThreadWorkerBackend(worker=Worker(gpus=1)),
                LocalThreadWorkerBackend(worker=Worker()),
            ],
        )


def test_run_splits_work_across_open_and_reserved_pools() -> None:
    cpu_leaf = CpuOnlyLeaf(value=uuid4().int)
    memory_leaf = MemoryLeaf(value=uuid4().int)

    furu.build(
        [cpu_leaf, memory_leaf],
        on=[
            LocalThreadWorkerBackend(worker=Worker()),
            LocalThreadWorkerBackend(
                worker=Worker(memory_gib=16),
                accepts=lambda spec: isinstance(spec, MemoryLeaf),
            ),
        ],
    )

    assert cpu_leaf.status == "done"
    assert memory_leaf.status == "done"


def test_run_completes_later_resource_stages_on_local_workers() -> None:
    seed_value = uuid4().int
    seed = DynamicCpuSeed(value=seed_value)
    first_gpu = DynamicGpuAfterSeed(parent=seed, value=20)
    second_cpu = DynamicCpuAfterGpu(parent=first_gpu, value=30)
    final_gpus = [
        DynamicGpuAfterCpu(parent=second_cpu, value=value) for value in range(4)
    ]

    furu.build(
        final_gpus,
        on=[
            LocalThreadWorkerBackend(max_workers=1, worker=Worker(gpus=0)),
            LocalThreadWorkerBackend(max_workers=3, worker=Worker(gpus=1)),
        ],
    )

    for obj in (seed, first_gpu, second_cpu, *final_gpus):
        assert obj.status == "done"
    assert [obj.create() for obj in final_gpus] == [
        seed_value + 50 + value for value in range(4)
    ]


def test_run_fails_when_local_worker_crashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def crashing_worker_loop(**_: object) -> None:
        raise RuntimeError("worker boom")

    monkeypatch.setattr(worker_loop_module, "worker_loop", crashing_worker_loop)

    with pytest.raises(
        RuntimeError, match="local worker thread crashed: RuntimeError: worker boom"
    ):
        furu.create(RunLeaf(value=42), on=[LocalThreadWorkerBackend(max_workers=1)])


def test_run_writes_log_to_its_execution_dir() -> None:
    leaf = RunLeaf(value=14)
    executions_dir = get_config().directories.anchored().executions

    run_on_workers([leaf], [LocalThreadWorkerBackend()])

    (executor_dir,) = executions_dir.iterdir()
    log_path = execution_log_path_in(executor_dir)
    # Local workers write no files of their own; their lines are in here.
    assert [path.name for path in executor_dir.iterdir()] == ["execution.log"]

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert any(f" planned exec {executor_dir.name[:5]} · " in line for line in lines)
    assert any(
        line.endswith(
            f" leased {leaf._log_label} to w0 → {run_log_path_in(leaf._base_dir)}"
        )
        for line in lines
    )
    assert any(" D w0    spawned child " in line for line in lines)
    assert any(f" completed {leaf._log_label} ok · " in line for line in lines)
    assert any(line.endswith(" progress 1/1 · 0 running") for line in lines)
    assert any(line.endswith(" run finished ok") for line in lines)
    log_text = "\n".join(lines)
    assert f"creating {leaf._log_label}" not in log_text
    assert "received" not in log_text


def test_run_returns_when_all_objects_are_already_completed() -> None:
    leaf = RunLeaf(value=15)
    leaf.create()
    executions_dir = get_config().directories.anchored().executions

    run_on_workers([leaf], [LocalThreadWorkerBackend()])

    # A no-op run returns before creating an executor dir or capturing
    # provenance; nothing appears under executions/.
    assert not executions_dir.exists() or list(executions_dir.iterdir()) == []


def test_local_pool_key_distinguishes_resources_not_worker_count() -> None:
    backend = LocalThreadWorkerBackend(max_workers=2)

    assert backend.pool_key.startswith("local:")
    assert LocalThreadWorkerBackend(max_workers=3).pool_key == backend.pool_key
    assert (
        LocalThreadWorkerBackend(max_workers=2, worker=Worker(gpus=1)).pool_key
        != backend.pool_key
    )


@pytest.mark.parametrize(
    ("backends", "match"),
    [
        ([], "at least one worker backend"),
        (
            [LocalThreadWorkerBackend(), LocalThreadWorkerBackend()],
            "identical configuration",
        ),
        (
            [
                LocalThreadWorkerBackend(),
                SlurmWorkerBackend(
                    max_workers=1, resources=SlurmResources(cpus_per_worker=1)
                ),
            ],
            "cannot mix local and Slurm",
        ),
    ],
)
def test_run_rejects_bad_backend_sets(backends: list, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        run_on_workers([RunLeaf(value=12)], backends)


# --- messages ------------------------------------------------------------------


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


def test_server_message_adapter_discriminates_job_and_cancel() -> None:
    job = _job(RunLeaf(value=1))

    assert server_message_adapter.validate_json(job.model_dump_json()) == job
    assert (
        server_message_adapter.validate_json(CancelMessage().model_dump_json())
        == CancelMessage()
    )


# --- the worker ----------------------------------------------------------------


def test_worker_loop_raises_when_manager_is_unavailable() -> None:
    with pytest.raises(OSError):
        worker_loop(
            url="ws://127.0.0.1:1",
            pool="cpu",
            idle_timeout=get_config().worker.idle_timeout_seconds,
            component="test-worker",
            backend="test",
        )


def test_worker_loop_exits_after_idle_timeout() -> None:
    with _scripted_worker_server([], hold_open=True) as server:
        worker_loop(
            url=server.server_url,
            pool="cpu",
            idle_timeout=0.05,
            component="test-worker",
            backend="test",
        )

        assert len(server.hellos) == 1
        assert server.results == []


def test_worker_with_its_own_log_notes_what_it_ran_there(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    leaf = BatchedRunLeaf(value=1)
    other_leaf = BatchedRunLeaf(value=2)
    job = _job(leaf, other_leaf)
    worker_log = tmp_path / "logs" / "100_0.log"

    with _scripted_worker_server([job]) as server, _captured_furu_logs(caplog):
        worker_loop(
            url=server.server_url,
            pool="cpu",
            idle_timeout=get_config().worker.idle_timeout_seconds,
            component="100_0",
            backend="test",
            worker_log=worker_log,
        )

        assert server.results == [JobCompletedResult()]
        (hello,) = server.hellos
        assert hello.log == worker_log

    assert (leaf.create(), other_leaf.create()) == (1, 2)
    worker_records = [
        (record.levelno, record.message, getattr(record, "path", None))
        for record in caplog.records
        if record.name == "furu.100_0" and record.levelno >= logging.INFO
    ]
    assert worker_records == [
        (logging.INFO, "serving exec e1", job.execution_log),
        (logging.INFO, f"running {leaf._log_label} ×2", job.run_logs[0]),
        (logging.INFO, "manager closed the connection; exiting", None),
    ]
    assert f"   worker log → {worker_log}\n" in job.run_logs[0].read_text()
    assert f"   batched with {leaf._log_label} → " in job.run_logs[1].read_text()


def test_worker_loop_does_not_swallow_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(self: ChildSlot, job: Job, *, cancelled: threading.Event) -> JobResult:
        raise KeyboardInterrupt

    monkeypatch.setattr(ChildSlot, "run", run)

    with _scripted_worker_server([_job(RunLeaf(value=1))]) as server:
        with pytest.raises(KeyboardInterrupt):
            worker_loop(
                url=server.server_url,
                pool="gpu",
                idle_timeout=get_config().worker.idle_timeout_seconds,
                component="test-worker",
                backend="test",
            )

        assert server.results == []
        (hello,) = server.hellos
        assert (hello.pool, hello.worker, hello.backend) == (
            "gpu",
            "test-worker",
            "test",
        )


def test_worker_loop_cancel_kills_running_job(tmp_path: Path) -> None:
    leaf = GatedRunLeaf(value=1, release_file=str(tmp_path / "go"))
    results: list[JobResult] = []

    def handler(connection: ServerConnection) -> None:
        HelloMessage.model_validate_json(connection.recv(timeout=5))
        connection.send(_job(leaf).model_dump_json())
        connection.send(CancelMessage().model_dump_json())
        results.append(job_result_adapter.validate_json(connection.recv(timeout=10)))

    with _serve(handler) as url:
        worker_loop(
            url=url, pool="cpu", idle_timeout=5, component="test-worker", backend="test"
        )

    (result,) = results
    assert isinstance(result, JobFailedResult)
    assert result.error.endswith("subprocess died: killed by signal 9 (SIGKILL)")
    assert leaf.status != "done"


def test_worker_finishes_its_job_after_losing_the_manager_then_exits(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    release_file = tmp_path / "go"
    leaf = GatedRunLeaf(value=1, release_file=str(release_file))

    def handler(connection: ServerConnection) -> None:
        HelloMessage.model_validate_json(connection.recv(timeout=5))
        connection.send(_job(leaf).model_dump_json())
        connection.close()
        release_file.touch()

    with _serve(handler) as url, _captured_furu_logs(caplog):
        worker_loop(
            url=url, pool="cpu", idle_timeout=5, component="test-worker", backend="test"
        )

    assert leaf.status == "done"
    assert (
        f"lost the manager; finishing {leaf._log_label}, then exiting"
        in caplog.messages
    )
    assert "manager closed the connection; exiting" in caplog.messages
