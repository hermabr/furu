import json
import logging
import threading
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest
from pydantic import TypeAdapter, ValidationError
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.protocol import State
from websockets.sync.client import ClientConnection, connect
from websockets.sync.server import ServerConnection, serve

import furu
import furu.dag as dag_module
import furu.execution.manager as manager_module
import furu.worker.loop as worker_loop_module
from furu import Spec, Throttle, Worker, plan, slurm
from furu.config import get_config
from furu.core import attempt_parent
from furu.dag import Dag
from furu.execution.launch import missing
from furu.execution.manager import Manager, _Lease, _Script
from furu.execution.run import Run
from furu.execution.run_on_workers import run_on_workers
from furu.logging import _display_path, _group_label
from furu.snapshot import CodeLocation
from furu.storage import attempt_lock, events
from furu.storage._layout import (
    attempt_parent_in,
    execution_log_path_in,
    run_name,
)
from furu.worker.backends.local import LocalThreadWorkerBackend
from furu.worker.backends.slurm.backend import SlurmWorkerBackend
from furu.worker.backends.slurm.resources import SlurmResources
from furu.worker.execute import ChildSlot
from furu.worker.loop import worker_loop
from furu.worker.messages import (
    Batch,
    CancelMessage,
    ErrorMessage,
    GaveUpMessage,
    HelloMessage,
    Job,
    JobBlockedResult,
    JobCompletedResult,
    JobFailedResult,
    JobResult,
    JobThrottle,
    Launch,
    LocalPool,
    ProcessSettings,
    RerunMessage,
    ResultMessage,
    SubmitMessage,
    WarningMessage,
    job_result_adapter,
    manager_message_adapter,
    server_message_adapter,
)

CPU_POOL = LocalThreadWorkerBackend(worker=Worker())
GPU_POOL = LocalThreadWorkerBackend(worker=Worker(gpus=1, memory_gib=16))


def _pool(worker: Worker) -> LocalThreadWorkerBackend:
    return LocalThreadWorkerBackend(worker=worker)


def _plan(name: str = "e1") -> Path:
    return get_config().directories.anchored().plans / f"{name}.plan"


def _launch() -> Launch:
    return Launch(
        plan=_plan(),
        execution_log=get_config().directories.anchored().executions
        / "e1"
        / "execution.log",
        config=get_config().anchored(),
        code=CodeLocation.here(),
    )


def _job(*objs: Spec) -> Job:
    return Job(
        specs=[obj.object_id for obj in objs],
        lines=list(map(plan.dump, plan.lines(objs))),
        class_dir=objs[0]._class_dir,
        process=ProcessSettings.from_metadata(objs[0]._metadata),
        **dict(_launch()),
    )


def _ids(objs: Sequence[Spec]) -> list[str]:
    return [obj.object_id for obj in objs]


def _lines(objs: Sequence[Spec], runs: Sequence[Spec]) -> list[str]:
    return list(map(plan.dump, plan.lines(objs, set(_ids(runs)))))


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
    plan_path: Path | None = None,
) -> tuple[Run, SubmitMessage]:
    """A script's run of ``objs``, and the message that asks the manager for them."""
    run = Run(
        backends={backend.pool_key: backend for backend in backends},
        launch=_launch(),
        max_retries_per_object=get_config().worker.max_retries_per_object
        if max_retries_per_object is None
        else max_retries_per_object,
    )
    running = missing(objs, Dag())
    lines = _lines(running, running)
    return run, run.submit(run.place(running), lines, plan_path or _plan())


def _result(objs: Sequence[Spec], result: JobResult) -> ResultMessage:
    return ResultMessage(object_ids=_ids(objs), result=result, seconds=1.5)


def _gave_up(objs: Sequence[Spec], error: str, attempts: int = 1) -> GaveUpMessage:
    """What a run sends that gives up on one call's ``objs`` after ``attempts`` each."""
    return GaveUpMessage(
        specs=_ids(objs),
        error=error,
        folder=attempt_parent(objs),
        attempts=dict.fromkeys(_ids(objs), attempts),
    )


def _blocked(*dependencies: Spec) -> JobBlockedResult:
    """What a worker reports whose create() asked for ``dependencies``."""
    under = [*dependencies, *missing(dependencies, Dag())]
    return JobBlockedResult(
        dependencies=_ids(dependencies),
        lines=_lines(dependencies, under),
        plan=_plan().with_suffix("") / "w0.plan",
    )


# --- the manager, without sockets ----------------------------------------------


def _submit(
    manager: Manager,
    objs: Sequence[Spec],
    *,
    backends: Sequence[LocalThreadWorkerBackend] = (CPU_POOL, GPU_POOL),
    max_workers: int = 10,
    plan_path: Path | None = None,
) -> tuple[_Script, Run]:
    """Send ``objs`` as one more script would, placed by its own Run."""
    run, message = _new_run(objs, backends=backends, plan_path=plan_path)
    script = _Script(
        connection=cast(ServerConnection, None),
        pools={key: LocalPool(max_workers=max_workers) for key in run.backends},
    )
    manager.scripts.append(script)
    manager._add(script, message)
    return script, run


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
) -> _Lease | None:
    # As Manager._lease does: the stopped batches that are due are looked at
    # first, without the lock.
    manager._look(backend.pool_key)
    return manager._hand_out(cast(ServerConnection, object()), backend.pool_key)


def _leased(manager: Manager, backend: LocalThreadWorkerBackend = CPU_POOL) -> _Lease:
    lease = _lease(manager, backend)
    assert lease is not None
    return lease


def _end(manager: Manager, lease: _Lease, result: JobResult | None = None) -> None:
    (worker,) = [w for w, held in manager.leases.items() if held is lease]
    del manager.leases[worker]
    manager._finish(lease, result or JobCompletedResult())


def _sent(manager: Manager) -> list[object]:
    """What the manager has to send, with every job taking no time."""
    outbox, manager.outbox = manager.outbox, []
    return [
        message.model_copy(update={"seconds": 0.0})
        if isinstance(message, ResultMessage)
        else message
        for _, message in outbox
    ]


def _answer(manager: Manager, script: _Script, run: Run) -> None:
    """Pass what the manager sent on to the script's run, and its replies back."""
    outbox, manager.outbox = manager.outbox, []
    for _, message in outbox:
        match message:
            case ResultMessage():
                replies = run.result(message)
            case GaveUpMessage():
                run.gave_up(message)
                replies = []
            case _:
                raise AssertionError(message)
        for reply in replies:
            match reply:
                case SubmitMessage():
                    manager._add(script, reply)
                case RerunMessage():
                    manager._rerun(script, reply)
                case GaveUpMessage():
                    manager._gave_up(script, reply)


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


class RunLeaf(Spec):
    value: int

    def create(self) -> int:
        return self.value


class FlakyRunLeaf(Spec):
    value: int
    attempts_file: str

    def create(self) -> int:
        # Workers run create() in a child process; count attempts on disk.
        with open(self.attempts_file, "a", encoding="utf-8") as f:
            f.write("x")
        if Path(self.attempts_file).read_text(encoding="utf-8") == "x":
            raise RuntimeError(f"temporary failure: {self.value}")
        return self.value


class GatedRunLeaf(Spec):
    value: int
    release_file: str

    def create(self) -> int:
        deadline = time.monotonic() + 30
        while not Path(self.release_file).exists():
            if time.monotonic() > deadline:
                raise TimeoutError("never released")
            time.sleep(0.01)
        return self.value


class LimitedRunLeaf(Spec):
    value: int
    throttle = Throttle(max_running=2)

    def create(self) -> int:
        return self.value


class BatchedRunLeaf(furu.Spec):
    value: int
    group: str = "g"
    cap: int = 10

    def batch_key(self, worker: furu.Worker) -> tuple[str, int]:
        return (self.group, self.cap)

    @furu.batched(batch_key)
    def create(
        objs: list["BatchedRunLeaf"], directory: furu.SpecDirectory
    ) -> list[int]:
        return [obj.value for obj in objs]


class ChainedBatchedLeaf(furu.Spec):
    value: int
    previous: "ChainedBatchedLeaf | None" = None

    @furu.batched(lambda _, __: (None, 10))
    def create(
        objs: list["ChainedBatchedLeaf"], directory: furu.SpecDirectory
    ) -> list[int]:
        return [obj.value for obj in objs]


class MixedMetadataLeaf(furu.Spec):
    value: int

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(reuse="never" if self.value % 2 else "same_environment")

    @furu.batched(lambda _, __: (None, 10))
    def create(
        objs: list["MixedMetadataLeaf"], directory: furu.SpecDirectory
    ) -> list[int]:
        return [obj.value for obj in objs]


class BatchSizeRunLeaf(furu.Spec):
    value: int

    @furu.batched(lambda _, __: (None, 10))
    def create(
        objs: list["BatchSizeRunLeaf"], directory: furu.SpecDirectory
    ) -> list[int]:
        return [len(objs)] * len(objs)


class OptionalLoadLeaf(furu.Spec):
    name: str

    def create(self) -> str:
        return self.name


class OptionalLoadParent(furu.Spec):
    name: str

    def create(self) -> str:
        try:
            return OptionalLoadLeaf(name=self.name).load_existing()
        except furu.Missing:
            return "missing"


class GpuBatchedLeaf(furu.Spec):
    value: int

    def runs_on(self, worker: Worker) -> bool:
        return worker.gpus in (1, 8)

    @furu.batched(lambda _, worker: (None, worker.gpus))
    def create(
        objs: list["GpuBatchedLeaf"], directory: furu.SpecDirectory
    ) -> list[int]:
        return [obj.value for obj in objs]


class PerGpuBatchedLeaf(furu.Spec):
    value: int

    @furu.batched(lambda _, worker: (None, worker.gpus))
    def create(
        objs: list["PerGpuBatchedLeaf"], directory: furu.SpecDirectory
    ) -> list[int]:
        return [obj.value for obj in objs]


class ThrottledBatchedRunLeaf(furu.Spec):
    value: int
    throttle = Throttle(max_running=2)

    def batch_key(self, worker: furu.Worker) -> tuple[None, int]:
        return (None, 3)

    @furu.batched(batch_key)
    def create(
        objs: list["ThrottledBatchedRunLeaf"], directory: furu.SpecDirectory
    ) -> list[int]:
        return [obj.value for obj in objs]


class RunParent(Spec):
    child: RunLeaf

    def create(self) -> int:
        return self.child.create() + 1


class RunLazyParent(Spec):
    value: int

    def create(self) -> int:
        return RunLeaf(value=self.value).create() + 1


class GpuLeaf(Spec):
    value: int

    def runs_on(self, worker: Worker) -> bool:
        return worker.gpus >= 1

    def create(self) -> int:
        return self.value


class CpuOnlyLeaf(Spec):
    value: int

    def create(self) -> int:
        return self.value


class MemoryLeaf(Spec):
    value: int

    def runs_on(self, worker: Worker) -> bool:
        return worker.memory_gib >= 8

    def create(self) -> int:
        return self.value


class DynamicCpuSeed(Spec):
    value: int

    def create(self) -> int:
        return self.value


class DynamicGpuAfterSeed(Spec):
    parent: DynamicCpuSeed
    value: int

    def runs_on(self, worker: Worker) -> bool:
        return worker.gpus >= 1

    def create(self) -> int:
        return self.parent.create() + self.value


class DynamicCpuAfterGpu(Spec):
    parent: DynamicGpuAfterSeed
    value: int

    def create(self) -> int:
        return self.parent.create() + self.value


class DynamicGpuAfterCpu(Spec):
    parent: DynamicCpuAfterGpu
    value: int

    def runs_on(self, worker: Worker) -> bool:
        return worker.gpus >= 1

    def create(self) -> int:
        return self.parent.create() + self.value


class SharesLeaves(Spec):
    name: str
    leaves: tuple[RunLeaf, ...]

    def create(self) -> int:
        return len(self.leaves)


class ReturnsLeaves(Spec):
    count: int

    @furu.dependency
    def leaves(self) -> tuple[RunLeaf, ...]:
        return tuple(RunLeaf(value=1000 + n) for n in range(self.count))

    def create(self) -> int:
        return len(self.leaves)


class SweepLeaf(furu.Spec):
    """A batch of these is as large as its worker has CPUs."""

    value: int

    @furu.batched(lambda _, worker: ("sweep", worker.cpus))
    def create(objs: list["SweepLeaf"], directory: furu.SpecDirectory) -> list[int]:
        return [obj.value for obj in objs]


TINY, SMALL, LARGE = (_pool(Worker(cpus=cpus)) for cpus in (2, 5, 20))


# --- the script's run ----------------------------------------------------------


def test_a_submit_holds_each_line_once_and_what_each_group_shares() -> None:
    leaves = tuple(RunLeaf(value=value) for value in range(30))
    sharing = [SharesLeaves(name=name, leaves=leaves) for name in "ab"]
    returning = ReturnsLeaves(count=2)
    limited = LimitedRunLeaf(value=1)
    batched = [BatchedRunLeaf(value=value) for value in range(2)]
    cpu = CPU_POOL.pool_key

    run, submit = _new_run(
        [*sharing, returning, limited, *batched], backends=[CPU_POOL]
    )

    assert submit.launch == _launch()
    # The shared tuple and what the method returned are a line each.
    assert len(submit.lines) == len(set(submit.lines)) == 32 + 6 + 2
    groups = {group.class_dir: group for group in submit.groups}
    assert len(groups) == len(submit.groups) == 5
    assert groups[sharing[0]._class_dir].specs == _ids(sharing)
    alone = groups[limited._class_dir]
    assert alone.pools == {cpu: Batch(batch=limited._stored_name, cap=1)}
    assert alone.throttle == JobThrottle(key=limited._stored_name, max_running=2)
    together = groups[batched[0]._class_dir]
    assert together.specs == _ids(batched)
    assert together.pools[cpu].cap == 10 and together.throttle is None
    assert run.left == {spec for group in submit.groups for spec in group.specs}


def test_two_metadata_under_one_batch_key_stop_the_run_before_anything_is_sent() -> (
    None
):
    with pytest.raises(ValueError, match="differ in Metadata reuse"):
        _new_run([MixedMetadataLeaf(value=1), MixedMetadataLeaf(value=2)])


def test_a_failed_attempt_reruns_until_its_retries_run_out(
    caplog: pytest.LogCaptureFixture,
) -> None:
    leaf = RunLeaf(value=1)
    ids, label = [leaf.object_id], leaf.object_id
    run, _ = _new_run([leaf], max_retries_per_object=3)
    errors = [
        "Traceback (most recent call last):\n  File a\nValueError: bad shape\n",
        "Traceback (most recent call last):\n  File b\nValueError: bad shape\n",
        "Traceback (most recent call last):\nOSError: disk full\n\n",
        "subprocess died: killed by signal 9 (SIGKILL)",
    ]

    with _captured_furu_logs(caplog):
        for error in errors[:3]:
            assert run.result(_result([leaf], JobFailedResult(error=error))) == [
                RerunMessage(specs=ids)
            ]
        gave_up = _gave_up([leaf], errors[3], attempts=4)
        assert run.result(_result([leaf], JobFailedResult(error=errors[3]))) == [
            gave_up
        ]
        assert run.left == set(ids)
        run.gave_up(gave_up)
        assert not run.left
        with pytest.raises(RuntimeError, match="1 spec failed, 0 blocked"):
            run.finish()

    failed = [record for record in caplog.records if record.message.startswith("fail")]
    assert [record.message for record in failed] == [
        f"failed {label} · attempt 1/4 · ValueError: bad shape",
        f"failed {label} · attempt 2/4 · same error",
        f"failed {label} · attempt 3/4 · OSError: disk full",
        f"failed {label} · attempt 4/4 · subprocess died: killed by signal 9 (SIGKILL)",
    ]
    assert [record.levelno for record in failed] == [logging.WARNING] * 3 + [
        logging.ERROR
    ]
    assert failed[0].__dict__["path"] == attempt_parent([leaf])
    # The last attempt is never one that retries.
    assert [message for message in caplog.messages if "progress" in message] == [
        *["progress 0/1 · 1 retrying"] * 3,
        "progress 0/1 · 1 failed",
    ]
    assert all("Traceback" not in message for message in caplog.messages)
    assert caplog.messages[-1] == f"run failed · 1 spec failed · {label} · {errors[3]}"


def test_a_retry_that_completes_is_no_failure() -> None:
    leaf = RunLeaf(value=1)
    run, _ = _new_run([leaf], max_retries_per_object=1)

    run.result(_result([leaf], JobFailedResult(error="boom")))
    assert run.result(_result([leaf], JobCompletedResult())) == []
    # A result for a spec that has ended is ignored.
    assert run.result(_result([leaf], JobFailedResult(error="late"))) == []

    assert run.failures == {} and not run.left
    run.finish()


def test_the_runs_error_lists_each_spec_it_gave_up_on_and_each_that_waits_for_one() -> (
    None
):
    first, second = RunLeaf(value=1), RunLeaf(value=2)
    parent = RunParent(child=first)
    run, _ = _new_run([parent, second], max_retries_per_object=0)

    for obj in (first, second):
        error = f"Traceback\n  File x\nValueError: {obj.value}"
        (gave_up,) = run.result(_result([obj], JobFailedResult(error=error)))
        assert isinstance(gave_up, GaveUpMessage)
        # The manager passes it on with the specs that wait for these.
        blocked = [parent.object_id] if obj is first else []
        run.gave_up(gave_up.model_copy(update={"blocked": blocked}))

    with pytest.raises(RuntimeError) as excinfo:
        run.finish()
    assert str(excinfo.value).splitlines() == [
        "2 specs failed, 1 blocked",
        *(
            f"  {obj.object_id} · 1 attempt · ValueError: {obj.value} → "
            f"{_display_path(attempt_parent([obj]))}"
            for obj in (first, second)
        ),
        f"  {parent.object_id} · blocked",
    ]


def test_a_failed_batch_counts_for_each_spec_and_points_at_the_batchs_folder(
    caplog: pytest.LogCaptureFixture,
) -> None:
    objs = [BatchedRunLeaf(value=value) for value in range(2)]
    run, _ = _new_run(objs, max_retries_per_object=1)
    folder = attempt_parent(objs)

    assert run.result(_result(objs, JobFailedResult(error="boom"))) == [
        RerunMessage(specs=_ids(objs))
    ]
    (gave_up,) = run.result(_result(objs, JobFailedResult(error="boom")))
    run.gave_up(cast(GaveUpMessage, gave_up))
    with pytest.raises(RuntimeError) as excinfo:
        run.finish()
    # Another launch that asked for them hears the same from the manager.
    other, _ = _new_run(objs)
    other.gave_up(cast(GaveUpMessage, gave_up))
    with pytest.raises(RuntimeError) as heard:
        other.finish()
    assert str(heard.value) == str(excinfo.value)

    assert [
        record.__dict__["path"]
        for record in caplog.records
        if record.message.startswith("failed ")
    ] == [folder] * 6
    assert all(
        line.endswith(f"→ {_display_path(folder)}")
        for line in str(excinfo.value).splitlines()[1:]
    )


def test_a_blocked_attempt_asks_for_what_it_found_missing_and_reruns_after_it() -> None:
    lazy, leaf = RunLazyParent(value=2), RunLeaf(value=uuid4().int)
    found = RunParent(child=leaf)
    run, _ = _new_run([lazy, leaf])
    blocked = _blocked(found)

    asked, rerun = run.result(_result([lazy], blocked))

    assert isinstance(asked, SubmitMessage)
    assert asked.launch == _launch().model_copy(update={"plan": blocked.plan})
    assert asked.lines == blocked.lines
    # The leaf is listed, so that the found spec waits for it, and not new.
    assert [group.specs for group in asked.groups] == [[found.object_id], _ids([leaf])]
    assert rerun == RerunMessage(specs=_ids([lazy]), after=_ids([found]))
    assert run.left == set(_ids([lazy, leaf, found]))


@pytest.mark.parametrize("built_here", [True, False])
def test_a_found_spec_does_not_wait_for_a_built_spec(built_here: bool) -> None:
    lazy, leaf = RunLazyParent(value=uuid4().int), RunLeaf(value=uuid4().int)
    found = RunParent(child=leaf)
    run, _ = _new_run([lazy, leaf])
    blocked = _blocked(found)
    if built_here:
        run.result(_result([leaf], JobCompletedResult()))
    else:
        leaf.create()  # another run builds it

    asked, _ = run.result(_result([lazy], blocked))

    assert isinstance(asked, SubmitMessage)
    assert [group.specs for group in asked.groups] == [[found.object_id]]


def test_an_attempt_that_found_only_built_specs_missing_reruns_at_the_next_poll() -> (
    None
):
    lazy, dependency = RunLazyParent(value=2), RunLeaf(value=2)
    dependency.create()
    run, _ = _new_run([lazy])

    # Rerunning at once would likely miss the same specs again.
    assert run.result(_result([lazy], _blocked(dependency))) == []
    assert run.poll() == [RerunMessage(specs=_ids([lazy]))]
    assert run.poll() == []


@pytest.fixture
def dag_clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """A fake clock for the grace on dependencies a worker can't see yet."""
    now = [0.0]
    monkeypatch.setattr(dag_module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    return now


def _rerun(run: Run, lazy: Spec, result: JobResult) -> list[object]:
    assert run.poll() == [RerunMessage(specs=_ids([lazy]))]
    return list(run.result(_result([lazy], result)))


def test_a_spec_that_keeps_missing_a_built_dependency_is_given_up_on_after_the_grace(
    dag_clock: list[float],
) -> None:
    lazy, dependency = RunLazyParent(value=2), RunLeaf(value=2)
    dependency.create()
    run, _ = _new_run([lazy])
    run.result(_result([lazy], _blocked(dependency)))

    while dag_clock[0] < Run.unseen_grace_s:
        dag_clock[0] += 5
        assert _rerun(run, lazy, _blocked(dependency)) == []  # 120 s is still within it
    dag_clock[0] += 5
    (gave_up,) = _rerun(run, lazy, _blocked(dependency))

    assert gave_up == _gave_up(
        [lazy],
        f"still finds {dependency.object_id} missing for 2m05s, though this "
        "run sees it built; its worker may not see the storage it was written "
        "to, or may be running different code than this run",
    )


def test_the_grace_is_per_dependency(dag_clock: list[float]) -> None:
    lazy = RunLazyParent(value=2)
    first, second, fresh = (RunLeaf(value=value) for value in (2, 3, 4))
    furu.create([first, second])
    run, _ = _new_run([lazy])
    run.result(_result([lazy], _blocked(first)))

    # Missing another one in between does not start the first one's over.
    dag_clock[0] = 60
    assert _rerun(run, lazy, _blocked(second)) == []
    # One published only now has the whole grace.
    dag_clock[0] = Run.unseen_grace_s - 1
    fresh.create()
    assert _rerun(run, lazy, _blocked(fresh)) == []
    dag_clock[0] = Run.unseen_grace_s + 1
    assert _rerun(run, lazy, _blocked(fresh)) == []
    (gave_up,) = _rerun(run, lazy, _blocked(first))
    assert first.object_id in cast(GaveUpMessage, gave_up).error


@pytest.mark.parametrize("progress", ["found a missing spec", "failed"])
def test_the_grace_starts_over_once_the_spec_got_somewhere(
    dag_clock: list[float], progress: str
) -> None:
    lazy, unseen = RunLazyParent(value=2), RunLeaf(value=2)
    unseen.create()
    run, _ = _new_run([lazy], max_retries_per_object=1)
    run.result(_result([lazy], _blocked(unseen)))
    dag_clock[0] = 100
    if progress == "failed":
        (rerun,) = _rerun(run, lazy, JobFailedResult(error="boom"))
    else:
        _, rerun = _rerun(run, lazy, _blocked(RunLeaf(value=3)))
    assert isinstance(rerun, RerunMessage)

    dag_clock[0] = 600
    assert run.result(_result([lazy], _blocked(unseen))) == []
    dag_clock[0] = 600 + Run.unseen_grace_s + 1
    (gave_up,) = _rerun(run, lazy, _blocked(unseen))
    assert isinstance(gave_up, GaveUpMessage)


def test_a_spec_blocked_on_nothing_or_on_a_spec_the_run_gave_up_on_is_given_up_on() -> (
    None
):
    lazy, leaf = RunLazyParent(value=2), RunLeaf(value=2)
    run, _ = _new_run([lazy, leaf], max_retries_per_object=0)

    assert run.result(_result([lazy], _blocked())) == [
        _gave_up([lazy], "reported itself blocked without naming a missing dependency")
    ]
    (gave_up,) = run.result(_result([leaf], JobFailedResult(error="boom")))
    run.gave_up(cast(GaveUpMessage, gave_up))
    assert run.result(_result([lazy], _blocked(leaf))) == [
        _gave_up(
            [lazy], f"needs {leaf.object_id}, which this run gave up on", attempts=2
        )
    ]


@pytest.mark.parametrize(
    ("found", "match"),
    [
        (lambda: [MemoryLeaf(value=uuid4().int)], "no worker pool can run"),
        (lambda: [PerGpuBatchedLeaf(value=1)], "cap must be a positive int, got 0"),
        (
            lambda: [MixedMetadataLeaf(value=1), MixedMetadataLeaf(value=2)],
            "ValueError: .* differ in Metadata reuse",
        ),
    ],
)
def test_found_specs_that_cannot_be_placed_fail_the_run(
    found: Callable[[], list[Spec]], match: str
) -> None:
    lazy = RunLazyParent(value=uuid4().int)
    run, _ = _new_run([lazy], backends=[CPU_POOL])

    with pytest.raises(RuntimeError, match=match):
        run.result(_result([lazy], _blocked(*found())))


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

    first, second, third = (_leased(manager) for _ in range(3))
    assert first.specs + second.specs == _ids(limited[:2])
    assert third.specs == [uncapped.object_id]
    # Busy workers still count; none more can start.
    assert manager.demand(CPU_POOL.pool_key) == 3
    assert _lease(manager) is None

    _end(manager, first)
    assert _leased(manager).specs == [limited[2].object_id]


def test_specs_with_one_batch_key_are_one_job_up_to_the_cap() -> None:
    first_x = BatchedRunLeaf(value=1, group="x", cap=2)
    only_y = BatchedRunLeaf(value=2, group="y", cap=2)
    more_x = [BatchedRunLeaf(value=value, group="x", cap=2) for value in (3, 4)]
    manager = _manager([first_x, only_y, *more_x])

    assert manager.demand(CPU_POOL.pool_key) == 3
    lease = _leased(manager)
    assert lease.specs == _ids([first_x, more_x[0]])
    assert lease.job.folder == attempt_parent([first_x, more_x[0]])
    assert _leased(manager).specs == [more_x[1].object_id]
    assert _leased(manager).specs == [only_y.object_id]
    assert _lease(manager) is None


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

    assert len(_leased(manager, eight_gpus).specs) == 8
    assert len(_leased(manager, one_gpu).specs) == 1


def test_a_batch_holds_specs_from_one_plan_and_a_throttle_counts_batches(
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = Manager(url="ws://unused", manager_dir=None)
    scripts = [
        _submit(
            manager,
            [ThrottledBatchedRunLeaf(value=value) for value in values],
            plan_path=_plan(f"e{index}"),
        )[0]
        for index, values in enumerate(((0, 1), (2, 3), (4,)))
    ]

    with _captured_furu_logs(caplog):
        first, second = _leased(manager), _leased(manager)

    assert len(first.specs) == len(second.specs) == 2
    assert [first.job.plan, second.job.plan] == [_plan("e0"), _plan("e1")]
    # A worker could take three. It is said once for each pair of plans, to
    # the script and in the manager's own log.
    warned = manager.outbox
    assert [script for script, _ in warned] == [scripts[0], scripts[0], scripts[1]]
    reasons = [
        message.reason for _, message in warned if isinstance(message, WarningMessage)
    ]
    assert len(reasons) == 3 and all("smaller batch" in reason for reason in reasons)
    assert caplog.messages == reasons
    # The throttle counts the two running batches across all scripts.
    assert manager.demand(CPU_POOL.pool_key) == 2
    assert _lease(manager) is None
    _end(manager, first)
    assert len(_leased(manager).specs) == 1


def test_a_spec_goes_to_the_pools_that_can_run_it() -> None:
    cpu_leaf, gpu_leaf, memory_leaf = (
        CpuOnlyLeaf(value=1),
        GpuLeaf(value=2),
        MemoryLeaf(value=3),
    )
    gpu_pool = _pool(Worker(gpus=1))
    reserved = LocalThreadWorkerBackend(
        worker=Worker(memory_gib=16),
        accepts=lambda spec: isinstance(spec, MemoryLeaf),
    )
    manager = _manager(
        [cpu_leaf, gpu_leaf, memory_leaf], backends=(CPU_POOL, gpu_pool, reserved)
    )

    assert _leased(manager, reserved).specs == [memory_leaf.object_id]
    assert _lease(manager, reserved) is None
    assert _leased(manager, gpu_pool).specs == [cpu_leaf.object_id]
    assert _leased(manager, gpu_pool).specs == [gpu_leaf.object_id]
    assert _lease(manager, CPU_POOL) is None


def test_a_spec_waits_for_what_its_line_points_to_and_its_job_holds_what_its_fields_reach() -> (
    None
):
    leaves = tuple(RunLeaf(value=value) for value in range(30))
    sharing = [SharesLeaves(name=name, leaves=leaves) for name in "ab"]
    returning = ReturnsLeaves(count=2)
    manager = _manager([*sharing, returning])

    jobs = {}
    while (lease := _lease(manager)) is not None:
        (spec,) = lease.specs
        # Everything it waits for is finished by now.
        assert all(
            key not in manager.dag.nodes for key in plan.holds(lease.job.lines[0])
        )
        jobs[spec] = {plan.key(json.loads(line)) for line in lease.job.lines}
        _end(manager, lease)

    assert list(jobs)[32:] == _ids([*sharing, returning])
    # The shared tuple's line and the lines of the specs in it.
    assert len(jobs[sharing[0].object_id]) == 1 + 1 + 30
    assert jobs[sharing[0].object_id] > set(_ids(leaves))
    # What a dependency method returned is one fingerprint in the spec's line.
    assert jobs[returning.object_id] == {returning.object_id}


def test_a_spec_that_two_launches_ask_for_is_queued_once_under_the_first() -> None:
    leaf, other = RunLeaf(value=1), RunLeaf(value=2)
    manager = Manager(url="ws://unused", manager_dir=None)
    first, _ = _submit(manager, [leaf, other])
    second, _ = _submit(manager, [RunParent(child=leaf), other], plan_path=_plan("e2"))

    lease = _leased(manager)
    assert lease.specs == [leaf.object_id] and lease.job.plan == _plan()
    _end(manager, lease)
    assert [script for script, _ in manager.outbox] == [first, second]
    assert _sent(manager) == 2 * [
        _result([leaf], JobCompletedResult()).model_copy(update={"seconds": 0.0})
    ]

    # The first launch leaves: what only it asked for goes with it.
    manager._drop(first)
    running = [_leased(manager), _leased(manager)]
    assert sorted(lease.specs for lease in running) == sorted(
        [[other.object_id], [RunParent(child=leaf).object_id]]
    )
    # Nobody asks for the two that run any more. They finish all the same.
    manager._drop(second)
    assert not manager.placed
    for lease in running:
        _end(manager, lease)
    assert not manager.dag.nodes and _sent(manager) == []


def test_an_attempt_that_did_not_complete_waits_for_its_script_to_say_what_happens() -> (
    None
):
    leaf = RunLeaf(value=1)
    parent = RunParent(child=leaf)
    failed = JobFailedResult(error="boom")
    manager = Manager(url="ws://unused", manager_dir=None)
    script, _ = _submit(manager, [parent])

    _end(manager, _leased(manager), failed)
    assert _sent(manager) == [
        _result([leaf], failed).model_copy(update={"seconds": 0.0})
    ]
    assert _lease(manager) is None
    manager._rerun(script, RerunMessage(specs=[leaf.object_id]))
    _end(manager, _leased(manager), failed)
    gave_up = _gave_up([leaf], "boom")
    manager._gave_up(script, gave_up)
    assert _sent(manager)[1:] == [
        gave_up.model_copy(update={"blocked": [parent.object_id]})
    ]
    assert _lease(manager) is None and manager.demand(CPU_POOL.pool_key) == 0

    # Another launch that asks for the spec gives it a fresh go. What waited
    # for it does not run for the script that was told it would not.
    _submit(manager, [leaf], plan_path=_plan("e2"))
    _end(manager, _leased(manager))
    assert _lease(manager) is None and not manager.placed


def test_each_script_answers_for_its_own_specs_of_a_batch_that_did_not_complete() -> (
    None
):
    objs = [BatchedRunLeaf(value=value) for value in range(2)]
    manager = Manager(url="ws://unused", manager_dir=None)
    first, _ = _submit(manager, objs)
    others = [
        _submit(manager, [obj], plan_path=_plan(f"e{index}"))[0]
        for index, obj in enumerate(objs, start=2)
    ]
    manager._drop(first)

    _end(manager, _leased(manager), JobFailedResult(error="boom"))
    assert [script for script, _ in manager.outbox] == others
    manager._rerun(others[0], RerunMessage(specs=_ids(objs[:1])))
    assert _leased(manager).specs == _ids(objs[:1])
    # The other spec waits for the script that answers for it.
    assert _lease(manager) is None
    manager.outbox.clear()
    gave_up = _gave_up(objs[1:], "boom")
    manager._gave_up(others[1], gave_up)
    assert manager.outbox == [(others[1], gave_up)]


def test_a_blocked_attempt_reruns_once_what_it_found_is_built() -> None:
    lazy, found = RunLazyParent(value=2), RunLeaf(value=2)
    manager = Manager(url="ws://unused", manager_dir=None)
    script, run = _submit(manager, [lazy])
    blocked = _blocked(found)

    _end(manager, _leased(manager), blocked)
    _answer(manager, script, run)
    lease = _leased(manager)
    assert lease.specs == [found.object_id] and lease.job.plan == blocked.plan
    assert _lease(manager) is None
    _end(manager, lease)
    _answer(manager, script, run)
    again = _leased(manager)
    assert again.specs == [lazy.object_id] and again.job.plan == _plan()

    # It now finds a spec missing whose create() asks for it in turn.
    cycle = RunLazyParent(value=3)
    _end(manager, again, _blocked(cycle))
    _answer(manager, script, run)
    _end(manager, _leased(manager), _blocked(lazy))
    run.result(cast(ResultMessage, manager.outbox.pop()[1]))
    manager._rerun(script, RerunMessage(specs=_ids([cycle]), after=_ids([lazy])))
    assert _sent(manager) == [
        ErrorMessage(
            reason=f"dependency cycle: {cycle.object_id} asks for "
            f"{lazy.object_id}, which already waits for it"
        )
    ]


def test_a_batch_whose_create_asks_for_one_of_its_own_specs_fails_loudly() -> None:
    objs = {obj.object_id: obj for obj in (BatchedRunLeaf(value=v) for v in (0, 1))}
    manager = Manager(url="ws://unused", manager_dir=None)
    script, run = _submit(manager, list(objs.values()))
    lease = _leased(manager)
    asks, found = lease.specs

    _end(manager, lease, _blocked(objs[found]))
    _answer(manager, script, run)
    _answer(manager, script, run)

    assert run.failed == set(objs) and not run.left
    (why,) = run.failures[asks]
    assert f"asked for {found}, which is in the same batch" in why
    assert str(attempt_parent([objs[asks], objs[found]])) in why
    assert manager.gave_up == set(objs) and _sent(manager) == []


def _stopped_sweep(count: int) -> tuple[Manager, list[str]]:
    """A launch of ``count`` specs whose first 5 were a batch that stopped."""
    specs = [SweepLeaf(value=value) for value in range(1, count + 1)]
    manager = Manager(url="ws://unused", manager_dir=None)
    script, _ = _submit(manager, specs, backends=(TINY, SMALL, LARGE))
    lease = _leased(manager, SMALL)
    _end(manager, lease, JobFailedResult(error="worker w0 disconnected"))
    manager._rerun(script, RerunMessage(specs=lease.specs))
    manager.outbox.clear()
    return manager, _ids(specs)


def _old_folder(ids: Sequence[str]) -> str:
    """The working folder of the batch that stopped: specs 1 to 5."""
    name = run_name([spec.rpartition(":")[2] for spec in ids[:5]])
    return str(attempt_parent_in(SweepLeaf(value=1)._class_dir, name))


@pytest.mark.parametrize(
    ("count", "pool", "handed", "warns", "then"),
    [
        # 25 are unstarted: a worker of 20 gets 6 to 25, and the stopped batch
        # stays whole.
        (30, LARGE, range(6, 26), False, range(26, 31)),
        # Only 3 are: it gets those and the stopped 5 as one batch of 8.
        (8, LARGE, [6, 7, 8, 1, 2, 3, 4, 5], True, []),
        # 18 are: it gets those and specs 1 and 2, and 3 to 5 are unstarted.
        (23, LARGE, [*range(6, 24), 1, 2], True, [3, 4, 5]),
        # Alone it goes out as it was, and continues in its folder.
        (5, LARGE, range(1, 6), False, []),
        # Too large for the worker that asks, it is cut.
        (5, TINY, [1, 2], True, [3, 4, 5]),
    ],
)
def test_a_worker_is_filled_with_unstarted_specs_then_stopped_batches(
    count: int,
    pool: LocalThreadWorkerBackend,
    handed: Sequence[int],
    warns: bool,
    then: Sequence[int],
) -> None:
    manager, ids = _stopped_sweep(count)

    lease = _leased(manager, pool)

    assert lease.specs == [ids[number - 1] for number in handed]
    warnings = _sent(manager)
    assert len(warnings) == warns
    for warning in warnings:
        assert isinstance(warning, WarningMessage)
        assert warning.reason.endswith(f"stays in attempts/: {_old_folder(ids)}")
    # What a worker of 5 gets next: unstarted specs.
    after = _lease(manager, SMALL)
    assert (after.specs if after else []) == [ids[number - 1] for number in then]
    assert _sent(manager) == []


def test_a_new_manager_takes_up_the_working_folders_on_disk_once_they_are_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(manager_module, "_HELD_CHECK_INTERVAL_S", 0.0)
    specs = [SweepLeaf(value=value) for value in range(4)]
    # A batch that a worker of another manager opened, in its order.
    started = [specs[2], specs[0]]
    folder = attempt_parent(started) / "local__node7__4242__9f3c2a7d51b0e6a4"
    folder.mkdir(parents=True)
    events.start(folder, [spec._artifact_hash for spec in started], _plan(), _plan())

    manager = _manager(specs, backends=[SMALL])
    assert _leased(manager, SMALL).specs == _ids([specs[1], specs[3]])
    # The folder is its worker's, which may be alive: nothing to hand out,
    # and the script hears why once, however often a worker asks.
    assert _lease(manager, SMALL) is None
    assert _lease(manager, SMALL) is None
    (warning,) = _sent(manager)
    assert isinstance(warning, WarningMessage)
    assert warning.reason == (
        f"{_group_label(_ids(started)[0], 2)} waits: {folder} is the working "
        "folder of process 4242 on node7, and furu does not ask whether a process "
        "outside Slurm is alive. If nothing is writing to it, free it by hand: "
        f"mv {folder} {folder.parent / 'free__manual'}"
    )

    # Another's now: the script hears the new reason, once.
    folder = folder.rename(folder.with_name("local__node8__77__0b1c2d3e4f5a6b7c"))
    assert _lease(manager, SMALL) is None
    assert _lease(manager, SMALL) is None
    (warning,) = _sent(manager)
    assert isinstance(warning, WarningMessage)
    assert f"{folder} is the working folder of process 77 on node8" in warning.reason

    folder.rename(folder.with_name("free__manual"))
    assert _leased(manager, SMALL).specs == _ids(started)
    assert _sent(manager) == []


def test_a_working_folder_is_looked_at_again_only_after_half_a_minute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 1000.0
    monkeypatch.setattr(manager_module.time, "monotonic", lambda: now)
    takeable, looked = attempt_lock.takeable, []

    def look(run: Path) -> Path | None:
        looked.append(run)
        return takeable(run)

    monkeypatch.setattr(attempt_lock, "takeable", look)
    started = [SweepLeaf(value=0), SweepLeaf(value=1)]
    folder = attempt_parent(started) / "local__node7__4242__9f3c2a7d51b0e6a4"
    folder.mkdir(parents=True)
    events.start(folder, [spec._artifact_hash for spec in started], _plan(), _plan())
    manager = _manager(started, backends=[SMALL])

    assert _lease(manager, SMALL) is None
    folder.rename(folder.with_name("free__manual"))
    # However often workers ask: what the look found stands until the next.
    now += 29.9
    assert _lease(manager, SMALL) is None
    assert _lease(manager, LARGE) is None
    assert looked == [folder.parent]

    now += 0.2
    assert _leased(manager, SMALL).specs == _ids(started)
    assert looked == [folder.parent] * 2
    assert len(_sent(manager)) == 1


def test_slurm_is_asked_about_a_jobs_working_folder_without_the_managers_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = [SweepLeaf(value=0), SweepLeaf(value=1)]
    folder = attempt_parent(started) / "slurm__sc__123__9f3c2a7d51b0e6a4"
    folder.mkdir(parents=True)
    events.start(folder, [spec._artifact_hash for spec in started], _plan(), _plan())
    manager = _manager(started, backends=[SMALL])
    ran: list[tuple[str, bool]] = []

    def lock_is_free() -> None:
        ran[-1] = (ran[-1][0], manager.lock.acquire(timeout=5.0))
        manager.lock.release()

    def answer(*argv: str) -> str:
        # A Slurm command can take a minute, and every worker and script
        # waits for the lock.
        ran.append((argv[0], False))
        other = threading.Thread(target=lock_is_free)
        other.start()
        other.join()
        return {"squeue": "", "sacct": "123|COMPLETED|None\n"}[argv[0]]

    monkeypatch.setattr(slurm, "cluster", lambda: "sc")
    monkeypatch.setattr(slurm, "run", answer)
    connection = cast(ServerConnection, type("Open", (), {"state": State.OPEN})())

    lease = manager._lease(connection, pool=SMALL.pool_key)

    assert lease is not None and lease.specs == _ids(started)
    assert ran == [("squeue", True), ("sacct", True)]


def test_a_slow_probe_does_not_starve_handing_out_a_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = [SweepLeaf(value=0)]
    extra = SweepLeaf(value=1)
    folder = attempt_parent(started) / "free__9f3c2a7d51b0e6a4"
    folder.mkdir(parents=True)
    events.start(folder, [started[0]._artifact_hash], _plan(), _plan())
    manager = _manager([*started, extra], backends=[SMALL])
    manager.scripts[0].connection = cast(
        ServerConnection, type("Sink", (), {"send": lambda self, _: None})()
    )

    now = [100.0]
    probes = 0
    monkeypatch.setattr(manager_module.time, "monotonic", lambda: now[0])

    def slow_held(path: Path) -> Path | None:
        nonlocal probes
        probes += 1
        now[0] += manager_module._HELD_CHECK_INTERVAL_S + 1.0
        return None

    monkeypatch.setattr(attempt_lock, "takeable", slow_held)
    connection = cast(ServerConnection, type("Open", (), {"state": State.OPEN})())

    lease = manager._lease(connection, pool=SMALL.pool_key)

    assert lease is not None
    assert set(lease.specs) == set(_ids([*started, extra]))
    assert probes == 1


def test_a_working_folder_on_disk_whose_specs_wait_for_one_another_is_not_resumed() -> (
    None
):
    first = ChainedBatchedLeaf(value=0)
    then = ChainedBatchedLeaf(value=1, previous=first)
    folder = attempt_parent([first, then]) / "free__9f3c2a7d51b0e6a4"
    folder.mkdir(parents=True)
    events.start(folder, [first._artifact_hash, then._artifact_hash], _plan(), _plan())

    manager = _manager([then])

    assert not manager.stopped
    lease = _leased(manager)
    assert lease.specs == [first.object_id]
    _end(manager, lease)
    assert _leased(manager).specs == [then.object_id]


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
    """Connect as a script whose specs only the raw "cpu" pool runs."""
    _, submit = _new_run(objs, backends=(CPU_POOL,))
    alone = {"cpu": Batch(batch="alone", cap=1)}
    submit = submit.model_copy(
        update={
            "pools": {"cpu": LocalPool(max_workers=0)},
            "groups": [
                group.model_copy(update={"pools": alone}) for group in submit.groups
            ],
        }
    )
    connection = connect(manager.url, legacy=True)
    connection.send(submit.model_dump_json())
    # Workers of a pool the manager has not started yet are turned away.
    deadline = time.monotonic() + 5
    while len(manager.placed) < len(objs) and time.monotonic() < deadline:
        time.sleep(0.01)
    return connection


def _received(script: ClientConnection) -> ResultMessage:
    message = manager_message_adapter.validate_json(script.recv(timeout=5))
    assert isinstance(message, ResultMessage)
    return message.model_copy(update={"seconds": 1.5})


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
        assert job.specs == [leaf.object_id]
        assert job.lines == [
            plan.dump({"fields": {"value": 1}, "runs": True, "spec": leaf.object_id})
        ]
        assert job.plan == _plan()

        worker.send(JobCompletedResult().model_dump_json())
        assert _received(script) == _result([leaf], JobCompletedResult())


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

        assert _received(script) == _result(
            [leaf], JobFailedResult(error="worker doomed-worker disconnected")
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

    assert all(obj.done for obj in objs)


def test_load_existing_in_worker_does_not_build_missing_spec() -> None:
    parent = OptionalLoadParent(name="optional")

    assert furu.create(parent, on=[LocalThreadWorkerBackend()]) == "missing"
    assert not OptionalLoadLeaf(name="optional").done


def test_top_level_create_runs_dag_on_worker_backends() -> None:
    leaf = RunLeaf(value=21)

    assert furu.create(leaf, on=(LocalThreadWorkerBackend(),)) == 21
    assert leaf.done
    assert furu.create([leaf], on=(LocalThreadWorkerBackend(),)) == [21]


def test_run_fails_fast_when_no_pool_can_run_a_job() -> None:
    with pytest.raises(RuntimeError, match=r"no worker pool can run.*workers: Worker"):
        furu.create(
            MemoryLeaf(value=uuid4().int),
            on=[LocalThreadWorkerBackend(worker=Worker())],
        )
    # Nothing was launched: no plan was written.
    assert not get_config().directories.anchored().plans.exists()


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

    assert cpu_leaf.done
    assert memory_leaf.done


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
        assert obj.done
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
        line.endswith(f" leased {leaf.object_id} to w0 → {attempt_parent([leaf])}")
        for line in lines
    )
    assert any(" D w0    spawned child " in line for line in lines)
    assert any(f" completed {leaf.object_id} ok · " in line for line in lines)
    assert any(line.endswith(" progress 1/1") for line in lines)
    assert any(line.endswith(" run finished ok") for line in lines)
    log_text = "\n".join(lines)
    assert f"creating {leaf.object_id}" not in log_text
    assert "received" not in log_text


def test_run_returns_when_all_objects_are_already_completed() -> None:
    leaf = RunLeaf(value=15)
    leaf.create()
    executions_dir = get_config().directories.anchored().executions

    run_on_workers([leaf], [LocalThreadWorkerBackend()])

    # A no-op run returns before creating an executor dir or capturing where
    # it was launched from; nothing appears under executions/.
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
        {"status": "blocked", "dependencies": [], "lines": [], "plan": _plan()}
    ) == JobBlockedResult(dependencies=[], lines=[], plan=_plan())
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
            idle_timeout=60.0,
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
            idle_timeout=60.0,
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
        (logging.INFO, f"running {leaf.object_id} ×2", job.folder),
        (logging.INFO, "manager closed the connection; exiting", None),
    ]


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
                idle_timeout=60.0,
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
    assert not leaf.done


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

    assert leaf.done
    assert (
        f"lost the manager; finishing {leaf.object_id}, then exiting" in caplog.messages
    )
    assert "manager closed the connection; exiting" in caplog.messages
