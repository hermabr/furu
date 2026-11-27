from __future__ import annotations

import glob
import json
import os
import re
import secrets
import signal
import subprocess
import threading
from collections.abc import Generator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

from furu.code_trace import _REPO_ROOT_ENV_VAR
from furu.config import _Config
from furu.logging import get_logger
from furu.snapshot import CodeLocation
from furu.storage import attempt_lock
from furu.storage._layout import log_path_in
from furu.worker.messages import Job, JobFailedResult, JobResult, job_result_adapter

_LOG_TAIL_BYTES = 32 * 1024
_RETIRE_TIMEOUT_SECONDS = 5.0


@dataclass(eq=False, slots=True)
class _Child:
    process: subprocess.Popen[str]
    environment: dict[str, str]
    code: CodeLocation
    config: _Config
    spec_name: str
    # The allocation's slots it runs on, and the lease it runs, if any.
    slots: frozenset[int]
    lease: int | None


class Children:
    """Warm child processes on an allocation's slots, each tagged with what it
    last ran.

    Each child runs in the job's code (its snapshot, or the live worktree for
    local threads) with the job's settings, so one worker can serve any repo.
    A child runs on slots of its own: with more than one slot, it sees only
    their GPUs and runs on their CPUs. An idle child stays warm until a job
    needs its slots, and is retired before that job's child starts, so the
    memory it held is free.
    """

    def __init__(
        self, *, worker: str, backend: str, log: Path | None = None, slots: int = 1
    ) -> None:
        self._logger = get_logger(worker)
        # What each child records about the worker it runs in; ``log`` is a
        # Slurm worker's output file.
        self._started_as = {"backend": backend, "output": log and str(log)}
        self._slots = slots
        self._shares = _shares(slots) if slots > 1 else None
        self._children: list[_Child] = []
        self._lock = threading.Lock()

    def run(self, job: Job, *, cancelled: threading.Event) -> JobResult:
        result = self._run(job, cancelled=cancelled)
        if isinstance(result, JobFailedResult) and result.stale_code:
            # The warm child imported code that has since changed on disk;
            # it was retired, so this runs in a fresh one.
            result = self._run(job, cancelled=cancelled)
        return result

    def _run(self, job: Job, *, cancelled: threading.Event) -> JobResult:
        settings = job.process
        environment = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        # Progress bars redraw up to 10 times a second, and every redraw lands
        # in the run's log; tqdm >= 4.66 reads this unless you set your own.
        environment.setdefault("TQDM_MININTERVAL", "30")
        for name, value in settings.environment.items():
            if value is None:
                environment.pop(name, None)
            else:
                environment[name] = value

        if missing := [
            name
            for name in settings.required_environment_variables
            if name not in environment
        ]:
            raise RuntimeError(
                f"required environment variables not set: {', '.join(missing)}"
            )

        with self._lock:
            child = self._take(job, environment)
        if cancelled.is_set():
            _kill(child.process)
        result = self._request(child, job)
        with self._lock:
            child.lease = None
            retire = (
                settings.reuse == "never"
                or child.process.poll() is not None
                or (isinstance(result, JobFailedResult) and result.stale_code)
            )
            if retire:
                self._children.remove(child)
        if retire:
            self._retire(child)
        return result

    def _take(self, job: Job, environment: dict[str, str]) -> _Child:
        """A child for ``job``: a warm one on as many idle slots that may run
        it, or a new one, on idle slots that no warm child holds if there
        are enough."""
        spec_name = job.specs[0].partition(":")[0]
        for child in [each for each in self._children if each.lease is None]:
            same_process_context = (
                child.process.poll() is None
                and child.environment == environment
                and child.code == job.code
                and child.config == job.config
            )
            match job.process.reuse:
                case "never":
                    can_reuse = False
                case "same_environment":
                    can_reuse = same_process_context
                case "same_environment_same_spec":
                    can_reuse = same_process_context and child.spec_name == spec_name
                case unreachable:
                    assert_never(unreachable)
            if can_reuse and len(child.slots) == job.slots:
                child.lease, child.spec_name = job.lease, spec_name
                return child
        busy = {
            slot
            for each in self._children
            if each.lease is not None
            for slot in each.slots
        }
        warm = {
            slot: each
            for each in self._children
            if each.lease is None
            for slot in each.slots
        }
        idle = sorted(
            (slot for slot in range(self._slots) if slot not in busy),
            key=lambda slot: slot in warm,
        )
        if len(idle) < job.slots:
            raise RuntimeError(
                f"{job.label} takes {job.slots} slots and {len(idle)} of "
                f"{self._slots} are idle"
            )
        slots = frozenset(idle[: job.slots])
        for child in {warm[slot] for slot in slots if slot in warm}:
            self._children.remove(child)
            self._retire(child)
        child = self._spawn(environment, job=job, slots=slots)
        child.lease, child.spec_name = job.lease, spec_name
        self._children.append(child)
        return child

    def kill(self, lease: int) -> None:
        """Kill the child that runs ``lease``, if one does."""
        for child in list(self._children):
            if child.lease == lease:
                _kill(child.process)

    def close(self) -> None:
        with self._lock:
            children, self._children = self._children, []
        for child in children:
            self._retire(child)

    def _retire(self, child: _Child) -> None:
        if child.process.stdin is not None:
            try:
                child.process.stdin.close()
            except OSError:
                pass
        try:
            child.process.wait(timeout=_RETIRE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            _kill(child.process)
            child.process.wait()
        self._logger.debug("retired child %d", child.process.pid)

    def _spawn(
        self, environment: dict[str, str], *, job: Job, slots: frozenset[int]
    ) -> _Child:
        cpus: set[int] | None = None
        added = {_REPO_ROOT_ENV_VAR: str(job.code.repo_root)}
        if self._shares is not None:
            gpus, cpus = [], set()
            for slot in sorted(slots):
                gpus += self._shares[slot][0]
                cpus |= self._shares[slot][1]
            if any(share[0] for share in self._shares):
                added["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
        # The child's stderr is ours: a create() points it at its run's log,
        # and anything written outside one lands in this worker's own output.
        # Its own process group lets a kill reach what the job started.
        with _pinned(cpus):
            process = subprocess.Popen(
                [str(job.code.python), "-m", "furu.worker._child"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                cwd=job.code.cwd,
                env=environment | added,
                text=True,
                process_group=0,
            )
        assert process.stdin is not None
        process.stdin.write(job.config.model_dump_json() + "\n")
        process.stdin.write(json.dumps(self._started_as) + "\n")
        process.stdin.flush()
        self._logger.debug("spawned child %d on slots %s", process.pid, sorted(slots))
        return _Child(
            process=process,
            environment=environment,
            code=job.code,
            config=job.config,
            spec_name="",
            slots=slots,
            lease=None,
        )

    def _request(self, child: _Child, job: Job) -> JobResult:
        assert child.process.stdin is not None
        assert child.process.stdout is not None
        # The child ends the name of the working folder it takes with this,
        # so the folder is found again if the child dies holding it.
        token = secrets.token_hex(8)
        try:
            child.process.stdin.write(f"{token} {job.model_dump_json()}\n")
            child.process.stdin.flush()
            line = child.process.stdout.readline()
        except OSError:
            line = ""
        if line:
            return job_result_adapter.validate_json(line)

        # No answer: the child died or was killed. Nothing of it may be left
        # writing to the folder when the folder is given up, and only this
        # process knows that the child is dead.
        _kill(child.process)
        returncode = child.process.wait()
        attempt_lock.release_child(job.folder, token)
        if returncode < 0:
            try:
                name = signal.Signals(-returncode).name
                reason = f"killed by signal {-returncode} ({name})"
            except ValueError:
                reason = f"killed by signal {-returncode}"
        else:
            reason = f"exited with code {returncode}"
        self._logger.warning(
            "child %d %s while running %s", child.process.pid, reason, job.label
        )
        # The log's tail is the child's last words, and the reason goes last
        # so it reads as the error's summary.
        return JobFailedResult(error=f"{_tail(job.folder)}subprocess died: {reason}")


def _shares(slots: int) -> list[tuple[list[str], set[int]]]:
    """Each slot's even share of this process's GPUs and CPUs.

    The GPUs are the ones ``CUDA_VISIBLE_DEVICES`` names, or every one this
    machine has. The CPUs are cut whole cores at a time where the shares
    allow, so two slots do not share a core's hyperthreads.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpus = (
        [str(index) for index in range(len(glob.glob("/dev/nvidia[0-9]*")))]
        if visible is None
        else [device.strip() for device in visible.split(",") if device.strip()]
    )
    cpus = sorted(os.sched_getaffinity(0), key=lambda cpu: (_core(cpu), cpu))
    return [
        (
            gpus[slot * len(gpus) // slots : (slot + 1) * len(gpus) // slots],
            set(cpus[slot * len(cpus) // slots : (slot + 1) * len(cpus) // slots]),
        )
        for slot in range(slots)
    ]


def _core(cpu: int) -> int:
    """The first CPU of ``cpu``'s core: the same for all its hyperthreads."""
    siblings = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
    try:
        return int(re.split("[,-]", siblings.read_text())[0])
    except (OSError, ValueError):
        return cpu


@contextmanager
def _pinned(cpus: set[int] | None) -> Generator[None]:
    """Run on ``cpus`` alone, and so do the processes this thread starts.

    Only this thread: the CPUs Linux lets a thread run on are its own, and a
    process it starts begins with them. Set before the child starts, they
    hold for every thread the child and its children will have.
    """
    if cpus is None:
        yield
        return
    before = os.sched_getaffinity(0)
    os.sched_setaffinity(0, cpus)
    try:
        yield
    finally:
        os.sched_setaffinity(0, before)


def _kill(process: subprocess.Popen[str]) -> None:
    """SIGKILL the child's process group, e.g. its DataLoader workers.

    Helpers that outlive the child would keep its GPU memory and its end of the
    protocol pipe, so the worker would hang waiting for the child's reply.
    Processes in their own session escape, e.g. the ranks torchrun starts.
    """
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)


def _tail(folder: Path) -> str:
    """The end of the log in the working folder that ``folder`` holds now."""
    if (attempt := attempt_lock.current(folder)) is None:
        return ""
    try:
        with log_path_in(attempt).open("rb") as file:
            file.seek(max(0, file.seek(0, os.SEEK_END) - _LOG_TAIL_BYTES))
            tail = file.read().decode(errors="replace")
    except OSError:
        return ""
    return tail if not tail or tail.endswith("\n") else tail + "\n"
