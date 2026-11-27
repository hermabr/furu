from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

from furu.code_trace import _REPO_ROOT_ENV_VAR
from furu.config import _Config
from furu.logging import get_logger
from furu.resources import visible_gpus
from furu.snapshot import CodeLocation
from furu.storage import attempt_lock
from furu.storage._layout import log_path_in
from furu.worker.messages import Job, JobFailedResult, JobResult, job_result_adapter

# The CPUs a child binds itself to when it starts, separated by commas.
CPUS_ENV_VAR = "_FURU_CPUS"

_LOG_TAIL_BYTES = 32 * 1024
_RETIRE_TIMEOUT_SECONDS = 5.0


@dataclass(eq=False, slots=True)
class _Child:
    process: subprocess.Popen[str]
    environment: dict[str, str]
    code: CodeLocation
    config: _Config
    spec_name: str
    # Its share of the allocation, which it holds for as long as it lives.
    slots: int
    cpus: list[int]
    gpus: list[str]


class Children:
    """The child processes of one allocation, each warm and tagged with what it last ran.

    Each child runs in the job's code (its snapshot, or the live worktree for
    local threads) with the job's settings, so one allocation can serve any
    repo. Jobs run at once, each in its own child.

    A child whose job takes ``k`` of the allocation's ``slots`` holds
    ``k / slots`` of its CPUs and GPUs, rounded down, until it is retired:
    it sees only those GPUs and is bound to those CPUs. A job that takes the
    whole allocation gets it as it is. A job goes to an idle child that holds
    as many slots and may be reused for it. Any other job gets a new child,
    and idle children are retired first until there is room for it.
    """

    def __init__(
        self, *, worker: str, backend: str, slots: int = 1, log: Path | None = None
    ) -> None:
        self._logger = get_logger(worker)
        # What each child records about the worker it runs in; ``log`` is a
        # Slurm worker's output file.
        self._started_as = {"backend": backend, "output": log and str(log)}
        self._slots = slots
        # What the allocation has, in the order it is given out, and what of
        # it no child holds.
        self._cpus, self._gpus = _cpus(), visible_gpus()
        self._free_cpus, self._free_gpus = set(self._cpus), set(self._gpus)
        self._lock = threading.Lock()
        # The children without a job, least recently used first, and the
        # child of each running job.
        self._idle: list[_Child] = []
        self._running: dict[int, _Child] = {}

    def run(self, job: Job, *, cancelled: threading.Event) -> JobResult:
        result = self._run(job, cancelled=cancelled, warm=True)
        if isinstance(result, JobFailedResult) and result.stale_code:
            # The warm child imported code that has since changed on disk.
            result = self._run(job, cancelled=cancelled, warm=False)
        return result

    def _run(self, job: Job, *, cancelled: threading.Event, warm: bool) -> JobResult:
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

        spec_name = job.specs[0].partition(":")[0]
        with self._lock:
            child = self._take_idle(job, environment, spec_name) if warm else None
            if child is None:
                child = self._spawn(environment, job=job)
            child.spec_name = spec_name
            self._running[job.lease] = child

        if cancelled.is_set():
            _kill(child.process)
        result = self._request(child, job)
        with self._lock:
            del self._running[job.lease]
            if (
                settings.reuse == "never"
                or child.process.poll() is not None
                or (isinstance(result, JobFailedResult) and result.stale_code)
            ):
                self._retire([child])
            else:
                self._idle.append(child)
        return result

    def kill(self, lease: int) -> None:
        """Kill the child that runs the job ``lease`` names, if one does."""
        with self._lock:
            if (child := self._running.get(lease)) is not None:
                _kill(child.process)

    def close(self) -> None:
        with self._lock:
            self._retire([*self._idle, *self._running.values()])

    def _take_idle(
        self, job: Job, environment: dict[str, str], spec_name: str
    ) -> _Child | None:
        """The idle child last used that ``job`` may run in, which is idle no longer."""
        for child in reversed(self._idle):
            same_process_context = (
                child.slots == job.slots
                and child.process.poll() is None
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
            if can_reuse:
                self._idle.remove(child)
                return child
        return None

    def _retire(self, children: list[_Child]) -> None:
        """End ``children`` and wait until they are gone, so that what they
        held, the memory of their GPUs too, is free for the next child."""
        for child in children:
            if child in self._idle:
                self._idle.remove(child)
            if child.process.stdin is not None:
                with suppress(OSError):
                    child.process.stdin.close()
        deadline = time.monotonic() + _RETIRE_TIMEOUT_SECONDS
        for child in children:
            try:
                child.process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                _kill(child.process)
                child.process.wait()
            self._free_cpus.update(child.cpus)
            self._free_gpus.update(child.gpus)
            self._logger.debug("retired child %d", child.process.pid)

    def _spawn(self, environment: dict[str, str], *, job: Job) -> _Child:
        # Dead ones first, then the least recently used.
        idle = sorted(self._idle, key=lambda child: child.process.poll() is None)
        held = sum(c.slots for c in (*self._idle, *self._running.values()))
        in_the_way: list[_Child] = []
        while self._slots - held < job.slots:
            if not idle:
                raise RuntimeError(
                    f"no room for a job on {job.slots} of {self._slots} slots: "
                    "running jobs hold the rest of the allocation"
                )
            in_the_way.append(idle.pop(0))
            held -= in_the_way[-1].slots
        self._retire(in_the_way)
        # Shares round down, so free slots always have their share free.
        need_cpus = job.slots * len(self._cpus) // self._slots
        need_gpus = job.slots * len(self._gpus) // self._slots
        cpus = [cpu for cpu in self._cpus if cpu in self._free_cpus][:need_cpus]
        gpus = [gpu for gpu in self._gpus if gpu in self._free_gpus][:need_gpus]
        self._free_cpus.difference_update(cpus)
        self._free_gpus.difference_update(gpus)
        share = {}
        if job.slots < self._slots:
            if self._cpus:
                share[CPUS_ENV_VAR] = ",".join(map(str, cpus))
            if self._gpus:
                share["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)

        # The child's stderr is ours: a create() points it at its run's log,
        # and anything written outside one lands in this worker's own output.
        # Its own process group lets a kill reach what the job started.
        process = subprocess.Popen(
            [str(job.code.python), "-m", "furu.worker._child"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            cwd=job.code.cwd,
            env=environment | share | {_REPO_ROOT_ENV_VAR: str(job.code.repo_root)},
            text=True,
            process_group=0,
        )
        assert process.stdin is not None
        # A child that is dead already is found by its first job.
        with suppress(OSError):
            process.stdin.write(job.config.model_dump_json() + "\n")
            process.stdin.write(json.dumps(self._started_as) + "\n")
            process.stdin.flush()
        self._logger.debug("spawned child %d", process.pid)
        return _Child(
            process=process,
            environment=environment,
            code=job.code,
            config=job.config,
            spec_name="",
            slots=job.slots,
            cpus=cpus,
            gpus=gpus,
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
        except (OSError, ValueError):  # ValueError: its stdin was closed
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


def _cpus() -> list[int]:
    """The CPUs this process may run on, the threads of one core side by side,
    so that a share is whole cores where it can be."""
    if not hasattr(os, "sched_getaffinity"):
        return []

    def core(cpu: int) -> tuple[int, ...]:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            return tuple(
                int((topology / name).read_text())
                for name in ("physical_package_id", "core_id")
            )
        except (OSError, ValueError):
            return ()

    return sorted(os.sched_getaffinity(0), key=lambda cpu: (core(cpu), cpu))


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
