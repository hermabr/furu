from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import threading
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

from furu import slurm
from furu.code_trace import _REPO_ROOT_ENV_VAR
from furu.config import _Config
from furu.logging import get_logger
from furu.snapshot import CodeLocation
from furu.storage import attempt_lock
from furu.storage._layout import log_path_in
from furu.worker.messages import (
    Job,
    JobFailedResult,
    JobResult,
    ProcessSettings,
    job_result_adapter,
)

_LOG_TAIL_BYTES = 32 * 1024
_RETIRE_TIMEOUT_SECONDS = 5.0


@dataclass(eq=False, slots=True)
class _Child:
    environment: dict[str, str]
    code: CodeLocation
    config: _Config
    # How many of the worker's slots it holds: it runs only jobs of that many.
    slots: int
    spec_name: str
    # The lease of the job it runs, if it runs one.
    lease: int | None
    # None while it starts.
    process: subprocess.Popen[str] | None = None
    # Its Slurm step, "<job>.<step>", if it runs in its own.
    step: str | None = None


class Children:
    """Warm child processes that share a worker's ``slots``, each tagged with
    what it last ran.

    Each child runs in the job's code (its snapshot, or the live worktree for
    local threads) with the job's settings, so one worker can serve any repo.
    A child holds as many slots as the job it was started for. With ``steps``,
    the srun arguments of each slot count, each child runs in its own Slurm
    step, which holds only its slice of the allocation: Slurm keeps it to its
    GPUs, CPUs and memory, and killing the step ends all it started.
    """

    def __init__(
        self,
        *,
        worker: str,
        backend: str,
        log: Path | None = None,
        slots: int = 1,
        steps: dict[int, list[str]] | None = None,
    ) -> None:
        self._logger = get_logger(worker)
        # What each child records about the worker it runs in; ``log`` is a
        # Slurm worker's output file.
        self._started_as = {"backend": backend, "output": log and str(log)}
        self._slots = slots
        self._steps = steps or {}
        # Least recently used first.
        self._children: list[_Child] = []
        self._lock = threading.Lock()

    def run(self, job: Job, *, cancelled: threading.Event) -> JobResult:
        result = self._run(job, cancelled=cancelled)
        if isinstance(result, JobFailedResult) and result.stale_code:
            # The warm child imported code that has since changed on disk,
            # and was retired.
            result = self._run(job, cancelled=cancelled)
        return result

    def _run(self, job: Job, *, cancelled: threading.Event) -> JobResult:
        settings = job.process
        environment = _environment(settings)
        child, retired = self._take(job, environment)
        # Before a new child starts, so that its step gets their resources.
        for old in retired:
            self._retire(old)
        if child.process is None:
            try:
                self._spawn(child, job)
            except BaseException:
                with self._lock:
                    self._children.remove(child)
                raise
        result: JobResult | None = None
        try:
            if cancelled.is_set():
                self._kill(child)
            result = self._request(child, job)
            return result
        finally:
            assert child.process is not None
            with self._lock:
                child.lease = None
                done = (
                    result is None
                    or settings.reuse == "never"
                    or child.process.poll() is not None
                    or (isinstance(result, JobFailedResult) and result.stale_code)
                )
                if done:
                    self._children.remove(child)
            if done:
                self._retire(child)

    def _take(
        self, job: Job, environment: dict[str, str]
    ) -> tuple[_Child, list[_Child]]:
        """A warm child for ``job``, or one to start; and the idle ones to retire
        first, to make room for it."""
        spec_name = job.specs[0].partition(":")[0]
        with self._lock:
            idle = [c for c in self._children if c.lease is None]
            for child in idle:
                assert child.process is not None
                same = (
                    child.process.poll() is None
                    and child.slots == job.slots
                    and child.environment == environment
                    and child.code == job.code
                    and child.config == job.config
                )
                match job.process.reuse:
                    case "never":
                        reusable = False
                    case "same_environment":
                        reusable = same
                    case "same_environment_same_spec":
                        reusable = same and child.spec_name == spec_name
                    case unreachable:
                        assert_never(unreachable)
                if reusable:
                    self._children.remove(child)
                    self._children.append(child)
                    child.lease, child.spec_name = job.lease, spec_name
                    return child, []
            # The manager gave the job free slots, so retiring idle children
            # always makes room. Dead ones go anyway. In Slurm steps all idle
            # ones go: Slurm rounds a step up, e.g. to whole cores, so an idle
            # step could hold what the new one waits for, forever.
            live = sum(c.slots for c in self._children)
            retired = []
            for child in idle:
                assert child.process is not None
                if (
                    live + job.slots > self._slots
                    or self._steps
                    or child.process.poll() is not None
                ):
                    retired.append(child)
                    self._children.remove(child)
                    live -= child.slots
            child = _Child(
                environment=environment,
                code=job.code,
                config=job.config,
                slots=job.slots,
                spec_name=spec_name,
                lease=job.lease,
            )
            self._children.append(child)
            return child, retired

    def kill(self, lease: int) -> None:
        """Kill the child that runs ``lease``'s job, if it has started."""
        with self._lock:
            child = next((c for c in self._children if c.lease == lease), None)
        if child is not None and child.process is not None:
            self._kill(child)

    def close(self) -> None:
        with self._lock:
            children, self._children = self._children, []
        for child in children:
            self._retire(child)

    def _retire(self, child: _Child) -> None:
        """End a child and wait for it, and for its step, to be gone."""
        process = child.process
        if process is None:
            return
        if process.stdin is not None:
            with suppress(OSError):
                process.stdin.close()
        try:
            process.wait(timeout=_RETIRE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            self._kill(child)
            process.wait()
        self._logger.debug("retired child %d", process.pid)

    def _spawn(self, child: _Child, job: Job) -> None:
        # The child's stderr is ours: a create() points it at its run's log,
        # and anything written outside one lands in this worker's own output.
        # Its own process group lets a kill reach what the job started.
        argv = [str(job.code.python), "-m", "furu.worker._child"]
        if self._steps:
            argv = ["srun", *self._steps[job.slots], *argv]
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            cwd=job.code.cwd,
            env=child.environment | {_REPO_ROOT_ENV_VAR: str(job.code.repo_root)},
            text=True,
            process_group=0,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        with suppress(OSError):  # it died: the job's request finds out
            process.stdin.write(job.config.model_dump_json() + "\n")
            process.stdin.write(json.dumps(self._started_as) + "\n")
            process.stdin.flush()
        # Its first line says which Slurm step it runs in, which a kill ends.
        step = process.stdout.readline().strip()
        if self._steps and step:
            child.step = step
        child.process = process
        self._logger.debug(
            "spawned child %d%s", process.pid, f" in step {step}" if child.step else ""
        )

    def _kill(self, child: _Child) -> None:
        """SIGKILL the child and all it started, e.g. its DataLoader workers.

        Helpers that outlive the child would keep its GPU memory and its end
        of the protocol pipe, so the worker would hang waiting for its reply.
        A child in its own Slurm step is killed with the step, which reaches
        all of it. Otherwise its process group is: processes in their own
        session escape, e.g. the ranks torchrun starts.
        """
        process = child.process
        assert process is not None
        if child.step is not None:
            try:
                slurm.run("scancel", "--signal=KILL", child.step)
                return
            except (OSError, subprocess.SubprocessError) as exc:
                if process.poll() is not None:  # srun ends once its step has
                    return
                self._logger.warning(
                    "could not kill step %s, so killing its srun: %s", child.step, exc
                )
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)

    def _request(self, child: _Child, job: Job) -> JobResult:
        process = child.process
        assert process is not None
        assert process.stdin is not None
        assert process.stdout is not None
        # The child ends the name of the working folder it takes with this,
        # so the folder is found again if the child dies holding it.
        token = secrets.token_hex(8)
        try:
            process.stdin.write(f"{token} {job.model_dump_json()}\n")
            process.stdin.flush()
            line = process.stdout.readline()
        except OSError:
            line = ""
        if line:
            return job_result_adapter.validate_json(line)

        # No answer: the child died or was killed. Nothing of it may be left
        # writing to the folder when the folder is given up, and only this
        # process knows that the child is dead. srun exits once its step,
        # with everything in it, is gone.
        self._kill(child)
        returncode = process.wait()
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
            "child %d %s while running %s", process.pid, reason, job.label
        )
        # The log's tail is the child's last words, and the reason goes last
        # so it reads as the error's summary.
        return JobFailedResult(error=f"{_tail(job.folder)}subprocess died: {reason}")


def _environment(settings: ProcessSettings) -> dict[str, str]:
    """The environment a job's child runs in."""
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
    return environment


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
