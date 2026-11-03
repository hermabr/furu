from __future__ import annotations

import os
import signal
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

from furu.code_trace import _REPO_ROOT_ENV_VAR
from furu.config import _Config
from furu.logging import _append, get_logger
from furu.snapshot import CodeLocation
from furu.worker.protocol import Job, JobFailedResult, JobResult, job_result_adapter

_RUN_LOG_TAIL_BYTES = 32 * 1024
_RETIRE_TIMEOUT_SECONDS = 5.0


@dataclass(slots=True)
class _Child:
    process: subprocess.Popen[str]
    environment: dict[str, str]
    code: CodeLocation
    config: _Config
    spec_name: str


class ChildSlot:
    """At most one warm child process, tagged with what it last ran.

    Each child runs in the job's code (its snapshot, or the live worktree for
    local threads) with the job's settings, so one worker can serve any repo.
    """

    _child: _Child | None

    def __init__(
        self, *, worker: str, backend: str, worker_log: Path | None = None
    ) -> None:
        self._worker = worker
        self._worker_log = worker_log
        self._logger = get_logger(worker)
        self._backend = backend
        self._child = None

    def run(self, job: Job, *, cancelled: threading.Event) -> JobResult:
        result = self._run(job, cancelled=cancelled)
        if isinstance(result, JobFailedResult) and result.stale_code:
            # The warm child imported code that has since changed on disk.
            self.close()
            result = self._run(job, cancelled=cancelled)
        return result

    def _run(self, job: Job, *, cancelled: threading.Event) -> JobResult:
        settings = job.process
        environment = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        # Progress bars redraw up to 10 times a second, and every redraw lands
        # in run.log; tqdm >= 4.66 reads this unless you set your own.
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

        spec_name = job.artifacts[0].fully_qualified_name
        child = self._child
        if child is not None:
            same_process_context = (
                child.process.poll() is None
                and child.environment == environment
                and child.code == job.code
                and child.config == job.config
            )
            match settings.reuse:
                case "never":
                    can_reuse = False
                case "same_environment":
                    can_reuse = same_process_context
                case "same_environment_same_spec":
                    can_reuse = same_process_context and child.spec_name == spec_name
                case unreachable:
                    assert_never(unreachable)
            if not can_reuse:
                self.close()
                child = None
        if child is None:
            child = self._child = self._spawn(environment, job=job)
        child.spec_name = spec_name

        if cancelled.is_set():
            child.process.kill()
        result = self._request(child, job)
        if (
            settings.reuse == "never"
            or child.process.poll() is not None
            or (isinstance(result, JobFailedResult) and result.stale_code)
        ):
            self.close()
        return result

    def kill(self) -> None:
        if (child := self._child) is not None:
            child.process.kill()

    def close(self) -> None:
        child, self._child = self._child, None
        if child is None:
            return
        if child.process.stdin is not None:
            try:
                child.process.stdin.close()
            except OSError:
                pass
        try:
            child.process.wait(timeout=_RETIRE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            child.process.kill()
            child.process.wait()
        self._logger.debug("retired child %d", child.process.pid)

    def _spawn(self, environment: dict[str, str], *, job: Job) -> _Child:
        # The child's stderr is ours: each job points it at its run.log, and
        # anything written between jobs lands in this worker's own output.
        process = subprocess.Popen(
            [str(job.code.python), "-m", "furu.worker._child"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            cwd=job.code.cwd,
            env=environment | {_REPO_ROOT_ENV_VAR: str(job.code.repo_root)},
            text=True,
        )
        assert process.stdin is not None
        process.stdin.write(job.config.model_dump_json() + "\n")
        process.stdin.write(self._backend + "\n")
        process.stdin.write(self._worker + "\n")
        process.stdin.write(f"{self._worker_log or ''}\n")
        process.stdin.flush()
        self._logger.debug("spawned child %d", process.pid)
        return _Child(
            process=process,
            environment=environment,
            code=job.code,
            config=job.config,
            spec_name="",
        )

    def _request(self, child: _Child, job: Job) -> JobResult:
        assert child.process.stdin is not None
        assert child.process.stdout is not None
        try:
            child.process.stdin.write(job.model_dump_json() + "\n")
            child.process.stdin.flush()
            line = child.process.stdout.readline()
        except OSError:
            line = ""
        if line:
            return job_result_adapter.validate_json(line)

        returncode = child.process.wait()
        if returncode < 0:
            try:
                name = signal.Signals(-returncode).name
                reason = f"killed by signal {-returncode} ({name})"
            except ValueError:
                reason = f"killed by signal {-returncode}"
        else:
            reason = f"exited with code {returncode}"
        self._logger.warning(
            "child %d %s while running %s",
            child.process.pid,
            reason,
            job.artifacts[0].log_label,
        )
        # The child never wrote its footer; the run.log tail is its last words,
        # and the reason goes last so it reads as the error's summary.
        tail = _tail(job.run_logs[0])
        for run_log in job.run_logs:
            _append(run_log, f"── died · {reason}\n\n")
        return JobFailedResult(error=f"{tail}subprocess died: {reason}")


def _tail(path: Path) -> str:
    try:
        with path.open("rb") as file:
            file.seek(max(0, file.seek(0, os.SEEK_END) - _RUN_LOG_TAIL_BYTES))
            tail = file.read().decode(errors="replace")
    except OSError:
        return ""
    return tail if not tail or tail.endswith("\n") else tail + "\n"
