from __future__ import annotations

import os
import signal
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

from furu.code_trace import _REPO_ROOT_ENV_VAR
from furu.config import get_config
from furu.logging import _append, get_logger
from furu.provenance import EnvironmentIdentity
from furu.snapshot import CodeLocation
from furu.worker.protocol import Job, JobFailedResult, JobResult, job_result_adapter

_RUN_LOG_TAIL_BYTES = 32 * 1024
_RETIRE_TIMEOUT_SECONDS = 5.0


@dataclass(slots=True)
class _Child:
    process: subprocess.Popen[str]
    environment: dict[str, str]
    code: CodeLocation
    spec_name: str


class ChildSlot:
    """At most one warm child process, tagged with what it last ran.

    A materializing slot (remote workers) spawns each child inside the job's
    code snapshot, so a job always runs the code it was submitted with; a
    non-materializing slot (local threads) runs children in the live worktree.
    """

    _child: _Child | None

    def __init__(
        self,
        *,
        worker: str,
        backend: str,
        materialize_snapshot: bool,
        worker_log: Path | None = None,
    ) -> None:
        self._worker = worker
        self._worker_log = worker_log
        self._logger = get_logger(worker)
        self._backend = backend
        self._materialize_snapshot = materialize_snapshot
        self._child = None

    def run(self, job: Job, *, cancelled: threading.Event) -> JobResult:
        result = self._run(job, cancelled=cancelled)
        if isinstance(result, JobFailedResult) and result.stale_code:
            # The warm child imported code that has since changed on disk.
            self.close()
            result = self._run(job, cancelled=cancelled)
        return result

    def _run(self, job: Job, *, cancelled: threading.Event) -> JobResult:
        if self._materialize_snapshot:
            code = CodeLocation.from_snapshot(job.provenance)
        else:
            code = CodeLocation.here()
            worker_hash = EnvironmentIdentity.capture().uv_lock_hash
            submitted_hash = job.provenance.environment.uv_lock_hash
            if worker_hash != submitted_hash:
                raise RuntimeError(
                    "worker uv.lock does not match the submitted environment\n"
                    f"  submitted : {submitted_hash}\n"
                    f"  worker    : {worker_hash}\n"
                    "The worker's project checkout is out of sync with the "
                    "submit host. Update the checkout (e.g. git pull) and run:\n"
                    "  uv sync"
                )

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
                and child.code == code
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
            child = self._child = self._spawn(environment, code=code)
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

    def _spawn(self, environment: dict[str, str], *, code: CodeLocation) -> _Child:
        # The child's stderr is ours: each job points it at its run.log, and
        # anything written between jobs lands in this worker's own output.
        child_environment = environment
        if code.repo_root is not None:  # traced code paths are relative to it
            child_environment = environment | {_REPO_ROOT_ENV_VAR: str(code.repo_root)}
        process = subprocess.Popen(
            [str(code.python), "-m", "furu.worker._child"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            cwd=code.cwd,
            env=child_environment,
            text=True,
        )
        assert process.stdin is not None
        process.stdin.write(get_config().model_dump_json() + "\n")
        process.stdin.write(self._backend + "\n")
        process.stdin.write(self._worker + "\n")
        process.stdin.write(f"{self._worker_log or ''}\n")
        process.stdin.flush()
        self._logger.debug("spawned child %d", process.pid)
        return _Child(process=process, environment=environment, code=code, spec_name="")

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
