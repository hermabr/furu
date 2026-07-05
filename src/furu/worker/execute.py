from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import traceback
from collections import deque
from dataclasses import dataclass
from typing import Any, assert_never

from pydantic import TypeAdapter

from furu.config import _WORKER_JSON_CONFIG_FILE_ENV_VAR
from furu.core import Spec
from furu.execution.load_or_create import _ensure_single_result
from furu.logging import get_logger
from furu.metadata import ArtifactSpec
from furu.migration.links import result_dir_for_loading
from furu.spec_metadata import Subprocess
from furu.worker.context import _DependencyNotReady, worker_execution_context
from furu.worker.protocol import (
    Job,
    JobBlockedResult,
    JobCompletedResult,
    JobFailedResult,
    JobResultRequest,
)

logger = get_logger("worker.execute")

_STDERR_TAIL_LINES = 200
_STDERR_TAIL_CHARS = 32 * 1024
_RETIRE_TIMEOUT_SECONDS = 5.0

_job_result_adapter: TypeAdapter[JobResultRequest] = TypeAdapter(JobResultRequest)


def execute_job(obj: Spec[Any], *, lease_id: str) -> JobResultRequest:
    try:
        with worker_execution_context(lease_id=lease_id):
            _ensure_single_result(obj)
        return JobCompletedResult()
    except _DependencyNotReady as exc:
        return JobBlockedResult(
            dependencies=[ArtifactSpec.from_furu(dep) for dep in exc.dependencies]
        )
    except Exception as exc:
        return JobFailedResult(
            error="".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
        )


@dataclass(slots=True)
class _Child:
    process: subprocess.Popen[str]
    environment: dict[str, str]
    spec_name: str
    stderr_thread: threading.Thread
    stderr_tail: deque[str]


class ChildSlot:
    """At most one warm child process, tagged with what it last ran."""

    _child: _Child | None

    def __init__(self) -> None:
        self._child = None

    def run(
        self, obj: Spec[Any], *, job: Job, execution: Subprocess
    ) -> JobResultRequest:
        if result_dir_for_loading(obj) is not None:
            obj.logger.info("cache hit for %s", obj._log_label)
            return JobCompletedResult()

        environment = dict(os.environ)
        for name, value in execution.environment.items():
            if value is None:
                environment.pop(name, None)
            else:
                environment[name] = value
        # Re-pin last so an override cannot sever the furu config plumbing.
        if (config_file := os.environ.get(_WORKER_JSON_CONFIG_FILE_ENV_VAR)) is not None:
            environment[_WORKER_JSON_CONFIG_FILE_ENV_VAR] = config_file

        if missing := [
            name for name in execution.required_environment if name not in environment
        ]:
            raise RuntimeError(
                f"required environment variables not set: {', '.join(missing)}"
            )

        child = self._child
        if child is not None:
            same_process_context = (
                child.process.poll() is None and child.environment == environment
            )
            match execution.reuse:
                case "never":
                    can_reuse = False
                case "same_environment":
                    can_reuse = same_process_context
                case "same_environment_same_spec":
                    can_reuse = (
                        same_process_context
                        and child.spec_name == obj._fully_qualified_name
                    )
                case unreachable:
                    assert_never(unreachable)
            if not can_reuse:
                self.close()
                child = None
        if child is None:
            child = self._child = _spawn(environment)
        child.spec_name = obj._fully_qualified_name

        result = _request(child, job)
        if execution.reuse == "never" or child.process.poll() is not None:
            self.close()
        return result

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
        child.stderr_thread.join(timeout=_RETIRE_TIMEOUT_SECONDS)
        logger.debug("retired child %d", child.process.pid)


def _spawn(environment: dict[str, str]) -> _Child:
    process = subprocess.Popen(
        [sys.executable, "-m", "furu.worker._child"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        text=True,
    )
    stderr_tail: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)

    def forward_stderr() -> None:
        assert process.stderr is not None
        for line in process.stderr:
            stderr_tail.append(line)
            logger.info("child %d: %s", process.pid, line.rstrip("\n"))

    stderr_thread = threading.Thread(
        target=forward_stderr,
        name=f"furu-child-stderr-{process.pid}",
        daemon=True,
    )
    stderr_thread.start()
    logger.debug("spawned child %d", process.pid)
    return _Child(
        process=process,
        environment=environment,
        spec_name="",
        stderr_thread=stderr_thread,
        stderr_tail=stderr_tail,
    )


def _request(child: _Child, job: Job) -> JobResultRequest:
    assert child.process.stdin is not None
    assert child.process.stdout is not None
    try:
        child.process.stdin.write(job.model_dump_json() + "\n")
        child.process.stdin.flush()
        line = child.process.stdout.readline()
    except OSError:
        line = ""
    if line:
        return _job_result_adapter.validate_json(line)

    returncode = child.process.wait()
    child.stderr_thread.join(timeout=_RETIRE_TIMEOUT_SECONDS)
    if returncode < 0:
        try:
            reason = f"signal {-returncode} ({signal.Signals(-returncode).name})"
        except ValueError:
            reason = f"signal {-returncode}"
    else:
        reason = f"exit code {returncode}"
    logger.warning("child %d died with %s", child.process.pid, reason)
    error = f"subprocess died: {reason}"
    if tail := "".join(child.stderr_tail)[-_STDERR_TAIL_CHARS:]:
        error += f"\nstderr tail:\n{tail}"
    return JobFailedResult(error=error)
