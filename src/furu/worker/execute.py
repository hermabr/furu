from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

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
    environment: dict[str, str | None]
    spec_name: str
    stderr_thread: threading.Thread | None = None
    stderr_tail: deque[str] = field(default_factory=deque)
    stderr_tail_chars: int = 0


def _forward_stderr(child: _Child) -> None:
    assert child.process.stderr is not None
    for line in child.process.stderr:
        child.stderr_tail.append(line)
        child.stderr_tail_chars += len(line)
        while (
            child.stderr_tail_chars > _STDERR_TAIL_CHARS and len(child.stderr_tail) > 1
        ):
            child.stderr_tail_chars -= len(child.stderr_tail.popleft())
        logger.info("child %d: %s", child.process.pid, line.rstrip("\n"))


def _describe_exit(returncode: int) -> str:
    if returncode < 0:
        try:
            return f"signal {-returncode} ({signal.Signals(-returncode).name})"
        except ValueError:
            return f"signal {-returncode}"
    return f"exit code {returncode}"


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

        environment = dict(execution.environment)
        if missing := [
            name
            for name in execution.required_environment
            if environment.get(name, os.environ.get(name)) is None
        ]:
            raise RuntimeError(
                f"required environment variables not set: {', '.join(missing)}"
            )
        child = self._child
        if child is not None and not _reusable(
            child,
            environment=environment,
            spec_name=obj._fully_qualified_name,
            reuse=execution.reuse,
        ):
            self.close()
            child = None
        if child is None:
            child = _spawn(environment)
            self._child = child
        child.spec_name = obj._fully_qualified_name

        result = _request(child, job)
        if execution.reuse == "never" or child.process.poll() is not None:
            self.close()
        return result

    def close(self) -> None:
        child = self._child
        self._child = None
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
            child.process.terminate()
            try:
                child.process.wait(timeout=_RETIRE_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                child.process.kill()
                child.process.wait()
        if child.stderr_thread is not None:
            child.stderr_thread.join(timeout=_RETIRE_TIMEOUT_SECONDS)
        logger.debug("retired child %d", child.process.pid)


def _reusable(
    child: _Child,
    *,
    environment: dict[str, str | None],
    spec_name: str,
    reuse: Literal["never", "same_environment", "same_environment_same_spec"],
) -> bool:
    if reuse == "never":
        return False
    if child.process.poll() is not None:
        return False
    if child.environment != environment:
        return False
    return reuse == "same_environment" or child.spec_name == spec_name


def _spawn(environment: dict[str, str | None]) -> _Child:
    child_environment = dict(os.environ)
    for name, value in environment.items():
        if value is None:
            child_environment.pop(name, None)
        else:
            child_environment[name] = value
    # Re-pin last so an override cannot sever the furu config plumbing.
    if (config_file := os.environ.get(_WORKER_JSON_CONFIG_FILE_ENV_VAR)) is not None:
        child_environment[_WORKER_JSON_CONFIG_FILE_ENV_VAR] = config_file

    process = subprocess.Popen(
        [sys.executable, "-m", "furu.worker._child"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=child_environment,
        text=True,
    )
    child = _Child(process=process, environment=environment, spec_name="")
    child.stderr_thread = threading.Thread(
        target=_forward_stderr,
        args=(child,),
        name=f"furu-child-stderr-{process.pid}",
        daemon=True,
    )
    child.stderr_thread.start()
    logger.debug("spawned child %d", process.pid)
    return child


def _request(child: _Child, job: Job) -> JobResultRequest:
    assert child.process.stdin is not None
    assert child.process.stdout is not None
    try:
        child.process.stdin.write(job.model_dump_json() + "\n")
        child.process.stdin.flush()
        line = child.process.stdout.readline()
    except OSError:
        line = ""
    if not line:
        return _crash_result(child)
    return _job_result_adapter.validate_json(line)


def _crash_result(child: _Child) -> JobFailedResult:
    returncode = child.process.wait()
    if child.stderr_thread is not None:
        child.stderr_thread.join(timeout=_RETIRE_TIMEOUT_SECONDS)
    reason = _describe_exit(returncode)
    logger.warning("child %d died with %s", child.process.pid, reason)
    error = f"subprocess died: {reason}"
    if tail := "".join(child.stderr_tail):
        error += f"\nstderr tail:\n{tail}"
    return JobFailedResult(error=error)
