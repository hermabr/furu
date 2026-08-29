from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import traceback
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import assert_never

from furu.config import get_config
from furu.core import Spec
from furu.execution.load_or_create import _cached_to_build_msg, _ensure_group_result
from furu.logging import get_logger
from furu.metadata import ArtifactSpec
from furu.migration.links import result_dir_for_loading
from furu.provenance import EnvironmentIdentity
from furu.spec_metadata import Metadata
from furu.worker.context import _DependencyNotReady, worker_execution_context
from furu.worker.protocol import (
    Job,
    JobBlockedResult,
    JobCompletedResult,
    JobFailedResult,
    JobResult,
    job_result_adapter,
)

logger = get_logger("worker.execute")

_STDERR_TAIL_LINES = 200
_STDERR_TAIL_CHARS = 32 * 1024
_RETIRE_TIMEOUT_SECONDS = 5.0


def execute_job(objs: Sequence[Spec], *, job: Job) -> JobResult:
    try:
        with worker_execution_context():
            _ensure_group_result(objs, submit_provenance=job.provenance)
        return JobCompletedResult()
    except _DependencyNotReady as exc:
        return JobBlockedResult(
            dependencies=[ArtifactSpec.from_furu(dep) for dep in exc.dependencies]
        )
    except Exception as exc:  # noqa: BLE001 -- fault barrier: any crash fails the job
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

    def __init__(self, *, backend: str) -> None:
        self._backend = backend
        self._child = None

    def run(self, objs: Sequence[Spec], *, job: Job, metadata: Metadata) -> JobResult:
        worker_hash = EnvironmentIdentity.capture().uv_lock_hash
        submitted_hash = job.provenance.environment.uv_lock_hash
        if worker_hash != submitted_hash:
            raise RuntimeError(
                "worker uv.lock does not match the submitted environment\n"
                f"  submitted : {submitted_hash}\n"
                f"  worker    : {worker_hash}\n"
                "The worker's project checkout is out of sync with the submit host. "
                "Update the checkout (e.g. git pull) and run:\n"
                "  uv sync"
            )
        if all(result_dir_for_loading(obj) is not None for obj in objs):
            logger.info("%s", _cached_to_build_msg(list(objs), []))
            return JobCompletedResult()

        environment = dict(os.environ)
        for name, value in metadata.environment.items():
            if value is None:
                environment.pop(name, None)
            else:
                environment[name] = value

        if missing := [
            name for name in metadata.required_environment if name not in environment
        ]:
            raise RuntimeError(
                f"required environment variables not set: {', '.join(missing)}"
            )

        child = self._child
        if child is not None:
            same_process_context = (
                child.process.poll() is None and child.environment == environment
            )
            match metadata.reuse:
                case "never":
                    can_reuse = False
                case "same_environment":
                    can_reuse = same_process_context
                case "same_environment_same_spec":
                    can_reuse = (
                        same_process_context
                        and child.spec_name == objs[0]._fully_qualified_name
                    )
                case unreachable:
                    assert_never(unreachable)
            if not can_reuse:
                self.close()
                child = None
        if child is None:
            child = self._child = _spawn(environment, backend=self._backend)
        child.spec_name = objs[0]._fully_qualified_name

        result = _request(child, job)
        if metadata.reuse == "never" or child.process.poll() is not None:
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


def _spawn(environment: dict[str, str], *, backend: str) -> _Child:
    process = subprocess.Popen(
        [sys.executable, "-m", "furu.worker._child"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        text=True,
    )
    assert process.stdin is not None
    process.stdin.write(get_config().model_dump_json() + "\n")
    process.stdin.write(backend + "\n")
    process.stdin.flush()
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


def _request(child: _Child, job: Job) -> JobResult:
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
