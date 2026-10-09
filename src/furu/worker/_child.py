import ctypes
import io
import os
import sys
import time
import traceback
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import assert_never

from furu.config import _Config, _set_config
from furu.core import Spec
from furu.execution.load_or_create import _ensure_group_result
from furu.logging import (
    _close_sections,
    _configure_child_logging,
    _display_path,
    _open_sections,
)
from furu.metadata import ArtifactSpec
from furu.provenance import _worker_backend
from furu.utils import error_summary, format_duration
from furu.worker.context import _DependencyNotReady, worker_execution_context
from furu.worker.protocol import (
    Job,
    JobBlockedResult,
    JobCompletedResult,
    JobFailedResult,
    JobResult,
)


def _execute(job: Job) -> JobResult:
    try:
        objs = [Spec.from_artifact(artifact) for artifact in job.artifacts]
        _ensure_group_result(objs, submit_provenance=job.provenance)
        return JobCompletedResult()
    except _DependencyNotReady as exc:
        return JobBlockedResult(
            dependencies=[ArtifactSpec.from_furu(dep) for dep in exc.dependencies]
        )
    except Exception as exc:  # noqa: BLE001 -- fault barrier: any crash fails the job
        return JobFailedResult(error="".join(traceback.format_exception(exc)))


def _flush() -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    with suppress(OSError, AttributeError):  # C stdio buffers, where reachable
        ctypes.CDLL(None).fflush(None)


@contextmanager
def _output_to(path: Path, *, restore_fd: int) -> Generator[None]:
    """Point fds 1 and 2 at ``path`` so everything the job writes lands there."""
    _flush()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    try:
        yield
    finally:
        _flush()
        os.dup2(restore_fd, 1)
        os.dup2(restore_fd, 2)


def _run(job: Job, *, worker: str, worker_log: Path | None) -> JobResult:
    run_logs = [
        (artifact.log_label, path)
        for artifact, path in zip(job.artifacts, job.run_logs, strict=True)
    ]
    notes = [
        f"exec {job.execution_log.parent.name[:5]} → {_display_path(job.execution_log)}"
    ]
    if worker_log is not None:
        notes.append(f"worker log → {_display_path(worker_log)}")
    _open_sections(run_logs, sys.stderr, f"attempt {job.attempt}", worker, notes=notes)
    started_at = time.monotonic()
    result = _execute(job)
    _flush()
    duration = format_duration(time.monotonic() - started_at)
    match result:
        case JobCompletedResult():
            outcome = f"ok · {duration}"
        case JobFailedResult(error=error):
            outcome = f"failed · {duration} · {error_summary(error)}"
        case JobBlockedResult(dependencies=dependencies):
            missing = "dependency" if len(dependencies) == 1 else "dependencies"
            outcome = f"blocked · {duration} · {len(dependencies)} missing {missing}"
        case _:
            assert_never(result)
    _close_sections(run_logs, sys.stderr, outcome)
    return result


def main() -> int:
    # Keep a private copy of stdout for the parent protocol. Between jobs fds 1
    # and 2 go to the worker's stderr; during a job, to the job's run.log.
    protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "w")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    worker_stderr = os.dup(sys.stderr.fileno())
    # Keep prints in order with log records, which go to stderr line by line.
    assert isinstance(sys.stdout, io.TextIOWrapper)
    sys.stdout.reconfigure(line_buffering=True)

    _set_config(_Config.model_validate_json(sys.stdin.readline()))
    _worker_backend.set(sys.stdin.readline().rstrip("\n"))
    worker = sys.stdin.readline().rstrip("\n")
    worker_log = Path(line) if (line := sys.stdin.readline().rstrip("\n")) else None
    _configure_child_logging()

    with worker_execution_context():
        for line in sys.stdin:
            job = Job.model_validate_json(line)
            with _output_to(job.run_logs[0], restore_fd=worker_stderr):
                result = _run(job, worker=worker, worker_log=worker_log)
            protocol_out.write(result.model_dump_json() + "\n")
            protocol_out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
