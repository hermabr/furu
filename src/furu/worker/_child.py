import io
import os
import sys
import traceback
from pathlib import Path

from furu.code_trace import StaleCodeError
from furu.config import _Config, _set_config
from furu.core import Spec
from furu.dependencies import _DependencyNotReady
from furu.execution.launch import asked
from furu.execution.load_or_create import _ensure_group_result
from furu.logging import _configure_child_logging
from furu.metadata import ArtifactSpec
from furu.provenance import _set_submitted_repo_root, _worker_backend
from furu.worker.messages import (
    Job,
    JobBlockedResult,
    JobCompletedResult,
    JobFailedResult,
    JobResult,
)


def _execute(job: Job) -> JobResult:
    _set_submitted_repo_root(Path(job.provenance.git.repo_root))
    try:
        objs = [Spec.from_artifact(artifact) for artifact in job.artifacts]
        with asked(job.asked):
            _ensure_group_result(objs, submit_provenance=job.provenance)
        return JobCompletedResult()
    except _DependencyNotReady as exc:
        return JobBlockedResult(
            dependencies=[ArtifactSpec.from_furu(dep) for dep in exc.dependencies]
        )
    except Exception as exc:  # noqa: BLE001 -- fault barrier: any crash fails the job
        return JobFailedResult(
            error="".join(traceback.format_exception(exc)),
            stale_code=isinstance(exc, StaleCodeError),
        )


def main() -> int:
    # Keep a private copy of stdout for the parent protocol. Fds 1 and 2 go to
    # the worker's stderr, and to a run's log while its create() runs.
    protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "w")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    worker_stderr = os.dup(sys.stderr.fileno())
    # Keep prints in order with log records, which go to stderr line by line.
    assert isinstance(sys.stdout, io.TextIOWrapper)
    sys.stdout.reconfigure(line_buffering=True)

    _set_config(_Config.model_validate_json(sys.stdin.readline()))
    _worker_backend.set(sys.stdin.readline().rstrip("\n"))
    _configure_child_logging(worker_stderr)

    for line in sys.stdin:
        result = _execute(Job.model_validate_json(line))
        protocol_out.write(result.model_dump_json() + "\n")
        protocol_out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
