import io
import json
import os
import sys
import traceback

from furu import plan, provenance
from furu.code_trace import StaleCodeError
from furu.config import _Config, _set_config
from furu.dependencies import _DependencyNotReady, creating
from furu.execution.load_or_create import _ensure_group_result
from furu.logging import _configure_child_logging
from furu.worker.execute import CPUS_ENV_VAR
from furu.worker.messages import (
    Job,
    JobBlockedResult,
    JobCompletedResult,
    JobFailedResult,
    JobResult,
)


def _execute(job: Job, token: str) -> JobResult:
    """Run one job; ``token`` ends the name of the working folder it takes."""
    provenance._set_submitted_repo_root(job.code.checkout)
    try:
        specs = plan.Specs(map(json.loads, job.lines))
        with creating(plan=job.plan, asked=job.asked):
            _ensure_group_result([specs[spec] for spec in job.specs], token=token)
        return JobCompletedResult()
    except _DependencyNotReady as exc:
        return JobBlockedResult(
            dependencies=[dep.object_id for dep in exc.dependencies],
            lines=list(map(plan.dump, exc.lines)),
            plan=exc.plan,
        )
    except Exception as exc:  # noqa: BLE001 -- fault barrier: any crash fails the job
        return JobFailedResult(
            error="".join(traceback.format_exception(exc)),
            stale_code=isinstance(exc, StaleCodeError),
        )


def main() -> int:
    # Before anything counts the CPUs: this child's share of its allocation.
    if cpus := os.environ.get(CPUS_ENV_VAR):
        os.sched_setaffinity(0, map(int, cpus.split(",")))
    # Keep a private copy of stdout for the parent protocol. Fds 1 and 2 go to
    # the worker's stderr, and to a run's log while its create() runs.
    protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "w")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    worker_stderr = os.dup(sys.stderr.fileno())
    # Keep prints in order with log records, which go to stderr line by line.
    assert isinstance(sys.stdout, io.TextIOWrapper)
    sys.stdout.reconfigure(line_buffering=True)

    _set_config(_Config.model_validate_json(sys.stdin.readline()))
    provenance._started_as.update(json.loads(sys.stdin.readline()))
    _configure_child_logging(worker_stderr)

    for line in sys.stdin:
        token, _, job = line.partition(" ")
        result = _execute(Job.model_validate_json(job), token)
        protocol_out.write(result.model_dump_json() + "\n")
        protocol_out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
