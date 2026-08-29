import os
import sys
import traceback

from furu.config import _Config, _set_config
from furu.core import Spec
from furu.execution.load_or_create import _ensure_group_result
from furu.metadata import ArtifactSpec
from furu.provenance import _worker_backend
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


def main() -> int:
    #  Keep a private copy of stdout for the parent protocol; send user stdout to the worker log.
    protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "w")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    _set_config(_Config.model_validate_json(sys.stdin.readline()))
    _worker_backend.set(sys.stdin.readline().rstrip("\n"))

    for line in sys.stdin:
        result = _execute(Job.model_validate_json(line))
        protocol_out.write(result.model_dump_json() + "\n")
        protocol_out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
