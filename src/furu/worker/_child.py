import os
import sys

from furu.config import _Config, _set_config
from furu.core import Spec
from furu.provenance import _worker_backend
from furu.worker.execute import execute_job
from furu.worker.protocol import Job


def main() -> int:
    #  Keep a private copy of stdout for the parent protocol; send user stdout to the worker log.
    protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "w")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    _set_config(_Config.model_validate_json(sys.stdin.readline()))
    _worker_backend.set(sys.stdin.readline().rstrip("\n"))

    for line in sys.stdin:
        job = Job.model_validate_json(line)
        objs = [Spec.from_artifact(artifact) for artifact in job.artifacts]
        result = execute_job(objs, job=job)
        protocol_out.write(result.model_dump_json() + "\n")
        protocol_out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
