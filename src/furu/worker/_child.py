import os
import sys

from furu.core import Spec
from furu.provenance import _worker_backend
from furu.worker.execute import execute_job
from furu.worker.protocol import Job


def main() -> int:
    _worker_backend.set("subprocess")
    # Keep the protocol channel to the parent on a private descriptor and send
    # everything user code writes to stdout (prints, progress bars) to stderr,
    # so it lands in the worker's log instead of corrupting the protocol.
    protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "w")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    for line in sys.stdin:
        job = Job.model_validate_json(line)
        obj = Spec.from_artifact(job.artifact)
        result = execute_job(obj, job=job)
        protocol_out.write(result.model_dump_json() + "\n")
        protocol_out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
