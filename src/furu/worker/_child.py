import os
import sys

from pydantic import TypeAdapter

from furu.core import Spec
from furu.provenance import _worker_backend
from furu.worker.execute import execute_job
from furu.worker.protocol import Job, JobResultRequest


def main() -> int:
    _worker_backend.set("subprocess")
    # Keep the protocol channel to the parent on a private descriptor and send
    # everything user code writes to stdout (prints, progress bars) to stderr,
    # so it lands in the worker's log instead of corrupting the protocol.
    protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "w")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    for line in sys.stdin:
        job = Job.model_validate_json(line)
        objs = [Spec.from_artifact(member.artifact) for member in job.members]
        results = execute_job(objs, job=job)
        payload = TypeAdapter(list[JobResultRequest]).dump_json(results).decode()
        protocol_out.write(payload + "\n")
        protocol_out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
