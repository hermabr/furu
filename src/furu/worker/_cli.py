import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from furu.resources import ResourceRequest
from furu.worker.loop import worker_loop


def _default_worker_component() -> str:
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if array_task_id:
        return f"sw{array_task_id}"
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        return f"sw{job_id[-3:]}"
    return "wkr"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--server-url",
        required=True,
        help="execution coordinator API URL",
    )
    parser.add_argument(
        "--auth-token-file",
        required=True,
        type=Path,
        help="path to a file containing the execution coordinator auth token",
    )
    parser.add_argument(
        "--resource-cpus",
        required=True,
        type=int,
        help="CPU count available to this worker",
    )
    parser.add_argument(
        "--resource-gpus",
        required=True,
        type=int,
        help="GPU count available to this worker",
    )
    parser.add_argument(
        "--idle-timeout",
        required=True,
        type=float,
        help="seconds to wait without a lease before this worker exits",
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=None,  # TODO: this should not be default
        help="consecutive failed jobs after which this worker exits",
    )
    parser.add_argument(
        "--component",
        default=None,  # TODO: consider migrating this to be either w{nums} or `slurm` and only if it is slurm do we take the array id. it might also be possible for the slurm script to give the component name directly from bash
        help="component label shown in this worker's logs (defaults to the "
        "Slurm array/job id, or 'wkr')",
    )
    args = parser.parse_args(argv)

    worker_loop(
        server_url=args.server_url,
        auth_token=args.auth_token_file.read_text(encoding="utf-8").rstrip(),
        resource_request=ResourceRequest(
            cpus=args.resource_cpus,
            gpus=args.resource_gpus,
        ),
        idle_timeout=args.idle_timeout,
        max_consecutive_failures=args.max_consecutive_failures,
        component=args.component or _default_worker_component(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
