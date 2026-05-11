from __future__ import annotations

import argparse
import os
import sys

from furu.server import SchedulerClient
from furu.worker_execution import execute_one_artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="furu worker")
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--job-group-id", required=True)
    parser.add_argument(
        "--array-index",
        type=int,
        default=None,
        help="Override SLURM_ARRAY_TASK_ID for testing",
    )
    args = parser.parse_args(argv)

    if args.array_index is not None:
        array_index = args.array_index
    else:
        env_value = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_value is None:
            print(
                "error: SLURM_ARRAY_TASK_ID is required when --array-index is not given",
                file=sys.stderr,
            )
            return 2
        array_index = int(env_value)

    with SchedulerClient.for_remote(
        base_url=args.server_url, token=args.token
    ) as client:
        lease = client.get_lease_for_array_index(
            job_group_id=args.job_group_id,
            array_index=array_index,
        )

        result = execute_one_artifact(lease.node)

        client.post_lease_result(lease_id=lease.lease_id, result=result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
