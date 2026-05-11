from __future__ import annotations

import argparse
import os

from furu.server import SchedulerClient
from furu.worker_execution import execute_one_artifact


def run_worker_once(
    client: SchedulerClient,
    *,
    lease_id: str | None = None,
    job_group_id: str | None = None,
    array_index: int | None = None,
) -> None:
    if lease_id is not None:
        lease = client.get_lease(lease_id)
    else:
        if job_group_id is None:
            raise ValueError("job_group_id is required when lease_id is not provided")
        if array_index is None:
            array_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        lease = client.get_lease_for_array_index(
            job_group_id=job_group_id,
            array_index=array_index,
        )

    result = execute_one_artifact(lease)
    client.post_lease_result(
        lease_id=lease.lease_id,
        result=result,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--job-group-id", required=True)
    args = parser.parse_args()

    client = SchedulerClient.for_url(args.server_url, token=args.token)
    try:
        run_worker_once(client, job_group_id=args.job_group_id)
    finally:
        client.close()


if __name__ == "__main__":
    main()
