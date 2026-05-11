from __future__ import annotations

import argparse
import os

from furu.server import HttpSchedulerClient
from furu.worker_execution import execute_one_artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--job-group-id", required=True)
    args = parser.parse_args()

    array_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    client = HttpSchedulerClient(server_url=args.server_url, token=args.token)
    try:
        lease = client.get_lease_for_array_index(
            job_group_id=args.job_group_id,
            array_index=array_index,
        )
        result = execute_one_artifact(lease)
        client.post_lease_result(lease_id=lease.lease_id, result=result)
    finally:
        client.close()


if __name__ == "__main__":
    main()
