from __future__ import annotations

import argparse

from furu.server.client import SchedulerClient
from furu.worker_execution import WorkerExecutionResultKind, execute_one_artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--job-group-id", required=True)
    parser.add_argument("--array-index", required=True, type=int)
    args = parser.parse_args()

    client = SchedulerClient(base_url=args.server_url, token=args.token)
    try:
        lease = client.get_lease_for_array_index(
            job_group_id=args.job_group_id,
            array_index=args.array_index,
        )

        result = execute_one_artifact(
            lease_id=lease.lease_id,
            node_key=lease.node_key,
            artifact=lease.artifact,
        )

        match result.kind:
            case WorkerExecutionResultKind.DONE:
                client.complete(
                    lease.lease_id,
                    node_key=result.node_key,
                )
            case WorkerExecutionResultKind.DEPENDENCY_NOT_READY:
                if result.call_kind is None or result.graph_fragment is None:
                    raise RuntimeError(
                        "dependency result is missing dependency payload"
                    )
                client.report_dependency(
                    lease.lease_id,
                    blocked=result.node_key,
                    call_kind=result.call_kind,
                    dependencies=result.dependencies,
                    graph_fragment=result.graph_fragment,
                )
            case WorkerExecutionResultKind.FAILED:
                client.fail(
                    lease.lease_id,
                    node_key=result.node_key,
                    error_type=result.error_type or "UnknownError",
                    error_message=result.error_message or "",
                    traceback=result.traceback or "",
                )
    finally:
        client.close()


if __name__ == "__main__":
    main()
