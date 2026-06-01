from __future__ import annotations

import time
import traceback
from typing import assert_never

from furu.core import Furu
from furu.execution import _ensure_single_result, api
from furu.metadata import ArtifactSpec
from furu.resources import ResourceRequest
from furu.worker.context import _DependencyNotReady, worker_execution_context
from furu.worker.protocol import (
    Job,
    JobBlockedResult,
    JobCompletedResult,
    JobFailedResult,
)


def worker_loop(
    *,
    server_url: str,
    auth_token: str,
    resource_request: ResourceRequest,
    idle_timeout: float,
    max_consecutive_failures: int | None = None,
) -> None:
    client = api.WorkerApiClient(server_url=server_url, auth_token=auth_token)
    idle_started_at: float | None = None
    consecutive_failures = 0

    while True:
        match client.lease_job(resources=resource_request):
            case "stop":
                return
            case "wait":
                now = time.monotonic()
                if idle_started_at is None:
                    idle_started_at = now
                if now - idle_started_at >= idle_timeout:
                    return
                time.sleep(0.1)  # TODO: make the wait poll interval configurable.
                continue
            case Job() as job:
                idle_started_at = None
                try:
                    obj = Furu.from_artifact(job.artifact)
                    with worker_execution_context(lease_id=job.lease_id):
                        _ensure_single_result(obj)
                    client.job_result(job.lease_id, JobCompletedResult())
                    consecutive_failures = 0
                except _DependencyNotReady as exc:
                    dependencies = [
                        ArtifactSpec.from_furu(dep) for dep in exc.dependencies
                    ]
                    client.job_result(
                        job.lease_id,
                        JobBlockedResult(dependencies=dependencies),
                    )
                    consecutive_failures = 0
                except Exception as exc:
                    client.job_result(
                        job.lease_id,
                        JobFailedResult(
                            error="".join(
                                traceback.format_exception(
                                    type(exc),
                                    exc,
                                    exc.__traceback__,
                                )
                            ),
                        ),
                    )
                    consecutive_failures += 1
                    if (
                        max_consecutive_failures is not None
                        and consecutive_failures > max_consecutive_failures
                    ):
                        return
            case unexpected:
                assert_never(unexpected)
