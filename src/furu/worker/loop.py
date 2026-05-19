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
) -> None:
    client = api.ManagerApiClient(server_url, auth_token=auth_token)

    while True:
        match client.lease_job(resources=resource_request):
            case "stop":
                return
            case "wait":
                time.sleep(0.1)
                continue
            case Job() as job:
                try:
                    obj = Furu.from_artifact(job.artifact)
                    with worker_execution_context(lease_id=job.lease_id):
                        _ensure_single_result(obj)
                    client.job_result(job.lease_id, JobCompletedResult())
                except _DependencyNotReady as exc:
                    dependencies = [
                        ArtifactSpec.from_furu(dep) for dep in exc.dependencies
                    ]
                    client.job_result(
                        job.lease_id,
                        JobBlockedResult(dependencies=dependencies),
                    )
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
            case unexpected:
                assert_never(unexpected)
