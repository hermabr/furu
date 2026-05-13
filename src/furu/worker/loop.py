from __future__ import annotations

import time
import traceback

from furu.core import Furu
from furu.execution import _execute_one
from furu.execution import api
from furu.metadata import ArtifactSpec
from furu.worker.context import _DependencyNotReady, worker_execution_context
from furu.worker.protocol import (
    BlockedRequest,
    FinishFailedRequest,
    FinishSuccessRequest,
    Job,
)


def worker_loop(
    *,
    server_url: str,
    wait_interval: float = 0.1,
) -> None:
    client = api.ManagerApiClient(server_url)

    while True:
        response = client.get_job()

        if response == "stop":
            return
        if response == "wait":
            time.sleep(wait_interval)
            continue

        job = response
        try:
            _run_job(job)
        except _DependencyNotReady as exc:
            dependencies = [ArtifactSpec.from_furu(dep) for dep in exc.dependencies]
            client.report_blocked(
                job.lease_id,
                BlockedRequest(dependencies=dependencies),
            )
        except Exception as exc:
            client.finish(
                job.lease_id,
                FinishFailedRequest(
                    error="".join(
                        traceback.format_exception(
                            type(exc),
                            exc,
                            exc.__traceback__,
                        )
                    ),
                ),
            )
        else:
            client.finish(job.lease_id, FinishSuccessRequest())


def _run_job(job: Job) -> None:
    obj = Furu.from_artifact(job.artifact)
    with worker_execution_context(lease_id=job.lease_id):
        _execute_one(obj)
