from __future__ import annotations

import time
import traceback
import threading
from typing import assert_never

from furu.core import Furu
from furu.execution import _ensure_single_result, api
from furu.metadata import ArtifactSpec
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
    stop_event: threading.Event | None = None,
) -> None:
    if stop_event is not None and stop_event.is_set():
        return

    client = api.ManagerApiClient(server_url)

    while True:
        if stop_event is not None and stop_event.is_set():
            return

        match client.lease_job():
            case "stop":
                return
            case "wait":
                if stop_event is None:
                    time.sleep(0.1)
                else:
                    stop_event.wait(timeout=0.1)
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
