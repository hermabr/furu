from __future__ import annotations

import logging
import time
import traceback
from typing import assert_never

from furu.core import Furu
from furu.execution import _ensure_single_result, api
from furu.logging import get_logger, log_component, log_event
from furu.metadata import ArtifactSpec
from furu.resources import ResourceRequest
from furu.worker.context import _DependencyNotReady, worker_execution_context
from furu.worker.protocol import (
    Job,
    JobBlockedResult,
    JobCompletedResult,
    JobFailedResult,
    JobResultRequest,
)

logger = get_logger("worker.loop")


def worker_loop(
    *,
    server_url: str,
    auth_token: str,
    resource_request: ResourceRequest,
    idle_timeout: float,
    max_consecutive_failures: int | None = None,
    component: str | None = None,
) -> None:
    client = api.WorkerApiClient(server_url=server_url, auth_token=auth_token)
    idle_started_at: float | None = None
    consecutive_failures = 0

    with log_component(component):
        while True:
            logger.debug("worker requesting new task from server")
            match client.lease_job(resources=resource_request):
                case "stop":
                    logger.info("worker told to stop")
                    return
                case "wait":
                    now = time.monotonic()
                    if idle_started_at is None:
                        idle_started_at = now
                        logger.info("worker told to wait")
                    if now - idle_started_at >= idle_timeout:
                        return
                    time.sleep(0.1)  # TODO: make the wait poll interval configurable.
                    continue
                case Job() as job:
                    idle_started_at = None
                    task_label: str | None = None
                    job_result: JobResultRequest
                    task_started = time.monotonic()
                    try:
                        obj = Furu.from_artifact(job.artifact)
                        task_label = obj._log_label
                        log_event(
                            logger,
                            logging.INFO,
                            "worker received task: lease_id=%s task=%s",
                            job.lease_id,
                            task_label,
                            event="received",
                            label=task_label,
                            lease=job.lease_id,
                        )
                        with worker_execution_context(lease_id=job.lease_id):
                            _ensure_single_result(obj)
                        job_result = JobCompletedResult()
                    except _DependencyNotReady as exc:
                        job_result = JobBlockedResult(
                            dependencies=[
                                ArtifactSpec.from_furu(dep) for dep in exc.dependencies
                            ]
                        )
                    except Exception as exc:
                        job_result = JobFailedResult(
                            error="".join(
                                traceback.format_exception(
                                    type(exc),
                                    exc,
                                    exc.__traceback__,
                                )
                            ),
                        )

                    client.job_result(job.lease_id, job_result)
                    elapsed = time.monotonic() - task_started

                    match job_result:
                        case JobCompletedResult():
                            status = "completed"
                            event, event_status = "finished", "ok"
                            consecutive_failures = 0
                        case JobBlockedResult():
                            status = "blocked"
                            event, event_status = "blocked", None
                            consecutive_failures = 0
                        case JobFailedResult():
                            status = "failed"
                            event, event_status = "failed", "error"
                            consecutive_failures += 1
                        case unexpected_result:
                            assert_never(unexpected_result)

                    if task_label is None:
                        logger.info(
                            "worker finished task: lease_id=%s status=%s",
                            job.lease_id,
                            status,
                        )
                    else:
                        log_event(
                            logger,
                            logging.INFO,
                            "worker finished task: lease_id=%s task=%s status=%s",
                            job.lease_id,
                            task_label,
                            status,
                            event=event,
                            label=task_label,
                            status=event_status,
                            duration=elapsed,
                            lease=job.lease_id,
                        )

                    if (
                        max_consecutive_failures is not None
                        and consecutive_failures > max_consecutive_failures
                    ):
                        return
                case unexpected:
                    assert_never(unexpected)
