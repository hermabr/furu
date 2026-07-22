from __future__ import annotations

import time
import traceback
from typing import assert_never, cast

from furu.core import Spec
from furu.execution import api
from furu.logging import _scoped_component, get_logger, log_detail
from furu.provenance import _worker_backend
from furu.resources import ResourceRequest
from furu.spec_metadata import Subprocess
from furu.utils import format_duration
from furu.worker.execute import ChildSlot, execute_job
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
    component: str,
    backend: str,
    max_consecutive_failures: int | None = None,
) -> None:
    worker_backend_token = _worker_backend.set(backend)
    with _scoped_component(component):
        client = api.WorkerApiClient(server_url=server_url, auth_token=auth_token)
        idle_started_at: float | None = None
        consecutive_failures = 0
        child_slot = ChildSlot()

        try:
            while True:
                logger.debug("worker requesting new task from server")
                match client.lease_job(resources=resource_request, worker=component):
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
                        time.sleep(
                            0.1
                        )  # TODO: make the wait poll interval configurable.
                        continue
                    case Job() as job:
                        idle_started_at = None
                        task_started_at = time.monotonic()
                        task_label: str | None = None
                        try:
                            objs = [
                                Spec.from_artifact(member.artifact)
                                for member in job.members
                            ]
                            task_label = objs[0]._log_label
                            logger.info(
                                "received %s%s",
                                task_label,
                                f" + {len(objs) - 1}" if len(objs) > 1 else "",
                                extra=log_detail(lease=job.lease_id),
                            )
                            match objs[0]._metadata.execution:
                                case "inline":
                                    result = execute_job(objs, job=job)
                                case Subprocess() as execution:
                                    result = child_slot.run(
                                        objs, job=job, execution=execution
                                    )
                                case unexpected_execution:
                                    assert_never(unexpected_execution)
                            job_results = result
                        except Exception as exc:
                            result = JobFailedResult(
                                error="".join(
                                    traceback.format_exception(
                                        type(exc),
                                        exc,
                                        exc.__traceback__,
                                    )
                                ),
                            )
                            job_results = cast(
                                list[JobResultRequest],
                                [result for _ in job.members],
                            )

                        for member, job_result in zip(
                            job.members, job_results, strict=True
                        ):
                            client.job_result(member.lease_id, job_result)

                        if any(isinstance(x, JobFailedResult) for x in job_results):
                            status = "failed"
                            consecutive_failures += 1
                        elif any(isinstance(x, JobBlockedResult) for x in job_results):
                            status = "blocked"
                            consecutive_failures = 0
                        else:
                            assert all(
                                isinstance(x, JobCompletedResult) for x in job_results
                            )
                            status = "completed"
                            consecutive_failures = 0

                        duration = format_duration(time.monotonic() - task_started_at)
                        status_word = "ok" if status == "completed" else status
                        if task_label is None:
                            logger.info(
                                "finished %s · %s",
                                status_word,
                                duration,
                                extra=log_detail(lease=job.lease_id, status=status),
                            )
                        else:
                            logger.info(
                                "finished %s %s · %s",
                                task_label,
                                status_word,
                                duration,
                                extra=log_detail(lease=job.lease_id, status=status),
                            )

                        if (
                            max_consecutive_failures is not None
                            and consecutive_failures > max_consecutive_failures
                        ):
                            return
                    case unexpected:
                        assert_never(unexpected)
        finally:
            child_slot.close()
            _worker_backend.reset(worker_backend_token)
