from __future__ import annotations

import time
import traceback
from typing import assert_never

from websockets.exceptions import ConnectionClosed
from websockets.headers import build_authorization_basic
from websockets.sync.client import connect

from furu.core import Spec
from furu.logging import _scoped_component, get_logger, log_detail
from furu.provenance import _worker_backend
from furu.resources import ResourceRequest
from furu.spec_metadata import Subprocess
from furu.utils import format_duration
from furu.worker import protocol
from furu.worker.execute import ChildSlot, execute_job

logger = get_logger("worker.loop")


def _run_job(
    job: protocol.Job, child_slot: ChildSlot
) -> tuple[protocol.JobResult, str | None]:
    task_label: str | None = None
    try:
        objs = [Spec.from_artifact(member.artifact) for member in job.members]
        task_label = objs[0]._log_label
        if len(objs) > 1:
            task_label += f" ×{len(objs)}"
        logger.info(
            "received %s",
            task_label,
            extra=log_detail(
                leases=",".join(member.lease_id for member in job.members),
                members=len(job.members),
            ),
        )
        match objs[0]._metadata.execution:
            case "inline":
                return execute_job(objs, job=job), task_label
            case Subprocess() as execution:
                return child_slot.run(objs, job=job, execution=execution), task_label
            case unexpected_execution:
                assert_never(unexpected_execution)
    except Exception as exc:  # noqa: BLE001 -- fault barrier: any crash fails the job
        return (
            protocol.JobFailedResult(
                error="".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
            ),
            task_label,
        )


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
        consecutive_failures = 0
        child_slot = ChildSlot()

        try:
            with connect(
                server_url,
                additional_headers={
                    "Authorization": build_authorization_basic("furu", auth_token)
                },
                max_size=None,
            ) as connection:
                connection.send(
                    protocol.HelloMessage(
                        worker=component,
                        backend=backend,
                        resources=resource_request,
                    ).model_dump_json()
                )
                while True:
                    try:
                        message = connection.recv(timeout=idle_timeout)
                    except TimeoutError:
                        logger.info(
                            "no work for %s; worker exiting",
                            format_duration(idle_timeout),
                        )
                        return
                    except ConnectionClosed:
                        logger.info("server closed the connection; worker exiting")
                        return

                    job = protocol.Job.model_validate_json(message)
                    task_started_at = time.monotonic()
                    job_result, task_label = _run_job(job, child_slot)
                    connection.send(
                        protocol.job_result_adapter.dump_json(job_result).decode()
                    )

                    status = job_result.status
                    if isinstance(job_result, protocol.JobFailedResult):
                        consecutive_failures += 1
                    else:
                        consecutive_failures = 0

                    duration = format_duration(time.monotonic() - task_started_at)
                    first_lease = job.members[0].lease_id
                    logger.info(
                        "finished %s%s · %s",
                        f"{task_label} " if task_label else "",
                        "ok" if status == "completed" else status,
                        duration,
                        extra=log_detail(lease=first_lease, status=status),
                    )

                    if (
                        max_consecutive_failures is not None
                        and consecutive_failures > max_consecutive_failures
                    ):
                        return
        finally:
            child_slot.close()
            _worker_backend.reset(worker_backend_token)
