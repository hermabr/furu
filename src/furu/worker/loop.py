from __future__ import annotations

import time
import traceback

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from furu.logging import _scoped_component, get_logger, log_detail
from furu.resources import ResourceRequest
from furu.utils import format_duration
from furu.worker import protocol
from furu.worker.execute import ChildSlot

logger = get_logger("worker.loop")


def _run_job(job: protocol.Job, child_slot: ChildSlot) -> protocol.JobResult:
    label = job.artifacts[0].log_label
    if len(job.artifacts) > 1:
        label += f" ×{len(job.artifacts)}"
    logger.info(
        "received %s",
        label,
        extra=log_detail(
            object_ids=",".join(artifact.object_id for artifact in job.artifacts)
        ),
    )
    started_at = time.monotonic()
    try:
        result = child_slot.run(job)
    except Exception as exc:  # noqa: BLE001 -- fault barrier: any crash fails the job
        result = protocol.JobFailedResult(
            error="".join(traceback.format_exception(exc))
        )
    logger.info(
        "finished %s %s · %s",
        label,
        "ok" if result.status == "completed" else result.status,
        format_duration(time.monotonic() - started_at),
        extra=log_detail(status=result.status),
    )
    return result


def worker_loop(
    *,
    coordinator_url: str,
    resource_request: ResourceRequest,
    idle_timeout: float | None,
    component: str,
    backend: str,
) -> None:
    with _scoped_component(component):
        child_slot = ChildSlot(backend=backend)

        try:
            with connect(coordinator_url, max_size=None) as connection:
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
                        assert idle_timeout is not None
                        logger.info(
                            "no work for %s; worker exiting",
                            format_duration(idle_timeout),
                        )
                        return
                    except ConnectionClosed:
                        logger.info("server closed the connection; worker exiting")
                        return

                    job = protocol.Job.model_validate_json(message)
                    job_result = _run_job(job, child_slot)
                    connection.send(
                        protocol.job_result_adapter.dump_json(job_result).decode()
                    )
        finally:
            child_slot.close()
