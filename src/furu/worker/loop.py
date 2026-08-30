from __future__ import annotations

import contextvars
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Literal

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import ClientConnection, connect

from furu.logging import _scoped_component, get_logger, log_detail
from furu.resources import ResourceRequest
from furu.utils import format_duration
from furu.worker import protocol
from furu.worker.execute import ChildSlot

logger = get_logger("worker.loop")

type _Event = (
    tuple[Literal["job"], protocol.Job]
    | tuple[Literal["cancel"], None]
    | tuple[Literal["closed"], None]
    | tuple[Literal["result"], protocol.JobResult]
    | tuple[Literal["crash"], BaseException]
)


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


def _read_messages(connection: ClientConnection, events: queue.Queue[_Event]) -> None:
    try:
        while True:
            match protocol.server_message_adapter.validate_json(connection.recv()):
                case protocol.Job() as job:
                    events.put(("job", job))
                case protocol.CancelMessage():
                    events.put(("cancel", None))
    except ConnectionClosed:
        events.put(("closed", None))
    except Exception as exc:  # noqa: BLE001 -- re-raised on the main thread
        events.put(("crash", exc))


def _start_job(
    job: protocol.Job, child_slot: ChildSlot, events: queue.Queue[_Event]
) -> None:
    def run() -> None:
        try:
            events.put(("result", _run_job(job, child_slot)))
        except BaseException as exc:  # noqa: BLE001 -- re-raised on the main thread
            events.put(("crash", exc))

    threading.Thread(
        target=contextvars.copy_context().run,
        args=(run,),
        name="furu-worker-job",
        daemon=True,
    ).start()


def worker_loop(
    *,
    coordinator: str | Path,
    resource_request: ResourceRequest,
    idle_timeout: float | None,
    component: str,
    backend: str,
) -> None:
    with _scoped_component(component):
        coordinator_url = (
            coordinator.read_text(encoding="utf-8").strip()
            if isinstance(coordinator, Path)
            else coordinator
        )
        child_slot = ChildSlot(backend=backend)
        events: queue.Queue[_Event] = queue.Queue()
        job: protocol.Job | None = None
        result: protocol.JobResult | None = None
        try:
            while True:
                with connect(coordinator_url, max_size=None) as connection:
                    connection.send(
                        protocol.HelloMessage(
                            worker=component,
                            backend=backend,
                            resources=resource_request,
                            running=job.artifacts if job is not None else [],
                        ).model_dump_json()
                    )
                    threading.Thread(
                        target=_read_messages,
                        args=(connection, events),
                        name="furu-worker-reader",
                        daemon=True,
                    ).start()
                    if result is not None:
                        events.put(("result", result))
                    while True:
                        try:
                            event = events.get(
                                timeout=None if job is not None else idle_timeout
                            )
                        except queue.Empty:
                            assert idle_timeout is not None
                            logger.info(
                                "no work for %s; worker exiting",
                                format_duration(idle_timeout),
                            )
                            return
                        match event:
                            case ("closed", _):
                                break
                            case ("crash", exc):
                                raise exc
                            case ("result", finished):
                                result = finished
                                try:
                                    connection.send(
                                        protocol.job_result_adapter.dump_json(
                                            finished
                                        ).decode()
                                    )
                                except ConnectionClosed:
                                    continue
                                job = result = None
                            case ("job", leased):
                                assert job is None
                                job = leased
                                _start_job(job, child_slot, events)
                            case ("cancel", _):
                                if job is not None and result is None:
                                    logger.info(
                                        "cancelled %s; killing child",
                                        job.artifacts[0].log_label,
                                    )
                                    child_slot.kill()

                if isinstance(coordinator, Path):
                    new_url = coordinator.read_text(encoding="utf-8").strip()
                    if new_url != coordinator_url:
                        coordinator_url = new_url
                        logger.info("coordinator moved; reconnecting")
                        continue
                if job is not None and result is None:
                    logger.warning(
                        "server closed the connection mid-job; killing %s",
                        job.artifacts[0].log_label,
                    )
                    child_slot.kill()
                logger.info("server closed the connection; worker exiting")
                return
        finally:
            child_slot.close()
