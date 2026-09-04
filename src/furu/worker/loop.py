from __future__ import annotations

import contextvars
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import assert_never

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import ClientConnection, connect

from furu.config import _Config, _read_worker_json_config, get_config
from furu.logging import _scoped_component, get_logger, log_detail
from furu.resources import ResourceRequest
from furu.utils import format_duration
from furu.worker import protocol
from furu.worker.execute import ChildSlot

logger = get_logger("worker.loop")

type _Event = protocol.ServerMessage | protocol.JobResult | BaseException | None


def _run_job(
    job: protocol.Job, child_slot: ChildSlot, cancelled: threading.Event
) -> protocol.JobResult:
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
        result = child_slot.run(job, cancelled=cancelled)
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
            events.put(protocol.server_message_adapter.validate_json(connection.recv()))
    except ConnectionClosed:
        events.put(None)
    except Exception as exc:  # noqa: BLE001 -- re-raised on the main thread
        events.put(exc)


def _start_job_thread(
    job: protocol.Job,
    child_slot: ChildSlot,
    cancelled: threading.Event,
    events: queue.Queue[_Event],
) -> threading.Thread:
    def run() -> None:
        try:
            events.put(_run_job(job, child_slot, cancelled))
        except BaseException as exc:  # noqa: BLE001 -- re-raised on the main thread
            events.put(exc)

    thread = threading.Thread(
        target=contextvars.copy_context().run,
        args=(run,),
        name="furu-worker-job",
        daemon=True,
    )
    thread.start()
    return thread


def _read_target(coordinator: str | Path) -> tuple[str, _Config | None]:
    if isinstance(coordinator, Path):
        return _read_worker_json_config(coordinator)
    return coordinator, None


def worker_loop(
    *,
    coordinator: str | Path,
    resource_request: ResourceRequest,
    idle_timeout: float | None,
    component: str,
    backend: str,
    materialize_snapshot: bool,
    max_failures: int | None = None,
) -> None:
    with _scoped_component(component):
        target = _read_target(coordinator)
        if target[1] is not None and target[1] != get_config():
            logger.info("worker configuration changed before startup; exiting")
            return
        child_slot = ChildSlot(
            backend=backend, materialize_snapshot=materialize_snapshot
        )
        events: queue.Queue[_Event] = queue.Queue()
        job: protocol.Job | None = None
        job_thread: threading.Thread | None = None
        cancelled = threading.Event()  # replaced with each new job
        result: protocol.JobResult | None = None
        failures = 0
        try:
            while True:
                with connect(target[0], max_size=None) as connection:
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
                        events.put(result)  # finished while disconnected
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
                            case None:
                                break
                            case BaseException():
                                raise event
                            case protocol.Job():
                                assert job is None
                                job = event
                                cancelled = threading.Event()
                                job_thread = _start_job_thread(
                                    job, child_slot, cancelled, events
                                )
                            case protocol.CancelMessage():
                                if job is not None and result is None:
                                    logger.info(
                                        "cancelled %s; killing child",
                                        job.artifacts[0].log_label,
                                    )
                                    cancelled.set()
                                    child_slot.kill()
                            case (
                                protocol.JobCompletedResult()
                                | protocol.JobFailedResult()
                                | protocol.JobBlockedResult()
                            ):
                                result = event
                                try:
                                    connection.send(
                                        protocol.job_result_adapter.dump_json(
                                            result
                                        ).decode()
                                    )
                                except ConnectionClosed:
                                    continue  # Wait for the reader's None before reconnecting.
                                job = result = None
                                job_thread = None
                                if isinstance(event, protocol.JobCompletedResult):
                                    failures = 0
                                elif isinstance(event, protocol.JobFailedResult):
                                    failures += 1
                                    if failures == max_failures:
                                        logger.error(
                                            "%d jobs failed in a row on this "
                                            "worker; exiting so the pool replaces it",
                                            failures,
                                        )
                                        raise SystemExit(1)
                            case _:
                                assert_never(event)

                new_target = _read_target(coordinator)
                if new_target != target:
                    if new_target[1] != target[1]:
                        if job is not None and result is None:
                            cancelled.set()
                            child_slot.kill()
                            assert job_thread is not None
                            job_thread.join()
                        logger.info("worker configuration changed; exiting")
                        return
                    target = new_target
                    logger.info("coordinator moved; reconnecting")
                    continue
                if job is not None and result is None:
                    logger.warning(
                        "server closed the connection mid-job; killing %s",
                        job.artifacts[0].log_label,
                    )
                    cancelled.set()
                    child_slot.kill()
                logger.info("server closed the connection; worker exiting")
                return
        finally:
            child_slot.close()
