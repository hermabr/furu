from __future__ import annotations

import queue
import threading
import time
import traceback
from contextlib import suppress
from pathlib import Path
from typing import assert_never

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import ClientConnection, connect

from furu.config import _Config, _read_worker_json_config, get_config
from furu.logging import get_logger
from furu.utils import format_duration
from furu.worker import protocol
from furu.worker.execute import ChildSlot

type _Event = protocol.ServerMessage | protocol.JobResult | BaseException | None


def _run_job(
    job: protocol.Job, child_slot: ChildSlot, cancelled: threading.Event
) -> protocol.JobResult:
    try:
        return child_slot.run(job, cancelled=cancelled)
    except Exception as exc:  # noqa: BLE001 -- fault barrier: any crash fails the job
        return protocol.JobFailedResult(error="".join(traceback.format_exception(exc)))


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
        target=run,
        name="furu-worker-job",
        daemon=True,
    )
    thread.start()
    return thread


def _label(job: protocol.Job) -> str:
    label = job.artifacts[0].log_label
    return label + (f" ×{len(job.artifacts)}" if len(job.artifacts) > 1 else "")


def _read_target(coordinator: str | Path) -> tuple[str, _Config | None]:
    if isinstance(coordinator, Path):
        return _read_worker_json_config(coordinator)
    return coordinator, None


def worker_loop(
    *,
    coordinator: str | Path,
    pool: str,
    idle_timeout: float | None,
    max_failures: int,
    component: str,
    backend: str,
    materialize_snapshot: bool,
    worker_log: Path | None = None,
    disconnect_grace: float = 120.0,
) -> None:
    """Run jobs from the coordinator until it closes or this worker gives up.

    Only a worker with its own log (``worker_log``: a Slurm worker's output
    file) notes what it ran there; a local worker's lines go to the
    coordinator's execution.log, which already says it.
    """
    logger = get_logger(component)
    note = logger.info if worker_log is not None else logger.debug
    target = _read_target(coordinator)
    if target[1] is not None and target[1] != get_config():
        note("worker configuration changed before startup; exiting")
        return
    child_slot = ChildSlot(
        worker=component,
        backend=backend,
        materialize_snapshot=materialize_snapshot,
        worker_log=worker_log,
    )
    events: queue.Queue[_Event] = queue.Queue()
    job: protocol.Job | None = None
    job_thread: threading.Thread | None = None
    cancelled = threading.Event()  # replaced with each new job
    result: protocol.JobResult | None = None
    execution_log: Path | None = None
    failures = 0
    try:
        while True:
            with connect(target[0], max_size=None) as connection:
                connection.send(
                    protocol.HelloMessage(
                        worker=component,
                        backend=backend,
                        pool=pool,
                        running=job.artifacts if job is not None else [],
                        log=worker_log,
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
                        note("no work for %s; exiting", format_duration(idle_timeout))
                        return
                    match event:
                        case None:
                            break
                        case BaseException():
                            raise event
                        case protocol.Job():
                            assert job is None
                            job = event
                            if worker_log is not None:
                                if job.execution_log != execution_log:
                                    execution_log = job.execution_log
                                    logger.info(
                                        "connected · exec %s",
                                        execution_log.parent.name[:5],
                                        extra={"path": execution_log},
                                    )
                                logger.info(
                                    "running %s",
                                    _label(job),
                                    extra={"path": job.run_logs[0]},
                                )
                            cancelled = threading.Event()
                            job_thread = _start_job_thread(
                                job, child_slot, cancelled, events
                            )
                        case protocol.CancelMessage():
                            if job is not None and result is None:
                                note(
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
                            failures = (
                                failures + 1
                                if isinstance(event, protocol.JobFailedResult)
                                else 0
                            )
                            if failures == max_failures:
                                logger.warning(
                                    "exiting after %d failed jobs in a row", failures
                                )
                                raise SystemExit(
                                    f"{component} exited after {failures} failed "
                                    "jobs in a row"
                                )
                        case _:
                            assert_never(event)

            deadline = time.monotonic() + (
                disconnect_grace if isinstance(coordinator, Path) else 0
            )
            new_target = target
            while True:
                with suppress(ValueError):  # truncated mid-rewrite
                    new_target = _read_target(coordinator)
                if new_target != target or time.monotonic() >= deadline:
                    break
                time.sleep(1)
            if new_target != target:
                if new_target[1] != target[1]:
                    if job is not None and result is None:
                        cancelled.set()
                        child_slot.kill()
                        assert job_thread is not None
                        job_thread.join()
                    note("worker configuration changed; exiting")
                    return
                target = new_target
                note("coordinator moved; reconnecting")
                continue
            if job is not None and result is None:
                logger.warning(
                    "server closed the connection mid-job; killing %s",
                    job.artifacts[0].log_label,
                )
                cancelled.set()
                child_slot.kill()
            note("server closed the connection; exiting")
            return
    finally:
        child_slot.close()
