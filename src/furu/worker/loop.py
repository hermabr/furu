from __future__ import annotations

import queue
import threading
import traceback
from pathlib import Path
from typing import assert_never

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import ClientConnection, connect

from furu.logging import get_logger
from furu.utils import format_duration
from furu.worker import messages
from furu.worker.execute import ChildSlot

type _Event = messages.ServerMessage | messages.JobResult | BaseException | None


def _read_messages(connection: ClientConnection, events: queue.Queue[_Event]) -> None:
    try:
        while True:
            events.put(messages.server_message_adapter.validate_json(connection.recv()))
    except ConnectionClosed:
        events.put(None)
    except Exception as exc:  # noqa: BLE001 -- re-raised on the main thread
        events.put(exc)


def _start_job_thread(
    job: messages.Job,
    child_slot: ChildSlot,
    cancelled: threading.Event,
    events: queue.Queue[_Event],
) -> None:
    def run() -> None:
        try:
            events.put(child_slot.run(job, cancelled=cancelled))
        except Exception as exc:  # noqa: BLE001 -- fault barrier: any crash fails the job
            error = "".join(traceback.format_exception(exc))
            events.put(messages.JobFailedResult(error=error))
        except BaseException as exc:  # noqa: BLE001 -- re-raised on the main thread
            events.put(exc)

    threading.Thread(target=run, name="furu-worker-job", daemon=True).start()


def worker_loop(
    *,
    url: str,
    pool: str,
    idle_timeout: float | None,
    component: str,
    backend: str,
    worker_log: Path | None = None,
) -> None:
    """Run jobs from the manager that started this worker, then exit.

    The worker never reconnects: when the connection drops it finishes its
    current job and exits. Only a worker with its own log (``worker_log``: a
    Slurm worker's output file) notes what it ran there.
    """
    logger = get_logger(component)
    note = logger.info if worker_log is not None else logger.debug
    child_slot = ChildSlot(worker=component, backend=backend, worker_log=worker_log)
    events: queue.Queue[_Event] = queue.Queue()
    job: messages.Job | None = None
    cancelled = threading.Event()  # replaced with each new job
    execution_log: Path | None = None
    connected = True
    try:
        with connect(url, max_size=None) as connection:
            connection.send(
                messages.HelloMessage(
                    worker=component, backend=backend, pool=pool, log=worker_log
                ).model_dump_json()
            )
            threading.Thread(
                target=_read_messages,
                args=(connection, events),
                name="furu-worker-reader",
                daemon=True,
            ).start()
            while connected or job is not None:
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
                        connected = False
                        if job is not None:
                            note(
                                "lost the manager; finishing %s, then exiting",
                                job.label,
                            )
                    case BaseException():
                        raise event
                    case messages.Job():
                        assert job is None
                        job = event
                        if worker_log is not None:
                            if job.execution_log != execution_log:
                                execution_log = job.execution_log
                                logger.info(
                                    "serving exec %s",
                                    execution_log.parent.name[:5],
                                    extra={"path": execution_log},
                                )
                            logger.info(
                                "running %s",
                                job.label,
                                extra={"path": job.run_logs[0]},
                            )
                        cancelled = threading.Event()
                        _start_job_thread(job, child_slot, cancelled, events)
                    case messages.CancelMessage():
                        if job is not None:
                            note("cancelled %s; killing child", job.label)
                            cancelled.set()
                            child_slot.kill()
                    case (
                        messages.JobCompletedResult()
                        | messages.JobFailedResult()
                        | messages.JobBlockedResult()
                    ):
                        job = None
                        if connected:
                            try:
                                connection.send(
                                    messages.job_result_adapter.dump_json(
                                        event
                                    ).decode()
                                )
                            except ConnectionClosed:
                                connected = False
                    case _:
                        assert_never(event)
        note("manager closed the connection; exiting")
    finally:
        child_slot.close()
