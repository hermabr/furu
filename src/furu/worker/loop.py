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
from furu.worker.execute import Children

type _Event = messages.ServerMessage | messages.DoneMessage | BaseException | None


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
    children: Children,
    cancelled: threading.Event,
    events: queue.Queue[_Event],
) -> None:
    def run() -> None:
        try:
            result = children.run(job, cancelled=cancelled)
        except Exception as exc:  # noqa: BLE001 -- fault barrier: any crash fails the job
            error = "".join(traceback.format_exception(exc))
            result = messages.JobFailedResult(error=error)
        except BaseException as exc:  # noqa: BLE001 -- re-raised on the main thread
            events.put(exc)
            return
        events.put(messages.DoneMessage(lease=job.lease, result=result))

    threading.Thread(target=run, name="furu-worker-job", daemon=True).start()


def worker_loop(
    *,
    url: str,
    pool: str,
    idle_timeout: float | None,
    component: str,
    backend: str,
    worker_log: Path | None = None,
    slots: int = 1,
) -> None:
    """Run jobs from the manager that started this worker, then exit.

    The worker is one allocation of ``slots`` slots, and runs a job of each
    lease at once, each in a child of its own. Idle for ``idle_timeout``, it
    says goodbye and exits once the manager closes the connection. It never
    reconnects: when the connection drops it finishes its jobs and exits.
    Only a worker with its own log (``worker_log``: a Slurm worker's output
    file) notes what it ran there.
    """
    logger = get_logger(component)
    note = logger.info if worker_log is not None else logger.debug
    children = Children(worker=component, backend=backend, log=worker_log, slots=slots)
    events: queue.Queue[_Event] = queue.Queue()
    # What runs, by lease, and the event that says it was cancelled.
    jobs: dict[int, tuple[messages.Job, threading.Event]] = {}
    execution_log: Path | None = None
    connected = True
    said_bye = False
    try:
        with connect(url, max_size=None) as connection:
            connection.send(
                messages.HelloMessage(
                    worker=component,
                    backend=backend,
                    pool=pool,
                    log=worker_log,
                    slots=slots,
                ).model_dump_json()
            )
            threading.Thread(
                target=_read_messages,
                args=(connection, events),
                name="furu-worker-reader",
                daemon=True,
            ).start()
            while connected or jobs:
                try:
                    event = events.get(
                        timeout=None if jobs or said_bye else idle_timeout
                    )
                except queue.Empty:
                    assert idle_timeout is not None
                    note(
                        "no work for %s; saying goodbye", format_duration(idle_timeout)
                    )
                    said_bye = True
                    try:
                        connection.send(messages.ByeMessage().model_dump_json())
                    except ConnectionClosed:
                        return
                    continue
                match event:
                    case None:
                        connected = False
                        if jobs:
                            note(
                                "lost the manager; finishing %s, then exiting",
                                ", ".join(job.label for job, _ in jobs.values()),
                            )
                    case BaseException():
                        raise event
                    case messages.Job() as job:
                        said_bye = False
                        if worker_log is not None:
                            if job.execution_log != execution_log:
                                execution_log = job.execution_log
                                logger.info(
                                    "serving exec %s",
                                    execution_log.parent.name[:5],
                                    extra={"path": execution_log},
                                )
                            logger.info(
                                "running %s", job.label, extra={"path": job.folder}
                            )
                        cancelled = threading.Event()
                        jobs[job.lease] = (job, cancelled)
                        _start_job_thread(job, children, cancelled, events)
                    case messages.CancelJobMessage(lease=lease):
                        if lease in jobs:
                            job, cancelled = jobs[lease]
                            note("cancelled %s; killing its child", job.label)
                            cancelled.set()
                            children.kill(lease)
                    case messages.DoneMessage() as done:
                        del jobs[done.lease]
                        if connected:
                            try:
                                connection.send(done.model_dump_json())
                            except ConnectionClosed:
                                connected = False
                    case _:
                        assert_never(event)
        note("manager closed the connection; exiting")
    finally:
        children.close()
