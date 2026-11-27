from __future__ import annotations

import os
import queue
import threading
import traceback
from pathlib import Path
from typing import assert_never

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import ClientConnection, connect

from furu import slurm
from furu.logging import get_logger
from furu.worker import messages
from furu.worker.execute import Children

# What the loop waits for: a message from the manager, how the job of a lease
# ended, an error to raise, or None once the connection is closed.
type _Event = (
    messages.ServerMessage | tuple[int, messages.JobResult] | BaseException | None
)


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
            events.put((job.lease, children.run(job, cancelled=cancelled)))
        except Exception as exc:  # noqa: BLE001 -- fault barrier: any crash fails the job
            error = "".join(traceback.format_exception(exc))
            events.put((job.lease, messages.JobFailedResult(error=error)))
        except BaseException as exc:  # noqa: BLE001 -- re-raised on the main thread
            events.put(exc)

    threading.Thread(target=run, name="furu-worker-job", daemon=True).start()


def worker_loop(
    *,
    url: str,
    pool: str,
    component: str,
    backend: str,
    slots: int = 1,
    worker_log: Path | None = None,
) -> None:
    """Run the jobs of the manager that started this worker, then exit.

    The worker is one allocation of ``slots`` slots. The manager sends it a
    job whenever the slots the job takes are free, so several run at once,
    and each is named by its lease. The manager also says when an idle
    worker leaves, by closing the connection.

    The worker never reconnects: when the connection drops it finishes its
    jobs and exits. Only a worker with its own log (``worker_log``: a Slurm
    worker's output file) notes what it ran there.
    """
    logger = get_logger(component)
    note = logger.info if worker_log is not None else logger.debug
    children = Children(worker=component, backend=backend, slots=slots, log=worker_log)
    events: queue.Queue[_Event] = queue.Queue()
    # Each running job and what tells its thread that it was cancelled.
    running: dict[int, tuple[messages.Job, threading.Event]] = {}
    execution_log: Path | None = None
    slurm_job_name = os.environ.get("SLURM_JOB_NAME")
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
            while connected or running:
                event = events.get()
                match event:
                    case None:
                        connected = False
                        if running:
                            note(
                                "lost the manager; finishing %s, then exiting",
                                ", ".join(job.label for job, _ in running.values()),
                            )
                    case BaseException():
                        raise event
                    case messages.Job() as job:
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
                                extra={"path": job.folder},
                            )
                        cancelled = threading.Event()
                        running[job.lease] = job, cancelled
                        _start_job_thread(job, children, cancelled, events)
                        # So squeue shows whose job a worker of a shared pool
                        # runs. Nice to have: not waited for, and may fail.
                        if job.slurm_job_name not in (None, slurm_job_name):
                            slurm_job_name = job.slurm_job_name
                            me, name = f"JobId={component}", f"JobName={slurm_job_name}"
                            rename = ("scontrol", "update", me, name)
                            threading.Thread(target=slurm.run, args=rename).start()
                    case messages.CancelJobMessage(lease=lease):
                        if lease in running:
                            job, cancelled = running[lease]
                            note("cancelled %s; killing child", job.label)
                            cancelled.set()
                            children.kill(lease)
                    case (lease, result):
                        del running[lease]
                        if connected:
                            try:
                                connection.send(
                                    messages.JobEndedMessage(
                                        lease=lease, result=result
                                    ).model_dump_json()
                                )
                            except ConnectionClosed:
                                connected = False
                    case _:
                        assert_never(event)
        note("manager closed the connection; exiting")
    finally:
        children.close()
