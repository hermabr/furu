"""The furu manager: one queue and one set of worker pools for every run.

It holds only memory. Scripts send their jobs and pool definitions on every
connection, so a restarted manager needs no state file: scripts plan again
from disk and resend. Run it with ``uvx furu@<version> manager``.
"""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Generator, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from secrets import token_urlsafe
from typing import assert_never

from websockets.exceptions import ConnectionClosed
from websockets.protocol import State
from websockets.sync.server import ServerConnection, basic_auth, serve

from furu.logging import get_logger
from furu.worker.backends.local import LocalThreadPool
from furu.worker.backends.protocol import WorkerPool
from furu.worker.backends.slurm.pool import SlurmWorkerPool
from furu.worker.messages import (
    CancelMessage,
    ErrorMessage,
    HelloMessage,
    Job,
    JobCompletedResult,
    JobFailedResult,
    JobResult,
    LocalPool,
    Pool,
    ResultMessage,
    SlurmPool,
    Submission,
    SubmitMessage,
    WarningMessage,
    first_message_adapter,
    job_result_adapter,
    script_message_adapter,
)

logger = get_logger("manager")

# How often an idle worker's handler checks that the worker is still there.
_IDLE_CHECK_INTERVAL_S = 5.0


@dataclass(eq=False)
class _Script:
    connection: ServerConnection
    pools: dict[str, Pool] = field(default_factory=dict)

    def send(self, message: ResultMessage | WarningMessage | ErrorMessage) -> None:
        with suppress(ConnectionClosed):
            self.connection.send(message.model_dump_json())


@dataclass(eq=False)  # compared by identity
class _Queued:
    script: _Script
    submission: Submission


@dataclass(eq=False)
class _Lease:
    script: _Script | None  # None once the script that submitted it is gone
    queued: list[_Queued]
    pool: str

    @property
    def job(self) -> Job:
        first, *_ = jobs = [queued.submission.job for queued in self.queued]
        return first.model_copy(
            update={
                "artifacts": [artifact for job in jobs for artifact in job.artifacts],
                "run_logs": [run_log for job in jobs for run_log in job.run_logs],
            }
        )

    @property
    def throttle(self) -> str | None:
        throttle = self.queued[0].submission.throttle
        return throttle.key if throttle is not None else None


@dataclass(eq=False)
class Manager:
    url: str
    manager_dir: Path | None  # None for a script's in-process manager
    lock: threading.Condition = field(default_factory=threading.Condition)
    scripts: list[_Script] = field(default_factory=list)
    queue: list[_Queued] = field(default_factory=list)
    leases: dict[ServerConnection, _Lease] = field(default_factory=dict)
    pools: dict[str, WorkerPool] = field(default_factory=dict)
    stopping: bool = False

    @classmethod
    @contextmanager
    def serve(
        cls, *, listen_host: str, connect_host: str, manager_dir: Path | None
    ) -> Generator[Manager]:
        token = token_urlsafe(32)
        server = serve(
            lambda connection: manager._serve(connection),
            listen_host,
            0,
            process_request=basic_auth(credentials=("furu", token)),
            max_size=None,
        )
        port = server.socket.getsockname()[1]
        manager = cls(
            url=f"ws://furu:{token}@{connect_host}:{port}", manager_dir=manager_dir
        )
        thread = threading.Thread(target=server.serve_forever, name="furu-manager")
        thread.start()
        try:
            yield manager
        finally:
            with manager.lock:
                manager.stopping = True
                manager.lock.notify_all()
            server.shutdown()  # closes every connection and waits for handlers
            thread.join(timeout=10)
            for pool in manager.pools.values():
                pool.stop()

    def _serve(self, connection: ServerConnection) -> None:
        try:
            match first_message_adapter.validate_json(connection.recv(timeout=10)):
                case SubmitMessage() as submit:
                    self._serve_script(connection, submit)
                case HelloMessage() as hello:
                    self._serve_worker(connection, hello)
                case _ as unreachable:
                    assert_never(unreachable)
        except (ConnectionClosed, TimeoutError):
            pass

    # --- scripts ---------------------------------------------------------------

    def _serve_script(
        self, connection: ServerConnection, submit: SubmitMessage
    ) -> None:
        script = _Script(connection)
        with self.lock:
            self.scripts.append(script)
        logger.info("script connected · %s", _peer(connection))
        message: SubmitMessage | CancelMessage = submit
        try:
            while True:
                match message:
                    case SubmitMessage():
                        self._submit(script, message)
                    case CancelMessage():
                        self._cancel(script)
                    case _ as unreachable:
                        assert_never(unreachable)
                message = script_message_adapter.validate_json(connection.recv())
        except ConnectionClosed:
            pass
        finally:
            with self.lock:
                self.scripts.remove(script)
                dropped = self._drop_queued_locked(script)
                running = [
                    lease for lease in self.leases.values() if lease.script is script
                ]
                for lease in running:
                    lease.script = None
                self.lock.notify_all()
            logger.info(
                "script left · %s · dropped %d queued · %d still running",
                _peer(connection),
                dropped,
                len(running),
            )

    def _submit(self, script: _Script, message: SubmitMessage) -> None:
        with self.lock:
            script.pools = message.pools
            for key, definition in message.pools.items():
                if key not in self.pools:
                    self.pools[key] = self._start_pool(key, definition)
            self.queue += [_Queued(script, job) for job in message.jobs]
            self.lock.notify_all()
        if message.jobs:
            logger.info(
                "queued %d jobs from %s", len(message.jobs), _peer(script.connection)
            )

    def _start_pool(self, key: str, definition: Pool) -> WorkerPool:
        logger.info("starting pool %s", key)
        match definition:
            case SlurmPool():
                if self.manager_dir is None:
                    raise RuntimeError("slurm pools need a shared furu manager")
                return SlurmWorkerPool(
                    manager=self,
                    key=key,
                    definition=definition,
                    directory=self.manager_dir / "workers" / key.replace(":", "-"),
                )
            case LocalPool():
                return LocalThreadPool(manager=self, key=key, definition=definition)
            case _ as unreachable:
                assert_never(unreachable)

    def _cancel(self, script: _Script) -> None:
        with self.lock:
            dropped = self._drop_queued_locked(script)
            workers = [
                worker
                for worker, lease in self.leases.items()
                if lease.script is script
            ]
            self.lock.notify_all()
        logger.info(
            "cancelled %d queued and %d running jobs from %s",
            dropped,
            len(workers),
            _peer(script.connection),
        )
        for worker in workers:
            with suppress(ConnectionClosed):
                worker.send(CancelMessage().model_dump_json())

    def _drop_queued_locked(self, script: _Script) -> int:
        before = len(self.queue)
        self.queue = [queued for queued in self.queue if queued.script is not script]
        return before - len(self.queue)

    def notify(self, pool: str, message: WarningMessage | ErrorMessage) -> None:
        """Tell every script that uses ``pool``, e.g. that its workers fail."""
        (logger.warning if isinstance(message, WarningMessage) else logger.error)(
            "%s · %s", pool, message.reason
        )
        with self.lock:
            scripts = [script for script in self.scripts if pool in script.pools]
        for script in scripts:
            script.send(message)

    # --- workers ---------------------------------------------------------------

    def _serve_worker(self, connection: ServerConnection, hello: HelloMessage) -> None:
        worker = hello.worker
        with self.lock:
            known = hello.pool in self.pools
        if not known:
            logger.warning("worker %s belongs to no pool here; closing", worker)
            return
        logger.debug("worker connected · %s", worker)
        while (lease := self._lease(connection, pool=hello.pool)) is not None:
            job = lease.job
            logger.info(
                "leased %s to %s", job.label, worker, extra={"path": job.run_logs[0]}
            )
            result: JobResult
            try:
                connection.send(job.model_dump_json())
                result = job_result_adapter.validate_json(connection.recv())
            except ConnectionClosed:  # the next _lease sees it and returns None
                if self.stopping:  # the worker finishes its job alone; it hasn't failed
                    return
                logger.warning("lost %s mid-job", worker, extra={"path": hello.log})
                result = JobFailedResult(error=f"worker {worker} disconnected")
            with self.lock:
                del self.leases[connection]
                if isinstance(result, JobCompletedResult):
                    self.pools[lease.pool].job_completed()
                script = lease.script
                self.lock.notify_all()
            if script is not None:
                script.send(
                    ResultMessage(
                        object_ids=[artifact.object_id for artifact in job.artifacts],
                        result=result,
                    )
                )

    def _lease(self, connection: ServerConnection, *, pool: str) -> _Lease | None:
        with self.lock:
            while not self.stopping and connection.state is State.OPEN:
                if (queued := next(self._leasable_locked(pool), None)) is not None:
                    self.queue = [q for q in self.queue if q not in queued]
                    lease = _Lease(script=queued[0].script, queued=queued, pool=pool)
                    self.leases[connection] = lease
                    return lease
                self.lock.wait(timeout=_IDLE_CHECK_INTERVAL_S)
            return None

    def _leasable_locked(self, pool: str) -> Iterator[list[_Queued]]:
        """Yield the jobs of each lease ``pool`` could start now, in arrival order.

        Throttles count leases across all scripts, so a running batch counts
        once. Batch-mates come from the same script only.
        """
        running = Counter(
            lease.throttle for lease in self.leases.values() if lease.throttle
        )
        taken: set[_Queued] = set()
        for index, queued in enumerate(self.queue):
            submission = queued.submission
            if queued in taken or pool not in submission.pools:
                continue
            throttle = submission.throttle
            if throttle is not None and running[throttle.key] >= throttle.max_running:
                continue
            group = [queued]
            if (batch := submission.pools[pool]) is not None:
                for other in self.queue[index + 1 :]:
                    if len(group) >= batch.cap:
                        break
                    if (
                        other.script is queued.script
                        and other not in taken
                        and other.submission.pools.get(pool) == batch
                    ):
                        group.append(other)
            taken.update(group)
            if throttle is not None:
                running[throttle.key] += 1
            yield group

    def demand(self, pool: str) -> int:
        """Workers ``pool`` should have: busy ones plus leases it could start."""
        with self.lock:
            max_workers = max(
                (
                    script.pools[pool].max_workers
                    for script in self.scripts
                    if pool in script.pools
                ),
                default=0,
            )
            busy = sum(1 for lease in self.leases.values() if lease.pool == pool)
            startable = sum(1 for _ in islice(self._leasable_locked(pool), max_workers))
            return min(busy + startable, max_workers)


def _peer(connection: ServerConnection) -> str:
    with suppress(OSError, TypeError):
        host, port, *_ = connection.remote_address
        return f"{host}:{port}"
    return "?"
