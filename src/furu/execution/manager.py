"""The furu manager: one queue and one set of worker pools for every run.

It holds only memory. Scripts send their jobs and pool definitions on every
connection, so a restarted manager needs no state file: scripts plan again
from disk and resend. Run it with ``uvx furu@<version> manager``.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import threading
from collections import Counter
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from importlib.metadata import version
from itertools import islice
from pathlib import Path
from secrets import token_urlsafe
from typing import assert_never

from websockets.exceptions import ConnectionClosed
from websockets.protocol import State
from websockets.sync.server import ServerConnection, basic_auth, serve

from furu.config import get_config
from furu.logging import _execution_log, get_logger
from furu.worker.backends.local import LocalThreadPool
from furu.worker.backends.protocol import WorkerPool
from furu.worker.backends.slurm.pool import SlurmWorkerPool
from furu.worker.protocol import (
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


@dataclass(eq=False)
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
        connections: set[ServerConnection] = set()
        manager: Manager | None = None

        def handler(connection: ServerConnection) -> None:
            assert manager is not None
            with manager.lock:
                connections.add(connection)
            try:
                manager._serve(connection)
            finally:
                with manager.lock:
                    connections.discard(connection)

        server = serve(
            handler,
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
                pools = list(manager.pools.values())
            server.shutdown()
            thread.join(timeout=10)
            with manager.lock:
                open_connections = list(connections)
            for connection in open_connections:
                connection.close()
            for pool in pools:
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
                running = 0
                for lease in self.leases.values():
                    if lease.script is script:
                        lease.script = None
                        running += 1
                self.lock.notify_all()
            logger.info(
                "script left · %s · dropped %d queued · %d still running",
                _peer(connection),
                dropped,
                running,
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
            try:
                connection.send(job.model_dump_json())
            except ConnectionClosed:
                with self.lock:
                    del self.leases[connection]
                    if lease.script in self.scripts:
                        self.queue[:0] = lease.queued
                    self.lock.notify_all()
                return
            label = job.artifacts[0].log_label
            if len(job.artifacts) > 1:
                label += f" ×{len(job.artifacts)}"
            logger.info(
                "leased %s to %s", label, worker, extra={"path": job.run_logs[0]}
            )
            result: JobResult
            lost = False
            try:
                result = job_result_adapter.validate_json(connection.recv())
            except ConnectionClosed:
                logger.warning("lost %s mid-job", worker, extra={"path": hello.log})
                result = JobFailedResult(error=f"worker {worker} disconnected")
                lost = True
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
            if lost:
                return

    def _lease(self, connection: ServerConnection, *, pool: str) -> _Lease | None:
        with self.lock:
            while not self.stopping and connection.state is State.OPEN:
                if (queued := next(self._leasable_locked(pool), None)) is not None:
                    taken = set(map(id, queued))
                    self.queue = [q for q in self.queue if id(q) not in taken]
                    lease = self.leases[connection] = _Lease(
                        script=queued[0].script,
                        queued=queued,
                        pool=pool,
                    )
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
        taken: set[int] = set()
        for index, queued in enumerate(self.queue):
            submission = queued.submission
            if id(queued) in taken or pool not in submission.pools:
                continue
            throttle = submission.throttle
            if throttle is not None and running[throttle.key] >= throttle.max_running:
                continue
            group = [queued]
            if (batch := submission.pools[pool]) is not None:
                for other in self.queue[index + 1 :]:
                    if len(group) >= batch.cap:
                        break
                    other_batch = other.submission.pools.get(pool)
                    if (
                        other.script is queued.script
                        and id(other) not in taken
                        and other_batch is not None
                        and other_batch.batch == batch.batch
                    ):
                        group.append(other)
            taken.update(map(id, group))
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="furu")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "manager", help="run the queue and the worker pools for every furu run"
    )
    parser.parse_args(argv)

    config = get_config()
    manager_dir = config.manager_dir.resolve()
    manager_dir.mkdir(parents=True, exist_ok=True)
    with (
        _execution_log(manager_dir / "manager.log"),
        Manager.serve(
            listen_host="0.0.0.0",
            connect_host=config.worker.connect_host or socket.getfqdn(),
            manager_dir=manager_dir,
        ) as manager,
    ):
        # In place, not by rename: on NFS a file replaced by rename stays
        # invisible to nodes that had it open.
        address = manager_dir / "address.json"
        fd = os.open(address, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump({"url": manager.url, "furu_version": version("furu")}, file)
        logger.info(
            "manager listening on %s · furu %s",
            manager.url.rpartition("@")[2],
            version("furu"),
            extra={"path": address},
        )
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            logger.info("stopping; pending workers are cancelled, running ones finish")
    return 0
