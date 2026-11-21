"""The furu manager: one queue and one set of worker pools for every launch.

It holds only memory. A script sends its plan's lines and its pool
definitions on every connection, so a restarted manager needs no state file:
scripts plan again from disk and resend, and the working folders in
``attempts/`` say which batches were started. Run it with
``uvx furu@<version> manager``.

This module is the part that talks: connections, the threads that run pools,
and one lock around the ``Queue``, which decides what runs. A Slurm pool only
starts and cancels jobs; what it has to tell scripts it returns.
"""

from __future__ import annotations

import threading
import traceback
from collections.abc import Generator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from secrets import token_urlsafe
from typing import assert_never

from websockets.exceptions import ConnectionClosed
from websockets.protocol import State
from websockets.sync.server import ServerConnection, basic_auth, serve

from furu.logging import get_logger
from furu.manager.queue import Queue, ToScript
from furu.manager.slurm_pool import SlurmWorkerPool
from furu.protocol.messages import (
    CancelMessage,
    ErrorMessage,
    GaveUpMessage,
    HelloMessage,
    Job,
    JobCompletedResult,
    JobFailedResult,
    JobResult,
    LocalPool,
    Pool,
    RerunMessage,
    ScriptMessage,
    SlurmPool,
    SubmitMessage,
    WarningMessage,
    first_message_adapter,
    job_result_adapter,
    script_message_adapter,
)
from furu.worker.loop import worker_loop

logger = get_logger("manager")


# How often an idle worker's handler checks that the worker is still there.
_IDLE_CHECK_INTERVAL_S = 5.0
# How often a Slurm pool is sized to its demand.
_SCALE_INTERVAL_S = 10.0
# Not part of a pool's key: its first script sets them and later ones must agree.
_SETTINGS = {"job_name", "idle_timeout_seconds"}


@dataclass(eq=False)
class _Script:
    connection: ServerConnection
    pools: dict[str, Pool] = field(default_factory=dict)

    def send(self, message: ToScript) -> None:
        with suppress(ConnectionClosed):
            self.connection.send(message.model_dump_json())


@dataclass(eq=False)
class Manager:
    url: str
    manager_dir: Path | None  # None for a script's in-process manager
    lock: threading.Condition = field(default_factory=threading.Condition)
    scripts: list[_Script] = field(default_factory=list)
    pools: dict[str, Pool] = field(default_factory=dict)  # those started
    slurm_pools: dict[str, SlurmWorkerPool] = field(default_factory=dict)
    # Each Slurm pool's scaling loop, and the local pools' workers.
    threads: list[threading.Thread] = field(default_factory=list)
    stopping: bool = False
    queue: Queue[_Script, ServerConnection] = field(default_factory=Queue)

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
                # A script that just left may still be forwarding its cancel.
                manager.lock.wait_for(
                    lambda: all(
                        s.connection.state is State.OPEN for s in manager.scripts
                    ),
                    timeout=5,
                )
                manager.stopping = True
                manager.lock.notify_all()
            server.shutdown()  # closes every connection and waits for handlers
            thread.join(timeout=10)
            for pool_thread in manager.threads:
                pool_thread.join(timeout=10)

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

    def _flush(self) -> None:
        with self.lock:
            outbox, self.queue.outbox = self.queue.outbox, []
        for script, message in outbox:
            script.send(message)

    # --- scripts ---------------------------------------------------------------

    def _serve_script(
        self, connection: ServerConnection, submit: SubmitMessage
    ) -> None:
        script = _Script(connection)
        with self.lock:
            self.scripts.append(script)
        logger.info("script connected · %s", _peer(connection))
        message: ScriptMessage = submit
        try:
            while True:
                match message:
                    case SubmitMessage():
                        self._submit(script, message)
                    case RerunMessage():
                        with self.lock:
                            self.queue.rerun(script, message)
                            self.lock.notify_all()
                    case GaveUpMessage():
                        with self.lock:
                            self.queue.gave_up(script, message)
                            self.lock.notify_all()
                    case CancelMessage():
                        self._cancel(script)
                    case _ as unreachable:
                        assert_never(unreachable)
                self._flush()
                message = script_message_adapter.validate_json(connection.recv())
        except ConnectionClosed:
            pass
        finally:
            with self.lock:
                self.scripts.remove(script)
                dropped = self.queue.leave(script)
                running = len(self.queue.unwanted())
                self.lock.notify_all()
            logger.info(
                "script left · %s · dropped %d queued · %d still running",
                _peer(connection),
                dropped,
                running,
            )

    def _submit(self, script: _Script, message: SubmitMessage) -> None:
        with self.lock:
            clashes = [
                f"pool {key} already runs with {first}, not {asked}"
                for key, definition in message.pools.items()
                if key in self.pools
                and (first := self.pools[key].model_dump(include=_SETTINGS))
                != (asked := definition.model_dump(include=_SETTINGS))
            ]
            if not clashes:
                script.pools = message.pools
                for key, definition in message.pools.items():
                    if key not in self.pools:
                        self._start_pool(key, definition)
                queued = self.queue.submit(script, message)
                self.lock.notify_all()
        if clashes:
            hint = "; restart the manager to change them"
            script.send(ErrorMessage(reason="; ".join(clashes) + hint))
        elif queued:
            logger.info("queued %d specs from %s", queued, _peer(script.connection))

    def _start_pool(self, key: str, definition: Pool) -> None:
        logger.info("starting pool %s", key)
        match definition:
            case SlurmPool():
                if self.manager_dir is None:
                    raise RuntimeError("slurm pools need a shared furu manager")
                pool = self.slurm_pools[key] = SlurmWorkerPool(
                    url=self.url,
                    key=key,
                    definition=definition,
                    directory=self.manager_dir / "workers" / key.replace(":", "-"),
                )
                threads = [
                    threading.Thread(
                        target=self._scale,
                        args=(key, pool),
                        name="furu-slurm-worker-pool-scale",
                    )
                ]
            case LocalPool():
                # Threads that stay until the manager closes.
                threads = [
                    threading.Thread(
                        target=self._run_local_worker,
                        args=(key, f"w{index}"),
                        name=f"furu-local-worker-{index}",
                    )
                    for index in range(definition.max_workers)
                ]
            case _ as unreachable:
                assert_never(unreachable)
        self.pools[key] = definition
        self.threads += threads
        for thread in threads:
            thread.start()

    def _scale(self, key: str, pool: SlurmWorkerPool) -> None:
        """Size ``pool`` to its demand until the manager stops, then cancel its queued workers.

        A failed tick fails the scripts using the pool now, but the pool keeps
        its workers and retries, so it still serves the scripts that come later.
        """
        while True:
            try:
                messages = pool.scale(self.demand(key))
            except Exception as exc:
                logger.exception("scale tick failed; retrying on the next tick")
                messages = [
                    ErrorMessage(
                        reason=f"slurm worker pool scale tick failed: {_summary(exc)}"
                    )
                ]
            for message in messages:
                self.notify(key, message)
            with self.lock:
                if self.lock.wait_for(lambda: self.stopping, _SCALE_INTERVAL_S):
                    break
        pool.stop()

    def _run_local_worker(self, key: str, component: str) -> None:
        try:
            worker_loop(
                url=self.url,
                pool=key,
                idle_timeout=None,
                component=component,
                backend="local-thread",
            )
        except Exception as exc:  # noqa: BLE001 -- fault barrier: fail the runs instead
            get_logger(component).exception("local worker thread crashed")
            self.notify(
                key,
                ErrorMessage(reason=f"local worker thread crashed: {_summary(exc)}"),
            )

    def _cancel(self, script: _Script) -> None:
        with self.lock:
            dropped, workers = self.queue.cancel(script)
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

    def notify(self, pool: str, message: WarningMessage | ErrorMessage) -> None:
        """Tell every script that uses ``pool``, e.g. that its workers fail."""
        (logger.warning if isinstance(message, WarningMessage) else logger.error)(
            "%s · %s", pool, message.reason
        )
        with self.lock:
            scripts = [script for script in self.scripts if pool in script.pools]
        for script in scripts:
            script.send(message)

    def demand(self, pool: str) -> int:
        """Workers ``pool`` should have: busy ones plus jobs it could start."""
        with self.lock:
            max_workers = max(
                (
                    script.pools[pool].max_workers
                    for script in self.scripts
                    if pool in script.pools
                ),
                default=0,
            )
            return self.queue.demand(pool, max_workers)

    # --- workers ---------------------------------------------------------------

    def _serve_worker(self, connection: ServerConnection, hello: HelloMessage) -> None:
        worker = hello.worker
        with self.lock:
            known = hello.pool in self.pools
        if not known:
            logger.warning("worker %s belongs to no pool here; closing", worker)
            return
        logger.debug("worker connected · %s", worker)
        while (job := self._lease(connection, pool=hello.pool)) is not None:
            self._flush()
            logger.info("leased %s to %s", job.label, worker, extra={"path": job.log})
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
                if isinstance(result, JobCompletedResult) and (
                    pool := self.slurm_pools.get(hello.pool)
                ):
                    pool.job_completed()
                self.queue.finish(connection, result)
                self.lock.notify_all()
            self._flush()

    def _lease(self, connection: ServerConnection, *, pool: str) -> Job | None:
        with self.lock:
            while not self.stopping and connection.state is State.OPEN:
                if (job := self.queue.hand_out(connection, pool)) is not None:
                    return job
                self.lock.wait(timeout=_IDLE_CHECK_INTERVAL_S)
            return None


def _summary(exc: Exception) -> str:
    return traceback.format_exception_only(type(exc), exc)[-1].strip()


def _peer(connection: ServerConnection) -> str:
    with suppress(OSError, TypeError):
        host, port, *_ = connection.remote_address
        return f"{host}:{port}"
    return "?"
