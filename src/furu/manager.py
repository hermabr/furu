from __future__ import annotations

import logging
import socket
import threading
import time
import traceback
import uuid
from _thread import LockType, allocate_lock
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from fastapi import FastAPI, HTTPException

from furu.core import Furu
from furu.dag import FuruDagNode, make_execution_dag
from furu.logging import get_logger
from furu.metadata import ArtifactSpec
from furu.worker_execution import BlockedUpdate, FinishUpdate, Job, worker_loop

type DagNode = FuruDagNode[Furu[Any]]
type DagNodeMap = dict[str, DagNode]


@dataclass(frozen=True)
class RunningJob:
    node: DagNode


@dataclass(frozen=True)
class FailedJob:
    node: DagNode
    error: str | None


@dataclass
class Manager:
    nodes_by_id: DagNodeMap = field(default_factory=dict)
    ready: DagNodeMap = field(default_factory=dict)
    blocked: DagNodeMap = field(default_factory=dict)
    running: dict[str, RunningJob] = field(default_factory=dict)
    completed: DagNodeMap = field(default_factory=dict)
    failed: dict[str, FailedJob] = field(default_factory=dict)
    worker_errors: list[str] = field(default_factory=list)
    lock: LockType = field(default_factory=allocate_lock, init=False, repr=False)
    done_event: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    logger: logging.Logger = field(
        default_factory=lambda: get_logger("manager"), init=False, repr=False
    )
    _finished: bool = field(default=False, init=False, repr=False)

    @classmethod
    def from_objects(cls, objs: Sequence[Furu[Any]]) -> Manager:
        manager = cls()
        make_execution_dag(
            objs,
            manager.nodes_by_id,
            ready=manager.ready,
            blocked=manager.blocked,
        )
        return manager

    @classmethod
    def submit(cls, objs: Sequence[Furu[Any]], *, n_workers: int = 1) -> None:
        manager = cls.from_objects(objs)
        manager.run(n_workers=n_workers)

    def run(self, *, n_workers: int = 1) -> None:
        if n_workers < 1:
            raise ValueError("n_workers must be at least 1")
        if not self.nodes_by_id:
            self.logger.info("submit finished successfully: no jobs")
            return

        server = start_manager_server(self)
        remaining_workers = n_workers

        def run_worker() -> None:
            nonlocal remaining_workers
            worker_error: str | None = None
            try:
                worker_loop(server.url)
            except BaseException as exc:
                worker_error = "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                )
            finally:
                with self.lock:
                    remaining_workers -= 1
                    if worker_error is not None:
                        self.worker_errors.append(worker_error)
                    if remaining_workers == 0 and not self._finished:
                        self.worker_errors.append(
                            "all workers exited before submit completed"
                        )
                        self._fail_active_jobs_locked(
                            "all workers exited before submit completed"
                        )
                        self._finished = True
                        self.logger.error(
                            "submit stopped because all workers exited before completion"
                        )
                        self.done_event.set()

        workers = [
            threading.Thread(
                target=run_worker,
                name=f"furu-worker-{i}",
            )
            for i in range(n_workers)
        ]

        try:
            for worker in workers:
                worker.start()
            self.done_event.wait()
            for worker in workers:
                worker.join()
        finally:
            server.stop()

        if error := self.completion_error():
            raise error

    def get_job(self) -> Job | Literal["wait", "stop"]:
        with self.lock:
            if self._finished:
                return "stop"

            if self.ready:
                object_id = next(iter(self.ready))
                node = self.ready.pop(object_id)
                lease_id = uuid.uuid4().hex
                self.running[lease_id] = RunningJob(node=node)
                return Job(lease_id=lease_id, artifact=ArtifactSpec.from_furu(node.obj))

            if self.running:
                return "wait"

            self._mark_finished_if_drained_locked()
            return "stop"

    def finish_job(self, lease_id: str, update: FinishUpdate) -> None:
        with self.lock:
            lease = self._pop_lease_locked(lease_id)
            node = lease.node
            object_id = node.obj.object_id

            if update.status == "failed":
                self.failed[object_id] = FailedJob(node=node, error=update.error)
            else:
                self.completed[object_id] = node
                self._release_dependents_locked(node)

            self._mark_finished_if_drained_locked()

    def block_job(self, lease_id: str, update: BlockedUpdate) -> None:
        from furu.core import Furu

        dependency_objs = tuple(Furu.from_artifact(dep) for dep in update.dependencies)

        with self.lock:
            lease = self._pop_lease_locked(lease_id)
            node = lease.node
            object_id = node.obj.object_id

            make_execution_dag(
                dependency_objs,
                self.nodes_by_id,
                ready=self.ready,
                blocked=self.blocked,
            )

            self.ready.pop(object_id, None)
            self.blocked.pop(object_id, None)

            for dependency in dependency_objs:
                dependency_id = dependency.object_id
                dependency_node = self.nodes_by_id[dependency_id]
                if (
                    dependency_id in self.completed
                    or dependency_node.obj.status() == "completed"
                ):
                    self.ready.pop(dependency_id, None)
                    self.blocked.pop(dependency_id, None)
                    self.completed.setdefault(dependency_id, dependency_node)
                    continue
                if dependency_node not in node.dependencies:
                    node.dependencies.append(dependency_node)
                if node not in dependency_node.dependents:
                    dependency_node.dependents.append(node)

            if node.dependencies:
                self.blocked[object_id] = node
            else:
                self.ready[object_id] = node

            self._mark_finished_if_drained_locked()

    def completion_error(self) -> RuntimeError | None:
        with self.lock:
            if self.worker_errors:
                return RuntimeError(
                    "submit() worker failure: " + self.worker_errors[0].strip()
                )
            if self.failed:
                failed = ", ".join(sorted(self.failed))
                message = f"submit() failed; failed jobs: {failed}"
                if self.blocked:
                    blocked = ", ".join(sorted(self.blocked))
                    message += f"; blocked jobs: {blocked}"
                return RuntimeError(message)
            if self.blocked:
                unresolved = ", ".join(sorted(self.blocked))
                return RuntimeError(
                    "submit() could not make progress; "
                    f"unresolved dependencies: {unresolved}"
                )
            return None

    def _fail_active_jobs_locked(self, error: str) -> None:
        for lease in list(self.running):
            node = self.running.pop(lease).node
            self.failed[node.obj.object_id] = FailedJob(node=node, error=error)

    def _pop_lease_locked(self, lease_id: str) -> RunningJob:
        try:
            return self.running.pop(lease_id)
        except KeyError as exc:
            raise KeyError(f"unknown lease_id: {lease_id}") from exc

    def _release_dependents_locked(self, node: DagNode) -> None:
        for dependent in tuple(node.dependents):
            if node in dependent.dependencies:
                dependent.dependencies.remove(node)
            dependent_id = dependent.obj.object_id
            if not dependent.dependencies and dependent_id in self.blocked:
                self.blocked.pop(dependent_id)
                if (
                    dependent_id not in self.completed
                    and dependent_id not in self.failed
                ):
                    self.ready[dependent_id] = dependent

    def _mark_finished_if_drained_locked(self) -> None:
        if self._finished or self.ready or self.running:
            return

        self._finished = True
        if self.failed:
            self.logger.error("submit finished with %d failed job(s)", len(self.failed))
        elif self.blocked:
            self.logger.error(
                "submit finished with %d blocked job(s)", len(self.blocked)
            )
        else:
            self.logger.info("submit finished successfully")
        self.done_event.set()


def create_manager_app(manager: Manager) -> FastAPI:
    app = FastAPI()

    @app.get("/get_job")
    def get_job() -> Job | Literal["wait", "stop"]:
        return manager.get_job()

    @app.post("/finish/{lease_id}")
    def finish_job(lease_id: str, update: FinishUpdate) -> dict[str, str]:
        try:
            manager.finish_job(lease_id, update)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"status": "ok"}

    @app.post("/blocked/{lease_id}")
    def block_job(lease_id: str, update: BlockedUpdate) -> dict[str, str]:
        try:
            manager.block_job(lease_id, update)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"status": "ok"}

    return app


@dataclass(frozen=True)
class ManagerServer:
    url: str
    server: Any
    thread: threading.Thread

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)


def _pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def start_manager_server(
    manager: Manager,
    *,
    host: str = "127.0.0.1",
    startup_timeout_seconds: float = 10.0,
) -> ManagerServer:
    import uvicorn

    port = _pick_free_port(host)
    config = uvicorn.Config(
        create_manager_app(manager),
        host=host,
        port=port,
        lifespan="off",
        log_level="warning",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(
        target=server.run,
        name="furu-manager-server",
        daemon=True,
    )
    thread.start()

    deadline = time.monotonic() + startup_timeout_seconds
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("manager server failed to start")
        if time.monotonic() >= deadline:
            server.should_exit = True
            raise TimeoutError("manager server did not start in time")
        time.sleep(0.01)

    return ManagerServer(
        url=f"http://{host}:{port}",
        server=server,
        thread=thread,
    )
