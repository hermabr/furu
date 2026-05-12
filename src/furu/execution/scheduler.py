from __future__ import annotations

import socket
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, assert_never
from uuid import uuid4

from furu.core import Furu
from furu.dependencies import collect_declared_refs
from furu.logging import get_logger
from furu.metadata import ArtifactSpec
from furu.worker.protocol import (
    FinishFailedRequest,
    FinishRequest,
    FinishSuccessRequest,
    GetJobResponse,
    Job,
)


@dataclass(eq=False)
class DagNode[TFuru: Furu]:
    obj: TFuru
    dependencies: list[DagNode[TFuru]] = field(default_factory=list)
    dependents: list[DagNode[TFuru]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RunningJob:
    lease_id: str
    node: DagNode[Furu[Any]]


@dataclass(frozen=True, slots=True)
class FailedJob:
    lease_id: str
    node: DagNode[Furu[Any]]
    error: str


class Scheduler:
    def __init__(self, objs: Sequence[Furu[Any]]) -> None:
        self.nodes_by_id: dict[str, DagNode[Furu[Any]]] = {}
        self.ready: dict[str, DagNode[Furu[Any]]] = {}
        self.blocked: dict[str, DagNode[Furu[Any]]] = {}
        self.running: dict[str, RunningJob] = {}
        self.completed: dict[str, DagNode[Furu[Any]]] = {}
        self.failed: dict[str, FailedJob] = {}
        self.lock = threading.Lock()
        self.done = threading.Event()
        self._finish_error: str | None = None

        self._add_to_dag(objs)

    def get_job(self) -> GetJobResponse:
        with self.lock:
            self._maybe_finish_locked()
            if self.done.is_set():
                return "stop"
            if not self.ready:
                return "wait"

            object_id = next(iter(self.ready))
            node = self.ready.pop(object_id)
            lease_id = str(uuid4())
            if lease_id in self.running:
                raise RuntimeError(f"generated duplicate lease_id: {lease_id}")
            self.running[lease_id] = RunningJob(lease_id=lease_id, node=node)
            return Job(
                lease_id=lease_id,
                artifact=ArtifactSpec.from_furu(node.obj),
            )

    def finish(self, lease_id: str, request: FinishRequest) -> None:
        with self.lock:
            running_job = self._pop_running_locked(lease_id)
            match request:
                case FinishSuccessRequest():
                    self.completed[running_job.node.obj.object_id] = running_job.node
                    self._release_dependents_locked(running_job.node)
                case FinishFailedRequest(error=error):
                    self.failed[running_job.node.obj.object_id] = FailedJob(
                        lease_id=lease_id,
                        node=running_job.node,
                        error=error,
                    )
                case _:
                    assert_never(request)
            self._maybe_finish_locked()

    def block(self, lease_id: str, dependencies: Sequence[ArtifactSpec]) -> None:
        with self.lock:
            running_job = self._pop_running_locked(lease_id)
            node = running_job.node

            dependency_ids: list[str] = []
            missing_dependency_ids: set[str] = set()
            missing_dependencies: list[Furu[Any]] = []
            for artifact in dependencies:
                if artifact.object_id in self.completed:
                    continue

                dependency_ids.append(artifact.object_id)
                if (
                    artifact.object_id not in self.nodes_by_id
                    and artifact.object_id not in missing_dependency_ids
                ):
                    missing_dependency_ids.add(artifact.object_id)
                    missing_dependencies.append(Furu.from_artifact(artifact))

            self._add_to_dag(missing_dependencies)

            for dependency_id in dependency_ids:
                dep_node = self.nodes_by_id[dependency_id]
                if dep_node not in node.dependencies:
                    node.dependencies.append(dep_node)
                if node not in dep_node.dependents:
                    dep_node.dependents.append(node)

            if node.dependencies:
                self.blocked[node.obj.object_id] = node
            else:
                self.ready[node.obj.object_id] = node
            self._maybe_finish_locked()

    def _add_to_dag(self, objs: Sequence[Furu[Any]]) -> None:
        if any(not isinstance(obj, Furu) for obj in objs):
            # TODO: accept pytrees of Furu objects (e.g. nested lists/dicts/dataclasses)
            # and flatten them before walking dependencies.
            raise TypeError("expected Furu objects")

        refs_by_id: dict[str, tuple[Furu[Any], ...]] = {}
        newly_added: list[DagNode[Furu[Any]]] = []
        # TODO: detect cycles and raise a clear error
        pending = list(objs)

        while pending:
            obj = pending.pop()
            if obj.object_id in self.nodes_by_id:
                continue
            node = DagNode(obj=obj)
            self.nodes_by_id[obj.object_id] = node
            newly_added.append(node)
            if obj.status() == "completed":
                refs_by_id[obj.object_id] = ()
                continue
            refs = collect_declared_refs(obj)
            refs_by_id[obj.object_id] = refs
            pending.extend(refs)

        for obj_id, refs in refs_by_id.items():
            node = self.nodes_by_id[obj_id]
            for ref in refs:
                dep_node = self.nodes_by_id[ref.object_id]
                node.dependencies.append(dep_node)
                dep_node.dependents.append(node)

        for node in newly_added:
            target = self.ready if not node.dependencies else self.blocked
            target[node.obj.object_id] = node

    def run(
        self,
        *,
        n_workers: int = 1,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        if n_workers < 1:
            raise ValueError("n_workers must be at least 1")

        with self.lock:
            self._maybe_finish_locked()
        if self.done.is_set():
            self.raise_for_failure()
            return

        import uvicorn

        from furu.worker.api import create_scheduler_app
        from furu.worker.loop import worker_loop

        app = create_scheduler_app(self)
        sock = _bind_socket(host=host, port=port)
        bound_host, bound_port = sock.getsockname()[:2]
        server_url = f"http://{bound_host}:{bound_port}"

        server = uvicorn.Server(
            uvicorn.Config(
                app,
                log_level="warning",
                lifespan="off",
                ws="none",
            )
        )
        server_thread = threading.Thread(
            target=server.run,
            kwargs={"sockets": [sock]},
            name="furu-scheduler-server",
        )
        workers = [
            threading.Thread(
                target=worker_loop,
                kwargs={"server_url": server_url},
                name=f"furu-worker-{idx}",
            )
            for idx in range(n_workers)
        ]

        try:
            server_thread.start()
            _wait_for_server(server, server_thread)

            for worker in workers:
                worker.start()

            while not self.done.wait(timeout=0.1):
                if any(not worker.is_alive() for worker in workers):
                    self.fail("a worker exited before scheduler run completed")
                    break

            for worker in workers:
                worker.join(timeout=5)
        finally:
            server.should_exit = True
            server_thread.join(timeout=10)

        self.raise_for_failure()

    def raise_for_failure(self) -> None:
        if self._finish_error is not None:
            raise RuntimeError(self._finish_error)

    def fail(self, message: str) -> None:
        with self.lock:
            if self.done.is_set():
                return
            self._finish_error = message
            get_logger().error("furu scheduler finished with error: %s", message)
            self.done.set()

    def _pop_running_locked(self, lease_id: str) -> RunningJob:
        try:
            return self.running.pop(lease_id)
        except KeyError as exc:
            raise KeyError(f"unknown running lease_id: {lease_id}") from exc

    def _release_dependents_locked(self, node: DagNode[Furu[Any]]) -> None:
        for dependent in tuple(node.dependents):
            if node in dependent.dependencies:
                dependent.dependencies.remove(node)

            dependent_id = dependent.obj.object_id
            if not dependent.dependencies and dependent_id in self.blocked:
                self.ready[dependent_id] = self.blocked.pop(dependent_id)

    def _maybe_finish_locked(self) -> None:
        if self.done.is_set() or self.ready or self.running:
            return

        if self.failed or self.blocked:
            self._finish_error = self._format_finish_error_locked()
            get_logger().error(
                "furu scheduler finished with error: %s", self._finish_error
            )
        else:
            get_logger().info("furu scheduler finished successfully")
        self.done.set()

    def _format_finish_error_locked(self) -> str:
        parts: list[str] = []
        if self.failed:
            failed = ", ".join(sorted(self.failed))
            parts.append(f"failed jobs: {failed}")
        if self.blocked:
            blocked = ", ".join(sorted(self.blocked))
            parts.append(f"blocked jobs: {blocked}")
        if self.failed:
            first_object_id = next(iter(sorted(self.failed)))
            failed_job = self.failed[first_object_id]
            parts.append(
                f"first failure for {first_object_id} "
                f"(lease {failed_job.lease_id}): {failed_job.error}"
            )
        return "scheduler run could not complete; " + "; ".join(parts)


def _bind_socket(*, host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen()
    sock.set_inheritable(True)
    return sock


def _wait_for_server(server: Any, server_thread: threading.Thread) -> None:
    deadline = time.monotonic() + 10
    while not server.started:
        if not server_thread.is_alive():
            raise RuntimeError("scheduler server exited before it was ready")
        if time.monotonic() > deadline:
            raise TimeoutError("scheduler server did not start within 10 seconds")
        time.sleep(0.01)
