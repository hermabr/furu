from __future__ import annotations

import socket
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from furu.core import Furu
from furu.dag import FuruDagNode, make_execution_dag
from furu.logging import get_logger
from furu.metadata import ArtifactSpec
from furu.worker.protocol import FinishRequest, GetJobResponse, Job


@dataclass
class Manager:
    nodes_by_id: dict[str, FuruDagNode[Furu[Any]]] = field(default_factory=dict)
    ready: dict[str, FuruDagNode[Furu[Any]]] = field(default_factory=dict)
    blocked: dict[str, FuruDagNode[Furu[Any]]] = field(default_factory=dict)
    running: dict[str, FuruDagNode[Furu[Any]]] = field(default_factory=dict)
    completed: dict[str, FuruDagNode[Furu[Any]]] = field(default_factory=dict)
    failed: dict[str, FuruDagNode[Furu[Any]]] = field(default_factory=dict)
    failure_messages: dict[str, str] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    done: threading.Event = field(default_factory=threading.Event, repr=False)
    _finish_error: str | None = field(default=None, init=False, repr=False)

    @classmethod
    def submit(cls, objs: Sequence[Furu[Any]]) -> Manager:
        manager = cls()
        make_execution_dag(
            objs,
            manager.nodes_by_id,
            ready=manager.ready,
            blocked=manager.blocked,
        )
        with manager.lock:
            manager._maybe_finish_locked()
        return manager

    def get_job(self) -> GetJobResponse:
        with self.lock:
            self._maybe_finish_locked()
            if self.done.is_set():
                return "stop"
            if not self.ready:
                return "wait"

            lease_id = next(iter(self.ready))
            node = self.ready.pop(lease_id)
            self.running[lease_id] = node
            return Job(
                lease_id=lease_id,
                artifact=ArtifactSpec.from_furu(node.obj),
            )

    def finish(self, lease_id: str, request: FinishRequest) -> None:
        with self.lock:
            node = self._pop_running_locked(lease_id)
            if request.status == "completed":
                self.completed[lease_id] = node
                self._release_dependents_locked(node)
            else:
                self.failed[lease_id] = node
                if request.error is not None:
                    self.failure_messages[lease_id] = request.error
            self._maybe_finish_locked()

    def block(self, lease_id: str, dependencies: Sequence[ArtifactSpec]) -> None:
        with self.lock:
            node = self._pop_running_locked(lease_id)

            for artifact in dependencies:
                if artifact.object_id in self.completed:
                    continue

                if (dep_node := self.nodes_by_id.get(artifact.object_id)) is None:
                    dep_obj = Furu.from_artifact(artifact)
                    make_execution_dag(
                        [dep_obj],
                        self.nodes_by_id,
                        ready=self.ready,
                        blocked=self.blocked,
                    )
                    dep_node = self.nodes_by_id[artifact.object_id]

                if dep_node not in node.dependencies:
                    node.dependencies.append(dep_node)
                if node not in dep_node.dependents:
                    dep_node.dependents.append(node)

            if node.dependencies:
                self.blocked[lease_id] = node
            else:
                self.ready[lease_id] = node
            self._maybe_finish_locked()

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

        from furu.worker.api import create_manager_app
        from furu.worker.loop import worker_loop

        import uvicorn

        app = create_manager_app(self)
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
            name="furu-manager-server",
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
                    self.fail("a worker exited before submit() completed")
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
            get_logger().error("furu manager finished with error: %s", message)
            self.done.set()

    def _pop_running_locked(self, lease_id: str) -> FuruDagNode[Furu[Any]]:
        try:
            return self.running.pop(lease_id)
        except KeyError as exc:
            raise KeyError(f"unknown running lease_id: {lease_id}") from exc

    def _release_dependents_locked(self, node: FuruDagNode[Furu[Any]]) -> None:
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
                "furu manager finished with error: %s", self._finish_error
            )
        else:
            get_logger().info("furu manager finished successfully")
        self.done.set()

    def _format_finish_error_locked(self) -> str:
        parts: list[str] = []
        if self.failed:
            failed = ", ".join(sorted(self.failed))
            parts.append(f"failed jobs: {failed}")
        if self.blocked:
            blocked = ", ".join(sorted(self.blocked))
            parts.append(f"blocked jobs: {blocked}")
        if self.failure_messages:
            first_lease_id = next(iter(sorted(self.failure_messages)))
            parts.append(
                f"first failure for {first_lease_id}: "
                f"{self.failure_messages[first_lease_id]}"
            )
        return "submit() could not complete; " + "; ".join(parts)


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
            raise RuntimeError("manager server exited before it was ready")
        if time.monotonic() > deadline:
            raise TimeoutError("manager server did not start within 10 seconds")
        time.sleep(0.01)
