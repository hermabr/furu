from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, assert_never
from uuid import uuid4

from furu.core import Furu
from furu.dag import DagNode, _add_to_dag
from furu.logging import get_logger
from furu.metadata import ArtifactSpec
from furu.worker.protocol import (
    FinishFailedRequest,
    FinishRequest,
    FinishSuccessRequest,
    GetJobResponse,
    Job,
)


@dataclass(frozen=True, slots=True)
class RunningJob:
    lease_id: str
    node: DagNode[Furu[Any]]


@dataclass(frozen=True, slots=True)
class FailedJob:
    lease_id: str
    node: DagNode[Furu[Any]]
    error: str


class Manager:
    def __init__(self, objs: Sequence[Furu[Any]]) -> None:
        if not objs:
            raise ValueError("Manager requires at least one Furu object")

        self.nodes_by_id: dict[str, DagNode[Furu[Any]]] = {}
        self.ready: dict[str, DagNode[Furu[Any]]] = {}
        self.blocked: dict[str, DagNode[Furu[Any]]] = {}
        self.running: dict[str, RunningJob] = {}
        self.completed: dict[str, DagNode[Furu[Any]]] = {}
        self.failed: dict[str, FailedJob] = {}
        self.lock = threading.Lock()
        self.done = threading.Event()
        self._finish_error: str | None = None

        _add_to_dag(self, objs)

    def run(
        self,
        *,
        n_workers: int = 1,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        from furu.execution.server import run_until_done

        run_until_done(self, n_workers=n_workers, host=host, port=port)

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

    def report_blocked(
        self,
        lease_id: str,
        dependencies: Sequence[ArtifactSpec],
    ) -> None:
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

            _add_to_dag(self, missing_dependencies)

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

    def unresolved_object_ids(self) -> list[str]:
        with self.lock:
            return sorted(self.blocked)

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
        if self.failed:
            first_object_id = next(iter(sorted(self.failed)))
            failed_job = self.failed[first_object_id]
            parts.append(
                f"first failure for {first_object_id} "
                f"(lease {failed_job.lease_id}): {failed_job.error}"
            )
        return "manager run could not complete; " + "; ".join(parts)
