from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, assert_never
from uuid import uuid4

from furu.core import Furu
from furu.dag import DagNode, _add_to_dag, _update_dag_blocking_dependencies
from furu.logging import get_logger
from furu.metadata import ArtifactSpec
from furu.worker.protocol import (
    LeaseJobResponse,
    Job,
    JobBlockedResult,
    JobCompletedResult,
    JobFailedResult,
    JobResultRequest,
)

if TYPE_CHECKING:
    from furu.worker.backend import WorkerBackend


@dataclass(frozen=True, slots=True)
class RunningJob:
    lease_id: str
    node: DagNode


@dataclass(frozen=True, slots=True)
class FailedJob:
    lease_id: str
    node: DagNode
    error: str


class Manager:
    def __init__(self, objs: Sequence[Furu]) -> None:
        if not objs:
            raise ValueError("Manager requires at least one Furu object")

        self.nodes_by_id: dict[str, DagNode] = {}
        self.ready: dict[str, DagNode] = {}
        self.blocked: dict[str, DagNode] = {}
        self.running: dict[str, RunningJob] = {}
        self.completed: dict[str, DagNode] = {}
        self.failed: dict[str, FailedJob] = {}
        self.lock = threading.Lock()
        self.done = threading.Event()
        self._finish_error: str | None = None

        _add_to_dag(self, objs)

    def run(
        self,
        *,
        n_workers: int = 1,
        worker_backend: WorkerBackend | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        from furu.execution.server import _run_until_done
        from furu.worker.backend import LocalThreadWorkerBackend

        if worker_backend is not None and n_workers != 1:
            raise ValueError("pass either worker_backend or n_workers, not both")

        backend = worker_backend or LocalThreadWorkerBackend(n_workers=n_workers)
        _run_until_done(self, worker_backend=backend, host=host, port=port)

    def lease_job(self) -> LeaseJobResponse:
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

    def job_result(self, lease_id: str, request: JobResultRequest) -> None:
        with self.lock:
            running_job = self.running.pop(lease_id)
            match request:
                case JobCompletedResult():
                    self.completed[running_job.node.obj.object_id] = running_job.node
                    for dependent in tuple(running_job.node.dependents):
                        if running_job.node in dependent.dependencies:
                            dependent.dependencies.remove(running_job.node)

                        dependent_id = dependent.obj.object_id
                        if not dependent.dependencies and dependent_id in self.blocked:
                            self.ready[dependent_id] = self.blocked.pop(dependent_id)

                case JobFailedResult(error=error):
                    self.failed[running_job.node.obj.object_id] = FailedJob(
                        lease_id=lease_id,
                        node=running_job.node,
                        error=error,
                    )
                case JobBlockedResult(dependencies=dependencies):
                    _update_dag_blocking_dependencies(
                        self, running_job.node, dependencies
                    )
                case _:
                    assert_never(request)
            self._maybe_finish_locked()

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

    def _maybe_finish_locked(self) -> None:
        if self.done.is_set() or self.ready or self.running:
            return

        if self.failed or self.blocked:
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
            self._finish_error = "manager run could not complete; " + "; ".join(parts)
            get_logger().error(
                "furu manager finished with error: %s", self._finish_error
            )
        else:
            get_logger().info("furu manager finished successfully")
        self.done.set()
