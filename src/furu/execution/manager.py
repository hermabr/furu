from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import assert_never
from uuid import uuid4

from furu.core import Furu
from furu.dag import DagNode, _add_to_dag
from furu.logging import get_logger
from furu.metadata import ArtifactSpec
from furu.worker.protocol import (
    GetJobResponse,
    Job,
    JobBlockedResult,
    JobCompletedResult,
    JobFailedResult,
    JobResultRequest,
)


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
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        from furu.execution.server import _run_until_done

        _run_until_done(self, n_workers=n_workers, host=host, port=port)

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
                    self._block_job_locked(running_job.node, dependencies)
                case _:
                    assert_never(request)
            self._maybe_finish_locked()

    def _block_job_locked(
        self,
        node: DagNode,
        dependencies: Sequence[ArtifactSpec],
    ) -> None:
        dependency_ids: list[str] = []
        missing_dependency_ids: set[str] = set()
        missing_dependencies: list[Furu] = []
        for artifact in dependencies:
            if artifact.object_id in self.completed:
                continue

            if artifact.object_id in self.nodes_by_id:
                if self.nodes_by_id[artifact.object_id].obj.status() == "completed":
                    continue
                dependency_ids.append(artifact.object_id)
                continue

            if artifact.object_id not in missing_dependency_ids:
                dependency = Furu.from_artifact(artifact)
                if dependency.status() == "completed":
                    continue
                missing_dependency_ids.add(artifact.object_id)
                missing_dependencies.append(dependency)
            dependency_ids.append(artifact.object_id)

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
