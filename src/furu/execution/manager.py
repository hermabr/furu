from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import assert_never
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


@dataclass(frozen=True, slots=True)
class RunningJob:
    lease_id: str
    node: DagNode
    leased_at: float


@dataclass(frozen=True, slots=True)
class FailedJob:
    lease_id: str
    node: DagNode
    error: str


class Manager:
    def __init__(
        self,
        objs: Sequence[Furu],
        *,
        lease_timeout: float | None = None,
    ) -> None:
        if not objs:
            raise ValueError("Manager requires at least one Furu object")
        if lease_timeout is not None and lease_timeout <= 0:
            raise ValueError("lease_timeout must be positive")

        self.nodes_by_id: dict[str, DagNode] = {}
        self.ready: dict[str, DagNode] = {}
        self.blocked: dict[str, DagNode] = {}
        self.running: dict[str, RunningJob] = {}
        self.completed: dict[str, DagNode] = {}
        self.failed: dict[str, FailedJob] = {}
        self.lock = threading.Lock()
        self.done = threading.Event()
        self._finish_error: str | None = None
        self._lease_timeout = lease_timeout

        _add_to_dag(self, objs)

    def run(
        self,
        *,
        n_workers: int = 1,
        host: str = "127.0.0.1",
        port: int = 0,
        advertise_host: str | None = None,
    ) -> None:
        from furu.execution.server import _run_until_done

        _run_until_done(
            self,
            n_workers=n_workers,
            bind_host=host,
            port=port,
            advertise_host=advertise_host,
        )

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
            self.running[lease_id] = RunningJob(
                lease_id=lease_id,
                node=node,
                leased_at=time.monotonic(),
            )
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
            self._fail_locked(message)

    def expire_old_leases(self, *, now: float | None = None) -> None:
        if self._lease_timeout is None:
            return
        checked_at = time.monotonic() if now is None else now
        with self.lock:
            if self.done.is_set():
                return
            expired = [
                running_job
                for running_job in self.running.values()
                if checked_at - running_job.leased_at >= self._lease_timeout
            ]
            if not expired:
                return
            expired.sort(key=lambda running_job: running_job.leased_at)
            first = expired[0]
            self._fail_locked(
                f"lease {first.lease_id} for {first.node.obj.object_id} "
                f"expired after {self._lease_timeout:g} seconds"
            )

    def _fail_locked(self, message: str) -> None:
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
