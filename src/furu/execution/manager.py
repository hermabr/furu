from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, assert_never
from uuid import uuid4

from furu._storage_layout import manager_log_path_in
from furu.config import get_config
from furu.core import Furu
from furu.dag import (
    DagNode,
    _add_to_dag,
    _sync_declared_refs,
    _update_dag_blocking_dependencies,
)
from furu.dependencies import declared_dependency_recheck_due
from furu.logging import _scoped_log_files, get_logger
from furu.metadata import ArtifactSpec
from furu.resources import ResourceRequest, resource_request_satisfies
from furu.worker.protocol import (
    Job,
    JobBlockedResult,
    JobCompletedResult,
    JobFailedResult,
    JobResultRequest,
    LeaseJobResponse,
)

if TYPE_CHECKING:
    from furu.worker.backends import WorkerBackend


logger = get_logger()


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

        digest = hashlib.blake2s(digest_size=16)
        for obj in objs:
            digest.update(obj.object_id.encode("utf-8"))
            digest.update(b"\0")
        self.executor_id = digest.hexdigest()

    @property
    def executor_dir(self) -> Path:
        return get_config().directories.executions / self.executor_id

    def run(
        self,
        *,
        worker_backends: tuple[WorkerBackend, ...],
        port: int = 0,
    ) -> None:
        from furu.execution.server import _run_until_done

        _run_until_done(
            self,
            worker_backends=worker_backends,
            port=port,
        )

    @contextmanager
    def log_context(self) -> Iterator[None]:
        with _scoped_log_files((manager_log_path_in(self.executor_dir),)):
            yield

    def lease_job(self, *, resources: ResourceRequest) -> LeaseJobResponse:
        with self.log_context(), self.lock:
            self._maybe_recheck_declared_dependencies_locked()
            self._maybe_finish_locked()
            if self.done.is_set():
                return "stop"
            if not self.ready:
                return "wait"

            object_id = next(
                (
                    object_id
                    for object_id, node in self.ready.items()
                    if resource_request_satisfies(
                        resources, node.obj.resource_requirements
                    )
                ),
                None,
            )
            if object_id is None:
                return "wait"

            node = self.ready.pop(object_id)
            lease_id = str(uuid4())
            if lease_id in self.running:
                raise RuntimeError(f"generated duplicate lease_id: {lease_id}")
            self.running[lease_id] = RunningJob(lease_id=lease_id, node=node)
            logger.debug(
                "leased job: lease_id=%s object_id=%s ready=%d running=%d blocked=%d completed=%d failed=%d",
                lease_id,
                node.obj.object_id,
                len(self.ready),
                len(self.running),
                len(self.blocked),
                len(self.completed),
                len(self.failed),
            )
            return Job(
                lease_id=lease_id,
                artifact=ArtifactSpec.from_furu(node.obj),
            )

    def count_satisfiable_jobs(
        self, *, resources: ResourceRequest, max_workers: int
    ) -> int:
        with self.lock:
            count = 0
            for node in self.ready.values():
                if resource_request_satisfies(
                    resources, node.obj.resource_requirements
                ):
                    count += 1
                    if count >= max_workers:
                        return max_workers
            return count

    def job_result(self, lease_id: str, request: JobResultRequest) -> None:
        with self.log_context(), self.lock:
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
                    logger.debug(
                        "job completed: lease_id=%s object_id=%s ready=%d running=%d blocked=%d completed=%d failed=%d",
                        lease_id,
                        running_job.node.obj.object_id,
                        len(self.ready),
                        len(self.running),
                        len(self.blocked),
                        len(self.completed),
                        len(self.failed),
                    )

                case JobFailedResult(error=error):
                    self.failed[running_job.node.obj.object_id] = FailedJob(
                        lease_id=lease_id,
                        node=running_job.node,
                        error=error,
                    )
                    logger.debug(
                        "job failed: lease_id=%s object_id=%s error=%s",
                        lease_id,
                        running_job.node.obj.object_id,
                        error,
                    )
                case JobBlockedResult(dependencies=dependencies):
                    _update_dag_blocking_dependencies(
                        self, running_job.node, dependencies
                    )
                    logger.debug(
                        "job blocked: lease_id=%s object_id=%s dependencies=%d ready=%d running=%d blocked=%d completed=%d failed=%d",
                        lease_id,
                        running_job.node.obj.object_id,
                        len(dependencies),
                        len(self.ready),
                        len(self.running),
                        len(self.blocked),
                        len(self.completed),
                        len(self.failed),
                    )
                case _:
                    assert_never(request)
            self._maybe_recheck_declared_dependencies_locked()
            self._maybe_finish_locked()

    def raise_for_failure(self) -> None:
        if self._finish_error is not None:
            raise RuntimeError(self._finish_error)

    def fail(self, message: str) -> None:
        with self.log_context(), self.lock:
            if self.done.is_set():
                return
            self._finish_error = message
            logger.error("furu manager finished with error: %s", message)
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
            logger.error("furu manager finished with error: %s", self._finish_error)
        else:
            logger.info("furu manager finished successfully")
        self.done.set()

    def _maybe_recheck_declared_dependencies_locked(self) -> None:
        for node in tuple(self.blocked.values()) + tuple(self.ready.values()):
            if declared_dependency_recheck_due(node.obj):
                _sync_declared_refs(self, node)
