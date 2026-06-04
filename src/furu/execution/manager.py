from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, assert_never
from uuid import uuid4

from furu._storage_layout import manager_log_path_in
from furu.config import get_config
from furu.core import Furu
from furu.dag import DagNode, _add_to_dag, _update_dag_blocking_dependencies
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
    failed_attempts: int
    lease_id: str
    node: DagNode
    error: str


@dataclass(slots=True, kw_only=True)
class Manager:
    max_retries_per_object: int
    executor_id: str = "not-computed-yet"
    nodes_by_id: dict[str, DagNode] = field(default_factory=dict)
    ready: dict[str, DagNode] = field(default_factory=dict)
    blocked: dict[str, DagNode] = field(default_factory=dict)
    running: dict[str, RunningJob] = field(default_factory=dict)
    completed: dict[str, DagNode] = field(default_factory=dict)
    failed: dict[str, FailedJob] = field(default_factory=dict)
    lock: Any = field(default_factory=threading.Lock)
    done: threading.Event = field(default_factory=threading.Event)
    finish_error: str | None = None

    @classmethod
    def run[ObjsT: Sequence[Furu]](
        cls,
        objs: ObjsT,
        *,
        max_retries_per_object: int | None = None,
        worker_backends: tuple[WorkerBackend, ...],
        port: int = 0,
    ) -> ObjsT:
        if max_retries_per_object is None:
            max_retries_per_object = get_config().worker.max_retries_per_object
        manager = cls(max_retries_per_object=max_retries_per_object)
        _add_to_dag(manager, objs)
        digest = hashlib.blake2s(digest_size=16)
        for obj in objs:
            digest.update(obj.object_id.encode("utf-8"))
            digest.update(b"\0")
        manager.executor_id = digest.hexdigest()

        if not manager.nodes_by_id:
            with manager.log_context(), manager.lock:
                logger.info("all objects already exist; no manager work to run")
                manager._maybe_finish_locked()
            return objs

        bind_hosts = {backend.manager_listen_host for backend in worker_backends}
        if not bind_hosts:
            raise ValueError("worker_backends must include at least one backend")
        if len(bind_hosts) != 1:
            raise ValueError(
                "all worker_backends must use the same manager_listen_host"
            )
        (bind_host,) = bind_hosts

        from furu.execution.server import manager_server

        with manager.log_context():
            logger.info(
                "starting furu manager: executor_id=%s executor_dir=%s ready=%d blocked=%d",
                manager.executor_id,
                manager.executor_dir,
                len(manager.ready),
                len(manager.blocked),
            )
            with manager_server(manager, bind_host=bind_host, port=port) as server:
                logger.info(
                    "manager server listening: server_url=%s",
                    server.server_url,
                )
                pools = []
                for backend in worker_backends:
                    pool = backend.start_pool(
                        server_url=server.server_url,
                        auth_token=server.auth_token,
                        executor_dir=manager.executor_dir,
                    )
                    pools.append(pool)
                    logger.info(
                        "worker pool started: backend=%s", type(backend).__name__
                    )
                manager.done.wait()

                with ThreadPoolExecutor(max_workers=len(pools)) as executor:
                    for pool in pools:
                        executor.submit(pool.stop, timeout=5)
        manager.raise_for_failure()
        return objs

    @property
    def executor_dir(self) -> Path:
        return get_config().directories.executions / self.executor_id

    @contextmanager
    def log_context(self) -> Iterator[None]:
        with _scoped_log_files((manager_log_path_in(self.executor_dir),)):
            yield

    def lease_job(self, *, resources: ResourceRequest) -> LeaseJobResponse:
        with self.log_context(), self.lock:
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
            node.obj.logger.info(
                "executor creating %s (object_id=%s)",
                node.obj._log_label,
                node.obj.object_id,
            )
            logger.info(
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
                    self.failed.pop(running_job.node.obj.object_id, None)
                    self.completed[running_job.node.obj.object_id] = running_job.node
                    for dependent in tuple(running_job.node.dependents):
                        if running_job.node in dependent.dependencies:
                            dependent.dependencies.remove(running_job.node)

                        dependent_id = dependent.obj.object_id
                        if not dependent.dependencies and dependent_id in self.blocked:
                            self.ready[dependent_id] = self.blocked.pop(dependent_id)
                    logger.info(
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
                    object_id = running_job.node.obj.object_id
                    previous_failed = self.failed.get(object_id)
                    failed_attempts = (
                        previous_failed.failed_attempts if previous_failed else 0
                    ) + 1
                    self.failed[object_id] = FailedJob(
                        failed_attempts=failed_attempts,
                        lease_id=lease_id,
                        node=running_job.node,
                        error=error,
                    )
                    if failed_attempts <= self.max_retries_per_object:
                        self.ready[object_id] = running_job.node
                    logger.info(
                        "job failed: lease_id=%s object_id=%s failed_attempts=%d max_retries=%d error=%s",
                        lease_id,
                        object_id,
                        failed_attempts,
                        self.max_retries_per_object,
                        error,
                    )
                case JobBlockedResult(dependencies=dependencies):
                    _update_dag_blocking_dependencies(
                        self, running_job.node, dependencies
                    )
                    logger.info(
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
            logger.info(
                "manager progress: completed=%d/%d failed=%d running=%d ready=%d blocked=%d",
                len(self.completed),
                len(self.nodes_by_id),
                len(self.failed),
                len(self.running),
                len(self.ready),
                len(self.blocked),
            )
            self._maybe_finish_locked()

    def raise_for_failure(self) -> None:
        if self.finish_error is not None:
            raise RuntimeError(self.finish_error)

    def fail(self, message: str) -> None:
        with self.log_context(), self.lock:
            if self.done.is_set():
                return
            self.finish_error = message
            logger.error("furu manager finished with error: %s", message)
            self.done.set()

    def _maybe_finish_locked(self) -> None:
        if self.done.is_set() or self.ready or self.running:
            return

        terminal_failed = {
            object_id: record
            for object_id, record in self.failed.items()
            if record.failed_attempts > self.max_retries_per_object
        }

        if terminal_failed or self.blocked:
            parts: list[str] = []
            if terminal_failed:
                failed = ", ".join(sorted(terminal_failed))
                parts.append(f"failed jobs: {failed}")
            if self.blocked:
                blocked = ", ".join(sorted(self.blocked))
                parts.append(f"blocked jobs: {blocked}")
            if terminal_failed:
                first_object_id = next(iter(sorted(terminal_failed)))
                failed_job = terminal_failed[first_object_id]
                parts.append(
                    f"first failure for {first_object_id} "
                    f"after {failed_job.failed_attempts} failed attempts "
                    f"(lease {failed_job.lease_id}): {failed_job.error}"
                )
            self.finish_error = "manager run could not complete; " + "; ".join(parts)
            logger.error("furu manager finished with error: %s", self.finish_error)
        else:
            logger.info("furu manager finished successfully")
        self.done.set()
