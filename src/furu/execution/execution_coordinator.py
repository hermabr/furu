from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, assert_never
from uuid import uuid4

from furu._storage_layout import execution_coordinator_log_path_in
from furu.config import get_config
from furu.core import Furu
from furu.dag import DagNode, _add_to_dag, _update_dag_blocking_dependencies
from furu.logging import (
    _fmt_duration,
    _scoped_log_files,
    get_logger,
    log_component,
    log_event,
)
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


def _exception_summary(error: str) -> str:
    """The last non-empty line of a traceback string — the exception itself."""

    lines = [line for line in error.strip().splitlines() if line.strip()]
    return lines[-1] if lines else error.strip()


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
class ExecutionCoordinator:
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
    started_monotonic: float = 0.0

    def _failed_counts(self) -> tuple[int, int]:
        failed_retry = sum(
            record.failed_attempts <= self.max_retries_per_object
            for record in self.failed.values()
        )
        return failed_retry, len(self.failed) - failed_retry

    @classmethod
    def run[ObjsT: Sequence[Furu]](
        cls,
        objs: ObjsT,  # TODO: support pytrees
        *,
        max_retries_per_object: int | None = None,
        worker_backends: tuple[WorkerBackend, ...],
        port: int = 0,
    ) -> ObjsT:
        if max_retries_per_object is None:
            max_retries_per_object = get_config().worker.max_retries_per_object
        coordinator = cls(max_retries_per_object=max_retries_per_object)
        _add_to_dag(coordinator, objs)
        digest = hashlib.blake2s(digest_size=16)
        for obj in objs:
            digest.update(obj.object_id.encode("utf-8"))
            digest.update(b"\0")
        coordinator.executor_id = digest.hexdigest()
        coordinator.started_monotonic = time.monotonic()

        if not coordinator.nodes_by_id:
            with coordinator.log_context(), coordinator.lock:
                logger.info(
                    "all objects already exist; no execution coordinator work to run"
                )
                coordinator._maybe_finish_locked()
            return objs

        (bind_host,) = {
            backend.execution_coordinator_listen_host for backend in worker_backends
        }

        from furu.execution.server import execution_coordinator_server

        with coordinator.log_context():
            log_event(
                logger,
                logging.INFO,
                "starting furu execution coordinator",
                event="starting",
                detail=(
                    f"exec={coordinator.executor_id[:5]} · "
                    f"{len(coordinator.ready)} ready · {len(coordinator.blocked)} blocked"
                ),
                executor_id=coordinator.executor_id,
                executor_dir=str(coordinator.executor_dir),
                ready=len(coordinator.ready),
                blocked=len(coordinator.blocked),
            )
            with execution_coordinator_server(
                coordinator, bind_host=bind_host, port=port
            ) as server:
                log_event(
                    logger,
                    logging.INFO,
                    "execution coordinator server listening",
                    event="listening",
                    detail=server.server_url,
                    server_url=server.server_url,
                )
                pools = []
                for backend in worker_backends:
                    pool = backend.start_pool(
                        bound_port=server.bound_port,
                        auth_token=server.auth_token,
                        executor_dir=coordinator.executor_dir,
                    )
                    pools.append(pool)
                    log_event(
                        logger,
                        logging.INFO,
                        "worker pool started",
                        event="pool up",
                        detail=type(backend).__name__,
                        backend=type(backend).__name__,
                    )
                coordinator.done.wait()

                with ThreadPoolExecutor(max_workers=len(pools)) as executor:
                    stop_futures = [
                        executor.submit(pool.stop, timeout=5) for pool in pools
                    ]
                for pool, future in zip(pools, stop_futures, strict=True):
                    if (exc := future.exception()) is not None:
                        log_event(
                            logger,
                            logging.ERROR,
                            "worker pool stop failed",
                            event="pool stop failed",
                            detail=f"{type(pool).__name__} · {exc}",
                            status="error",
                            pool=type(pool).__name__,
                            error=str(exc),
                        )
        coordinator.raise_for_failure()
        return objs

    @property
    def executor_dir(self) -> Path:
        return get_config().run_directories.executions / self.executor_id

    @contextmanager
    def log_context(self) -> Iterator[None]:
        with (
            log_component("coord"),
            _scoped_log_files((execution_coordinator_log_path_in(self.executor_dir),)),
        ):
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
            self.running[lease_id] = RunningJob(lease_id=lease_id, node=node)
            failed_retry, failed = self._failed_counts()
            log_event(
                logger,
                logging.INFO,
                "leased job",
                event="leased",
                label=node.obj._log_label,
                lease=lease_id,
                object_id=node.obj.object_id,
                ready=len(self.ready),
                running=len(self.running),
                blocked=len(self.blocked),
                completed=len(self.completed),
                failed_retry=failed_retry,
                failed=failed,
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
            running_job = self.running.pop(lease_id, None)
            if running_job is None:
                logger.info(
                    "ignoring job result for unknown lease: lease_id=%s", lease_id
                )
                return
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
                    failed_retry, failed = self._failed_counts()
                    log_event(
                        logger,
                        logging.DEBUG,
                        "job completed",
                        event="completed",
                        label=running_job.node.obj._log_label,
                        lease=lease_id,
                        object_id=running_job.node.obj.object_id,
                        ready=len(self.ready),
                        running=len(self.running),
                        blocked=len(self.blocked),
                        completed=len(self.completed),
                        failed_retry=failed_retry,
                        failed=failed,
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
                    will_retry = failed_attempts <= self.max_retries_per_object
                    if will_retry:
                        self.ready[object_id] = running_job.node
                    log_event(
                        logger,
                        logging.INFO,
                        "job failed (retry)" if will_retry else "job failed",
                        event="failed (retry)" if will_retry else "failed",
                        label=running_job.node.obj._log_label,
                        detail=_exception_summary(error),
                        status="error",
                        lease=lease_id,
                        object_id=object_id,
                        failed_attempts=failed_attempts,
                        max_retries=self.max_retries_per_object,
                        error=error,
                    )
                case JobBlockedResult(dependencies=dependencies):
                    _update_dag_blocking_dependencies(
                        self, running_job.node, dependencies
                    )
                    failed_retry, failed = self._failed_counts()
                    log_event(
                        logger,
                        logging.DEBUG,
                        "job blocked",
                        event="blocked",
                        label=running_job.node.obj._log_label,
                        dependencies=len(dependencies),
                        lease=lease_id,
                        object_id=running_job.node.obj.object_id,
                        ready=len(self.ready),
                        running=len(self.running),
                        blocked=len(self.blocked),
                        completed=len(self.completed),
                        failed_retry=failed_retry,
                        failed=failed,
                    )
                case _:
                    assert_never(request)
            failed_retry, failed = self._failed_counts()
            progress_detail = f"{len(self.completed)}/{len(self.nodes_by_id)}"
            if failed_retry or failed:
                progress_detail += f" · {failed_retry + failed} failed"
            log_event(
                logger,
                logging.INFO,
                "execution coordinator progress",
                event="progress",
                detail=progress_detail,
                completed=len(self.completed),
                total=len(self.nodes_by_id),
                failed_retry=failed_retry,
                failed=failed,
                running=len(self.running),
                ready=len(self.ready),
                blocked=len(self.blocked),
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
            log_event(
                logger,
                logging.ERROR,
                "furu execution coordinator finished with error: %s",
                message,
                event="failed",
                detail=_exception_summary(message),
                status="error",
                error=message,
            )
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
            self.finish_error = (
                "execution coordinator run could not complete; " + "; ".join(parts)
            )
            log_event(
                logger,
                logging.ERROR,
                "furu execution coordinator finished with error: %s",
                self.finish_error,
                event="failed",
                detail=_exception_summary(self.finish_error),
                status="error",
                error=self.finish_error,
            )
        else:
            elapsed = max(0.0, time.monotonic() - self.started_monotonic)
            log_event(
                logger,
                logging.INFO,
                "furu execution coordinator finished successfully",
                event="done",
                detail=(
                    f"{len(self.completed)}/{len(self.nodes_by_id)} "
                    f"· {_fmt_duration(elapsed)}"
                ),
            )
        self.done.set()
