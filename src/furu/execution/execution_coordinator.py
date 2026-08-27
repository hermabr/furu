from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, assert_never

from furu.config import get_config
from furu.core import Spec
from furu.dag import DagNode, _add_to_dag, _update_dag_blocking_dependencies
from furu.logging import (
    _scoped_component,
    _scoped_log_files,
    get_logger,
    log_detail,
)
from furu.metadata import ArtifactSpec
from furu.provenance import SubmitProvenance, capture_submit_provenance
from furu.resources import ResourceRequest, resource_request_satisfies
from furu.storage._layout import execution_coordinator_log_path_in
from furu.utils import format_duration
from furu.worker.protocol import (
    Job,
    JobBlockedResult,
    JobCompletedResult,
    JobFailedResult,
    JobResult,
)

if TYPE_CHECKING:
    from furu.worker.backends.protocol import WorkerBackend


logger = get_logger()


@dataclass(frozen=True, slots=True)
class RunningJob:
    node: DagNode
    started_at: float
    worker: str


@dataclass(frozen=True, slots=True)
class FailedJob:
    failed_attempts: int
    node: DagNode
    error: str


@dataclass(slots=True, kw_only=True)
class ExecutionCoordinator:
    max_retries_per_object: int
    # None skips fail-fast validation when no pool configuration is available.
    pool_resources: tuple[ResourceRequest, ...] | None = None
    executor_id: str = "not-computed-yet"
    nodes_by_id: dict[str, DagNode] = field(default_factory=dict)
    ready: dict[str, DagNode] = field(default_factory=dict)
    blocked: dict[str, DagNode] = field(default_factory=dict)
    running: dict[str, RunningJob] = field(default_factory=dict)
    completed: dict[str, DagNode] = field(default_factory=dict)
    failed: dict[str, FailedJob] = field(default_factory=dict)
    lock: threading.Condition = field(default_factory=threading.Condition)
    done: threading.Event = field(default_factory=threading.Event)
    finish_error: str | None = None
    submit_provenance: SubmitProvenance | None = None

    def _failed_counts(self) -> tuple[int, int]:
        failed_retry = sum(
            record.failed_attempts <= self.max_retries_per_object
            for record in self.failed.values()
        )
        return failed_retry, len(self.failed) - failed_retry

    def _counts_detail(self) -> dict[str, object]:
        failed_retry, failed = self._failed_counts()
        return {
            "ready": len(self.ready),
            "running": len(self.running),
            "blocked": len(self.blocked),
            "completed": len(self.completed),
            "total": len(self.nodes_by_id),
            "failed_retry": failed_retry,
            "failed": failed,
        }

    @classmethod
    def run[ObjsT: Sequence[Spec]](
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
        coordinator.pool_resources = tuple(
            backend.resource_request for backend in worker_backends
        )
        _add_to_dag(coordinator, objs)
        digest = hashlib.blake2s(digest_size=16)
        for obj in objs:
            digest.update(obj.object_id.encode("utf-8"))
            digest.update(b"\0")
        coordinator.executor_id = digest.hexdigest()

        if not coordinator.nodes_by_id:
            with coordinator.log_context(), coordinator.lock:
                logger.info(
                    "all objects already exist; no execution coordinator work to run"
                )
                coordinator._maybe_finish_locked()
            return objs

        # One capture (and at most one snapshot build) for the whole batch;
        # every job carries this same frozen submit half.
        coordinator.submit_provenance = capture_submit_provenance(
            snapshot=get_config().provenance.snapshot
        )

        (bind_host,) = {
            backend.execution_coordinator_listen_host for backend in worker_backends
        }

        from furu.execution.server import execution_coordinator_server

        with coordinator.log_context():
            logger.info(
                "starting exec=%s · %d ready · %d blocked",
                coordinator.executor_id[:5],
                len(coordinator.ready),
                len(coordinator.blocked),
                extra=log_detail(
                    executor_id=coordinator.executor_id,
                    executor_dir=coordinator.executor_dir,
                ),
            )
            pools = []
            try:
                with execution_coordinator_server(
                    coordinator, bind_host=bind_host, port=port
                ) as server:
                    logger.info("server listening on %s", server.server_url)
                    for backend in worker_backends:
                        pool = backend.start_pool(
                            coordinator=coordinator,
                            bound_port=server.bound_port,
                            auth_token=server.auth_token,
                            executor_dir=coordinator.executor_dir,
                            provenance=coordinator.submit_provenance,
                        )
                        pools.append(pool)
                        logger.info("pool started · %s", type(backend).__name__)
                    coordinator.done.wait()
            finally:
                if pools:
                    with ThreadPoolExecutor(max_workers=len(pools)) as executor:
                        stop_futures = [
                            executor.submit(pool.stop, timeout=5) for pool in pools
                        ]
                    for pool, future in zip(pools, stop_futures, strict=True):
                        if (exc := future.exception()) is not None:
                            logger.error(
                                "pool stop failed · %s · %s",
                                type(pool).__name__,
                                exc,
                            )
        coordinator.raise_for_failure()
        return objs

    @property
    def executor_dir(self) -> Path:
        return get_config().run_directories.executions / self.executor_id

    @contextmanager
    def log_context(self) -> Iterator[None]:
        with (
            _scoped_component("coord"),
            _scoped_log_files((execution_coordinator_log_path_in(self.executor_dir),)),
        ):
            yield

    def lease_job(self, *, resources: ResourceRequest, worker: str) -> Job | None:
        with self.log_context(), self.lock:
            while True:
                if self.done.is_set():
                    return None
                lease = next(self._satisfiable_leases_locked(resources), None)
                if lease is not None:
                    break
                self.lock.wait()
            node, member_ids = lease
            nodes = [
                self.ready.pop(member_id)
                for member_id in (node.obj.object_id, *member_ids)
            ]
            started_at = time.monotonic()
            for member_node in nodes:
                self.running[member_node.obj.object_id] = RunningJob(
                    node=member_node,
                    started_at=started_at,
                    worker=worker,
                )
            logger.info(
                "leased %s ×%d to %s",
                node.obj._log_label,
                len(nodes),
                worker,
                extra=log_detail(
                    object_ids=",".join(node.obj.object_id for node in nodes),
                    members=len(nodes),
                    worker=worker,
                    **self._counts_detail(),
                ),
            )
            assert self.submit_provenance is not None
            return Job(
                artifacts=[ArtifactSpec.from_furu(node.obj) for node in nodes],
                provenance=self.submit_provenance,
            )

    def worker_lost(self, worker: str) -> None:
        with self.log_context(), self.lock:
            if self.done.is_set():
                return
            self._release_worker_locked(worker, reason="worker is no longer active")
            self.lock.notify_all()

    def count_satisfiable_jobs(
        self, *, resources: ResourceRequest, max_workers: int
    ) -> int:
        with self.lock:
            if self.done.is_set():
                return 0
            leases = self._satisfiable_leases_locked(resources)
            return sum(1 for _ in islice(leases, max_workers))

    def _satisfiable_leases_locked(
        self, resources: ResourceRequest
    ) -> Iterator[tuple[DagNode, list[str]]]:
        """Yield (node, batch member ids) for each lease that could start now.

        Throttles count concurrent create calls, so a running batch counts
        once. A worker holds at most one job at a time, which makes distinct
        (worker, spec type) pairs the number of running jobs per type.
        """
        running_counts: dict[type[Spec], int] = {}
        for _, obj_type in {
            (job.worker, type(job.node.obj)) for job in self.running.values()
        }:
            running_counts[obj_type] = running_counts.get(obj_type, 0) + 1
        consumed: set[str] = set()
        for object_id, node in self.ready.items():
            if object_id in consumed:
                continue
            if not resource_request_satisfies(resources, node.obj._metadata.requires):
                continue
            throttle = node.obj.throttle
            if (
                throttle is not None
                and running_counts.get(type(node.obj), 0) >= throttle.max_running
            ):
                continue
            consumed.add(object_id)
            member_ids: list[str] = []
            if node.batch_group is not None:
                group_key, cap = node.batch_group
                for other_id, other in self.ready.items():
                    if len(member_ids) + 1 >= cap:
                        break
                    if other_id in consumed:
                        continue
                    if other.batch_group is None or other.batch_group[0] != group_key:
                        continue
                    member_ids.append(other_id)
                consumed.update(member_ids)
            running_counts[type(node.obj)] = running_counts.get(type(node.obj), 0) + 1
            yield node, member_ids

    def _release_worker_locked(self, worker: str, *, reason: str) -> None:
        for object_id, running_job in tuple(self.running.items()):
            if running_job.worker != worker:
                continue
            self.running.pop(object_id)
            self.ready[object_id] = running_job.node
            logger.warning(
                "released %s from %s: %s",
                running_job.node.obj._log_label,
                worker,
                reason,
                extra=log_detail(
                    object_id=object_id,
                    worker=worker,
                    **self._counts_detail(),
                ),
            )

    def job_result(self, object_id: str, request: JobResult) -> None:
        with self.log_context(), self.lock:
            running_job = self.running.pop(object_id, None)
            if running_job is None:
                logger.info(
                    "ignoring result for job that is no longer running",
                    extra=log_detail(object_id=object_id),
                )
                return
            match request:
                case JobCompletedResult():
                    self.failed.pop(object_id, None)
                    self.completed[object_id] = running_job.node
                    for dependent in tuple(running_job.node.dependents):
                        if running_job.node in dependent.dependencies:
                            dependent.dependencies.remove(running_job.node)

                        dependent_id = dependent.obj.object_id
                        if not dependent.dependencies and dependent_id in self.blocked:
                            self.ready[dependent_id] = self.blocked.pop(dependent_id)
                    logger.info(
                        "completed %s ok · %s",
                        running_job.node.obj._log_label,
                        format_duration(time.monotonic() - running_job.started_at),
                        extra=log_detail(
                            object_id=object_id,
                            **self._counts_detail(),
                        ),
                    )

                case JobFailedResult(error=error):
                    previous_failed = self.failed.get(object_id)
                    failed_attempts = (
                        previous_failed.failed_attempts if previous_failed else 0
                    ) + 1
                    self.failed[object_id] = FailedJob(
                        failed_attempts=failed_attempts,
                        node=running_job.node,
                        error=error,
                    )
                    will_retry = failed_attempts <= self.max_retries_per_object
                    if will_retry:
                        self.ready[object_id] = running_job.node
                    duration = format_duration(
                        time.monotonic() - running_job.started_at
                    )
                    label = running_job.node.obj._log_label
                    fail_detail = log_detail(object_id=object_id, error=error)
                    if will_retry:
                        logger.warning(
                            "failed %s · attempt %d/%d, will retry · %s: %s",
                            label,
                            failed_attempts,
                            self.max_retries_per_object,
                            duration,
                            error,
                            extra=fail_detail,
                        )
                    else:
                        logger.error(
                            "failed %s · attempt %d/%d · %s",
                            label,
                            failed_attempts,
                            self.max_retries_per_object,
                            duration,
                            extra=fail_detail,
                        )
                case JobBlockedResult(dependencies=dependencies):
                    try:
                        _update_dag_blocking_dependencies(
                            self, running_job.node, dependencies
                        )
                    except RuntimeError as exc:
                        self.fail(str(exc))
                        return
                    logger.info(
                        "blocked %s · %d deps",
                        running_job.node.obj._log_label,
                        len(dependencies),
                        extra=log_detail(
                            object_id=object_id,
                            dependencies=len(dependencies),
                            **self._counts_detail(),
                        ),
                    )
                case _:
                    assert_never(request)
            failed_retry, failed = self._failed_counts()
            parts = [f"{len(self.running)} running"]
            if self.ready:
                parts.append(f"{len(self.ready)} ready")
            if self.blocked:
                parts.append(f"{len(self.blocked)} blocked")
            if failed:
                parts.append(f"{failed} failed")
            if failed_retry:
                parts.append(f"{failed_retry} retrying")
            logger.info(
                "progress %s",
                f"{len(self.completed)}/{len(self.nodes_by_id)} · " + " · ".join(parts),
                extra=log_detail(**self._counts_detail()),
            )
            self._maybe_finish_locked()
            self.lock.notify_all()

    def raise_for_failure(self) -> None:
        if self.finish_error is not None:
            raise RuntimeError(self.finish_error)

    def fail(self, message: str) -> None:
        with self.log_context(), self.lock:
            if self.done.is_set():
                return
            self.finish_error = message
            logger.error("furu execution coordinator finished with error: %s", message)
            self.done.set()
            self.lock.notify_all()

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
                    f"after {failed_job.failed_attempts} failed attempts: "
                    f"{failed_job.error}"
                )
            self.finish_error = (
                "execution coordinator run could not complete; " + "; ".join(parts)
            )
            logger.error(
                "furu execution coordinator finished with error: %s",
                self.finish_error,
            )
        else:
            logger.info("furu execution coordinator finished successfully")
        self.done.set()
