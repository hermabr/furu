from __future__ import annotations

import logging
import os
import secrets
import threading
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, assert_never

from furu.config import get_config
from furu.core import Spec
from furu.dag import DagNode, _add_to_dag, _update_dag_blocking_dependencies
from furu.logging import _display_path, _execution_log, get_logger
from furu.metadata import ArtifactSpec
from furu.provenance import SubmitProvenance, capture_submit_provenance
from furu.storage._layout import execution_log_path_in, run_log_path_in
from furu.utils import error_summary, format_duration
from furu.worker.backends.protocol import can_run
from furu.worker.protocol import (
    Job,
    JobBlockedResult,
    JobCompletedResult,
    JobFailedResult,
    JobResult,
    PoolHandoff,
    ProcessSettings,
)

if TYPE_CHECKING:
    from furu.worker.backends.protocol import WorkerBackend, WorkerPool


logger = get_logger("coord")

_RUNNING_ELSEWHERE_POLL_INTERVAL_S = 5.0


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
    max_retries_per_object: int = field(
        default_factory=lambda: get_config().worker.max_retries_per_object
    )
    backends: dict[str, WorkerBackend]
    submit_provenance: SubmitProvenance
    executor_id: str = field(default_factory=lambda: secrets.token_hex(16))
    nodes_by_id: dict[str, DagNode] = field(default_factory=dict)
    ready: dict[str, DagNode] = field(default_factory=dict)
    running_elsewhere: set[str] = field(default_factory=set)
    blocked: dict[str, DagNode] = field(default_factory=dict)
    running: dict[str, RunningJob] = field(default_factory=dict)
    completed: dict[str, DagNode] = field(default_factory=dict)
    failed: dict[str, FailedJob] = field(default_factory=dict)
    pools: dict[str, WorkerPool] = field(default_factory=dict)
    lock: threading.Condition = field(default_factory=threading.Condition)
    done: threading.Event = field(default_factory=threading.Event)
    finish_error: str | None = None
    taken_over_by: str | None = None

    def _failed_counts(self) -> tuple[int, int]:
        failed_retry = sum(
            record.failed_attempts <= self.max_retries_per_object
            for record in self.failed.values()
        )
        return failed_retry, len(self.failed) - failed_retry

    @classmethod
    def run[ObjsT: Sequence[Spec]](
        cls,
        objs: ObjsT,  # TODO: support pytrees
        *,
        worker_backends: tuple[WorkerBackend, ...],
        port: int = 0,
    ) -> ObjsT:
        if all(isinstance(obj, Spec) and obj.status == "done" for obj in objs):
            logger.info(
                "all objects already exist; no execution coordinator work to run"
            )
            return objs

        takeover = (
            _resolve_takeover(prefix)
            if (prefix := os.environ.get("FURU_TAKEOVER")) is not None
            else None
        )
        backends = {backend.pool_key: backend for backend in worker_backends}
        if len(backends) != len(worker_backends):
            raise ValueError(
                "worker backends with identical configuration; "
                "use one backend with a larger max_workers instead"
            )
        coordinator = cls(
            backends=backends,
            submit_provenance=capture_submit_provenance(
                snapshot=get_config().provenance.snapshot
            ),
        )
        _add_to_dag(coordinator, objs)

        if not coordinator.nodes_by_id:
            logger.info(
                "all objects already exist; no execution coordinator work to run"
            )
            return objs

        (bind_host,) = {
            backend.execution_coordinator_listen_host for backend in worker_backends
        }
        from furu.execution.server import (
            execution_coordinator_server,
            request_takeover,
        )

        execution_log = execution_log_path_in(coordinator.executor_dir)
        with _execution_log(execution_log):
            logger.info(
                "starting exec %s · %d ready · %d blocked",
                coordinator.executor_id[:5],
                len(coordinator.ready),
                len(coordinator.blocked),
                extra={"path": execution_log},
            )
            try:
                with execution_coordinator_server(
                    coordinator, bind_host=bind_host, port=port
                ) as server:
                    logger.info("server listening on %s", server.server_url)
                    handshake = (
                        request_takeover(
                            executor_id=coordinator.executor_id,
                            source_id=takeover[0],
                            url=takeover[1],
                            pool_keys=list(backends),
                        )
                        if takeover is not None
                        else nullcontext({})
                    )
                    with handshake as handoffs:
                        if takeover is not None:
                            if os.environ.get("FURU_TAKEOVER") == prefix:
                                del os.environ["FURU_TAKEOVER"]
                            logger.info(
                                "taking over exec=%s · inherited %d workers",
                                takeover[0][:5],
                                sum(len(h.job_ids) for h in handoffs.values()),
                            )
                        for pool_key, backend in backends.items():
                            handoff = handoffs.get(pool_key, PoolHandoff())
                            coordinator.pools[pool_key] = backend.start_pool(
                                coordinator=coordinator,
                                bound_port=server.bound_port,
                                auth_token=server.auth_token,
                                executor_dir=coordinator.executor_dir,
                                handoff=handoff,
                            )
                            logger.info(
                                "pool started · %s%s",
                                type(backend).__name__,
                                f" · inherited {len(handoff.job_ids)} workers"
                                if handoff.job_ids
                                else "",
                            )
                    coordinator.done.wait()
            finally:
                if pools := list(coordinator.pools.values()):
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

    def lease_job(self, *, backend: WorkerBackend, worker: str) -> Job | None:
        with self.lock:
            while True:
                if self.done.is_set() or self.taken_over_by is not None:
                    return None
                running_elsewhere: set[str] = set()
                for node, member_ids in self._satisfiable_leases_locked(backend):
                    object_ids: list[str] = []
                    for object_id in (node.obj.object_id, *member_ids):
                        if self.nodes_by_id[object_id].obj.status == "running":
                            running_elsewhere.add(object_id)
                        else:
                            object_ids.append(object_id)
                    if not object_ids:
                        continue
                    self._defer_running_elsewhere_locked(running_elsewhere)
                    nodes = self._start_locked(object_ids, worker=worker)
                    node = nodes[0]
                    run_logs = [run_log_path_in(node.obj._base_dir) for node in nodes]
                    logger.info(
                        "leased %s%s to %s",
                        node.obj._log_label,
                        f" ×{len(nodes)}" if len(nodes) > 1 else "",
                        worker,
                        extra={"path": run_logs[0]},
                    )
                    previous = self.failed.get(node.obj.object_id)
                    return Job(
                        artifacts=[ArtifactSpec.from_furu(node.obj) for node in nodes],
                        run_logs=run_logs,
                        attempt=(previous.failed_attempts if previous else 0) + 1,
                        execution_log=execution_log_path_in(self.executor_dir),
                        provenance=self.submit_provenance,
                        process=ProcessSettings.from_metadata(node.obj._metadata),
                    )
                self._defer_running_elsewhere_locked(running_elsewhere)
                if not self.running_elsewhere:
                    self.lock.wait()
                    continue
                if self.lock.wait(timeout=_RUNNING_ELSEWHERE_POLL_INTERVAL_S):
                    continue
                available = {
                    object_id
                    for object_id in self.running_elsewhere
                    if self.nodes_by_id[object_id].obj.status != "running"
                }
                self.running_elsewhere.difference_update(available)
                if available:
                    self.lock.notify_all()

    def _defer_running_elsewhere_locked(self, object_ids: set[str]) -> None:
        newly_deferred = object_ids - self.running_elsewhere
        if not newly_deferred:
            return
        self.running_elsewhere.update(newly_deferred)
        logger.info(
            "run is waiting on %d external spec%s",
            len(newly_deferred),
            "" if len(newly_deferred) == 1 else "s",
        )

    def _start_locked(self, object_ids: Sequence[str], *, worker: str) -> list[DagNode]:
        started_at = time.monotonic()
        self.running_elsewhere.difference_update(object_ids)
        nodes = [self.ready.pop(object_id) for object_id in object_ids]
        for node in nodes:
            self.running[node.obj.object_id] = RunningJob(
                node=node, started_at=started_at, worker=worker
            )
        return nodes

    def adopt(self, artifacts: Sequence[ArtifactSpec], *, worker: str) -> bool:
        with self.lock:
            object_ids = [artifact.object_id for artifact in artifacts]
            label = artifacts[0].log_label
            if len(artifacts) > 1:
                label += f" ×{len(artifacts)}"
            if self.done.is_set() or any(
                object_id not in self.ready for object_id in object_ids
            ):
                logger.info("cancelled %s on %s: not in this run", label, worker)
                return False
            self._start_locked(object_ids, worker=worker)
            logger.info("adopted %s from %s", label, worker)
            return True

    def worker_lost(self, worker: str) -> None:
        with self.lock:
            if self.done.is_set():
                return
            self._release_worker_locked(worker, reason="worker is no longer active")
            self.lock.notify_all()

    def count_satisfiable_jobs(
        self, *, backend: WorkerBackend, max_workers: int
    ) -> int:
        with self.lock:
            if self.done.is_set():
                return 0
            return sum(
                1 for _ in islice(self._satisfiable_leases_locked(backend), max_workers)
            )

    def _satisfiable_leases_locked(
        self, backend: WorkerBackend
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
            if object_id in consumed or object_id in self.running_elsewhere:
                continue
            if not can_run(backend, node.obj):
                continue
            throttle = node.obj.throttle
            if (
                throttle is not None
                and running_counts.get(type(node.obj), 0) >= throttle.max_running
            ):
                continue
            consumed.add(object_id)
            member_ids: list[str] = []
            if (group := node.batch_group(backend.worker)) is not None:
                group_key, cap = group
                for other_id, other in self.ready.items():
                    if len(member_ids) + 1 >= cap:
                        break
                    if other_id in consumed:
                        continue
                    if other_id in self.running_elsewhere:
                        continue
                    other_group = other.batch_group(backend.worker)
                    if other_group is None or other_group[0] != group_key:
                        continue
                    if not can_run(backend, other.obj):
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
            )

    def job_result(self, object_id: str, request: JobResult) -> None:
        with self.lock:
            running_job = self.running.pop(object_id, None)
            if running_job is None:
                logger.info("ignoring result for %s: no longer running", object_id)
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
                    # The traceback lives in run.log; repeats say so instead.
                    summary = error_summary(error)
                    if previous_failed and error_summary(previous_failed.error) == (
                        summary
                    ):
                        summary = "same error"
                    logger.log(
                        logging.WARNING if will_retry else logging.ERROR,
                        "failed %s · attempt %d/%d · %s",
                        running_job.node.obj._log_label,
                        failed_attempts,
                        self.max_retries_per_object,
                        summary,
                        extra={"path": run_log_path_in(running_job.node.obj._base_dir)},
                    )
                case JobBlockedResult(dependencies=dependencies):
                    try:
                        _update_dag_blocking_dependencies(
                            self, running_job.node, dependencies
                        )
                    except Exception as exc:  # noqa: BLE001 -- runs user runs_on, accepts and batch fns
                        self.fail(f"{type(exc).__name__}: {exc}")
                        return
                    logger.info(
                        "blocked %s · %d deps",
                        running_job.node.obj._log_label,
                        len(dependencies),
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
            )
            self._maybe_finish_locked()
            self.lock.notify_all()

    def raise_for_failure(self) -> None:
        if self.finish_error is not None:
            raise RuntimeError(self.finish_error)

    def fail(self, message: str) -> None:
        with self.lock:
            if self.done.is_set():
                return
            self.finish_error = message
            logger.error(
                "run failed · %s",
                message,
                extra={"path": execution_log_path_in(self.executor_dir)},
            )
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

        if not terminal_failed and not self.blocked:
            logger.info("run finished ok")
            self.done.set()
            return

        def plural(count: int) -> str:
            return f"{count} spec{'' if count == 1 else 's'}"

        lines = [f"{plural(len(terminal_failed))} failed, {len(self.blocked)} blocked"]
        for record in terminal_failed.values():
            obj = record.node.obj
            lines.append(
                f"  {obj._log_label} · {record.failed_attempts} attempt"
                f"{'' if record.failed_attempts == 1 else 's'} · "
                f"{error_summary(record.error)} → "
                f"{_display_path(run_log_path_in(obj._base_dir))}"
            )
        lines += [
            f"  {node.obj._log_label} · blocked" for node in self.blocked.values()
        ]
        self.finish_error = "\n".join(lines)

        headline = ["run failed"]
        if terminal_failed:
            headline.append(f"{plural(len(terminal_failed))} failed")
        if self.blocked:
            headline.append(f"{plural(len(self.blocked))} blocked")
        if len(terminal_failed) == 1:
            (record,) = terminal_failed.values()
            headline += [record.node.obj._log_label, error_summary(record.error)]
        logger.error(
            " · ".join(headline),
            extra={"path": execution_log_path_in(self.executor_dir)},
        )
        self.done.set()


def _resolve_takeover(prefix: str) -> tuple[str, str]:
    from furu.config import _read_worker_json_config

    executions = get_config().run_directories.executions
    matches = sorted(
        path.name
        for path in (executions.iterdir() if executions.is_dir() else ())
        if path.name.startswith(prefix)
    )
    if len(matches) != 1:
        found = f"; candidates: {', '.join(matches)}" if matches else ""
        raise RuntimeError(
            f"FURU_TAKEOVER={prefix} matches {len(matches)} executions{found}"
        )
    (executor_id,) = matches
    worker_file = next(
        (executions / executor_id / "workers").glob("*/worker.config.json"), None
    )
    if worker_file is None:
        raise RuntimeError(f"exec={executor_id[:5]} has no worker pools to take over")
    coordinator_url, _ = _read_worker_json_config(worker_file)
    return executor_id, coordinator_url
