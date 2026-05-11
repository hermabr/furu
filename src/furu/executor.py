from __future__ import annotations

import os
import threading
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal, overload, cast

from furu.core import Furu
from furu.execution import BlockedOnDependencies, _run_claimed_artifact
from furu.migration import result_dir_for_loading
from furu.result import load_result_bundle

type JobState = Literal["queued", "running", "completed", "failed"]


@dataclass(slots=True)
class ExecutionGraph:
    root_ids: tuple[str, ...]
    artifacts: dict[str, Furu[Any]] = field(default_factory=dict)
    dependencies_by_parent: dict[str, set[str]] = field(default_factory=dict)

    @property
    def edges(self) -> frozenset[tuple[str, str]]:
        return frozenset(
            (parent_id, dependency_id)
            for parent_id, dependency_ids in self.dependencies_by_parent.items()
            for dependency_id in dependency_ids
        )

    def dependency_ids_for(self, object_id: str) -> frozenset[str]:
        return frozenset(self.dependencies_by_parent.get(object_id, ()))

    def add_artifact(self, obj: Furu[Any]) -> None:
        self.artifacts.setdefault(obj.object_id, obj)
        self.dependencies_by_parent.setdefault(obj.object_id, set())

    def add_edge(self, parent: Furu[Any], dependency: Furu[Any]) -> None:
        self.add_artifact(parent)
        self.add_artifact(dependency)
        self.dependencies_by_parent[parent.object_id].add(dependency.object_id)


class Planner:
    def plan(self, artifacts: Furu[Any] | Sequence[Furu[Any]]) -> ExecutionGraph:
        roots = _normalize_artifacts(artifacts)
        graph = ExecutionGraph(root_ids=tuple(obj.object_id for obj in roots))
        visited: set[str] = set()

        def visit(obj: Furu[Any]) -> None:
            graph.add_artifact(obj)
            if obj.object_id in visited:
                return
            visited.add(obj.object_id)

            for dependency in obj._declared_refs():
                graph.add_edge(obj, dependency)
                visit(dependency)

        for root in roots:
            visit(root)

        return graph


@dataclass(frozen=True, slots=True)
class ClaimedJob:
    object_id: str
    artifact: Furu[Any]
    dependencies: frozenset[str]
    suspension_count: int


@dataclass(frozen=True, slots=True)
class JobSnapshot:
    object_id: str
    artifact: Furu[Any]
    state: JobState
    dependencies: frozenset[str]
    suspension_count: int
    error: BaseException | None = None


@dataclass(slots=True)
class _SchedulerJob:
    object_id: str
    artifact: Furu[Any]
    state: JobState = "queued"
    dependencies: set[str] = field(default_factory=set)
    suspension_count: int = 0
    error: BaseException | None = None

    def snapshot(self) -> JobSnapshot:
        return JobSnapshot(
            object_id=self.object_id,
            artifact=self.artifact,
            state=self.state,
            dependencies=frozenset(self.dependencies),
            suspension_count=self.suspension_count,
            error=self.error,
        )


class InMemoryScheduler:
    def __init__(self, *, root_ids: Iterable[str] = ()) -> None:
        self._jobs: dict[str, _SchedulerJob] = {}
        self._root_ids = set(root_ids)
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)

    def submit(
        self,
        artifact_or_graph: Furu[Any] | ExecutionGraph,
        *,
        dependencies: Iterable[str] = (),
    ) -> None:
        with self._changed:
            if isinstance(artifact_or_graph, ExecutionGraph):
                self._root_ids.update(artifact_or_graph.root_ids)
                for object_id, artifact in artifact_or_graph.artifacts.items():
                    self._submit_artifact_locked(
                        artifact,
                        artifact_or_graph.dependencies_by_parent.get(object_id, ()),
                    )
            elif isinstance(artifact_or_graph, Furu):
                self._submit_artifact_locked(artifact_or_graph, dependencies)
            else:
                raise TypeError("submit() expected a Furu artifact or ExecutionGraph")
            self._changed.notify_all()

    def claim_ready(self, *, wait: bool = False) -> ClaimedJob | None:
        with self._changed:
            while True:
                completed_ids = {
                    object_id
                    for object_id, job in self._jobs.items()
                    if job.state == "completed"
                }
                for job in self._jobs.values():
                    if job.state != "queued":
                        continue
                    if not job.dependencies.issubset(completed_ids):
                        continue
                    job.state = "running"
                    return ClaimedJob(
                        object_id=job.object_id,
                        artifact=job.artifact,
                        dependencies=frozenset(job.dependencies),
                        suspension_count=job.suspension_count,
                    )

                if (
                    not wait
                    or self._has_failed_locked()
                    or self._roots_completed_locked()
                    or not self._has_running_locked()
                ):
                    return None
                self._changed.wait()

    def add_dependencies(self, object_id: str, dependencies: Iterable[str]) -> int:
        dependency_ids = set(dependencies)
        with self._changed:
            job = self._get_job_locked(object_id)
            if job.state == "completed":
                raise RuntimeError(
                    f"cannot add dependencies to completed job {object_id}"
                )
            if job.state == "failed":
                raise RuntimeError(f"cannot add dependencies to failed job {object_id}")
            job.dependencies.update(dependency_ids)
            job.suspension_count += 1
            job.state = "queued"
            self._changed.notify_all()
            return job.suspension_count

    def complete_job(self, object_id: str) -> None:
        with self._changed:
            job = self._get_job_locked(object_id)
            if job.state != "running":
                raise RuntimeError(f"cannot complete {job.state} job {object_id}")
            job.state = "completed"
            job.error = None
            self._changed.notify_all()

    def fail_job(self, object_id: str, error: BaseException) -> None:
        with self._changed:
            job = self._get_job_locked(object_id)
            if job.state == "completed":
                raise RuntimeError(f"cannot fail completed job {object_id}")
            job.state = "failed"
            job.error = error
            self._changed.notify_all()

    def roots_completed(self) -> bool:
        with self._lock:
            return self._roots_completed_locked()

    def has_running(self) -> bool:
        with self._lock:
            return self._has_running_locked()

    def failed_jobs(self) -> tuple[JobSnapshot, ...]:
        with self._lock:
            return tuple(
                job.snapshot() for job in self._jobs.values() if job.state == "failed"
            )

    def snapshot(self) -> tuple[JobSnapshot, ...]:
        with self._lock:
            return tuple(job.snapshot() for job in self._jobs.values())

    def _submit_artifact_locked(
        self, artifact: Furu[Any], dependencies: Iterable[str]
    ) -> None:
        object_id = artifact.object_id
        job = self._jobs.get(object_id)
        if job is None:
            self._jobs[object_id] = _SchedulerJob(
                object_id=object_id,
                artifact=artifact,
                dependencies=set(dependencies),
            )
            return
        if job.state in {"queued", "running"}:
            job.dependencies.update(dependencies)

    def _get_job_locked(self, object_id: str) -> _SchedulerJob:
        try:
            return self._jobs[object_id]
        except KeyError as exc:
            raise KeyError(f"unknown scheduler job {object_id}") from exc

    def _has_failed_locked(self) -> bool:
        return any(job.state == "failed" for job in self._jobs.values())

    def _has_running_locked(self) -> bool:
        return any(job.state == "running" for job in self._jobs.values())

    def _roots_completed_locked(self) -> bool:
        if self._root_ids:
            return all(
                (job := self._jobs.get(root_id)) is not None
                and job.state == "completed"
                for root_id in self._root_ids
            )
        return bool(self._jobs) and all(
            job.state == "completed" for job in self._jobs.values()
        )


class LocalExecutorError(RuntimeError):
    pass


class LocalExecutorFailed(LocalExecutorError):
    def __init__(self, failed_jobs: Sequence[JobSnapshot]) -> None:
        self.failed_jobs = tuple(failed_jobs)
        details = "; ".join(
            f"{job.object_id}: {job.error!r}" for job in self.failed_jobs
        )
        super().__init__(f"local executor failed jobs: {details}")


class NoRunnableJobsError(LocalExecutorError):
    def __init__(self, jobs: Sequence[JobSnapshot]) -> None:
        self.jobs = tuple(jobs)
        queued = [job for job in self.jobs if job.state == "queued"]
        details = "; ".join(
            f"{job.object_id} waiting on {sorted(job.dependencies)}" for job in queued
        )
        super().__init__(f"no runnable queued jobs remain: {details}")


class ExcessiveSuspensions(LocalExecutorError):
    def __init__(
        self,
        *,
        object_id: str,
        suspension_count: int,
        max_suspensions: int,
        dependency_ids: Iterable[str],
    ) -> None:
        self.object_id = object_id
        self.suspension_count = suspension_count
        self.max_suspensions = max_suspensions
        self.dependency_ids = tuple(sorted(dependency_ids))
        super().__init__(
            f"job {object_id} suspended {suspension_count} times, "
            f"exceeding max_suspensions_per_job={max_suspensions}; "
            f"latest dependencies: {', '.join(self.dependency_ids)}"
        )


class LocalExecutor:
    def __init__(
        self,
        *,
        max_workers: int | None = None,
        max_suspensions_per_job: int = 100,
        planner: Planner | None = None,
    ) -> None:
        if max_workers is not None and max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if max_suspensions_per_job < 1:
            raise ValueError("max_suspensions_per_job must be at least 1")

        self.max_workers = max_workers
        self.max_suspensions_per_job = max_suspensions_per_job
        self.planner = planner or Planner()
        self.scheduler: InMemoryScheduler | None = None

    @overload
    def run[T](self, artifacts: Furu[T]) -> T: ...

    @overload
    def run[T](self, artifacts: Sequence[Furu[T]]) -> list[T]: ...

    def run[T](self, artifacts: Furu[T] | Sequence[Furu[T]]) -> T | list[T]:
        normalized, unwrap = _normalize_artifact_request(artifacts)
        if not normalized:
            return []

        graph = self.planner.plan(normalized)
        scheduler = InMemoryScheduler(root_ids=graph.root_ids)
        scheduler.submit(graph)
        self.scheduler = scheduler

        max_workers = self._resolve_max_workers()
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="furu-local-worker",
        ) as pool:
            futures = [
                pool.submit(self._worker_loop, scheduler) for _ in range(max_workers)
            ]
            for future in futures:
                future.result()

        failed = scheduler.failed_jobs()
        if failed:
            raise LocalExecutorFailed(failed) from failed[0].error
        if not scheduler.roots_completed():
            raise NoRunnableJobsError(scheduler.snapshot())

        results = [_load_completed_result(obj) for obj in normalized]
        if unwrap:
            return cast(T, results[0])
        return cast(list[T], results)

    def _resolve_max_workers(self) -> int:
        if self.max_workers is not None:
            return self.max_workers
        return min(32, (os.cpu_count() or 1) + 4)

    def _worker_loop(self, scheduler: InMemoryScheduler) -> None:
        while True:
            claimed = scheduler.claim_ready(wait=True)
            if claimed is None:
                return
            self._run_claimed_job(scheduler, claimed)

    def _run_claimed_job(
        self, scheduler: InMemoryScheduler, claimed: ClaimedJob
    ) -> None:
        try:
            _run_claimed_artifact(claimed.artifact)
        except BlockedOnDependencies as exc:
            try:
                dependency_graph = self.planner.plan(exc.deps)
                scheduler.submit(dependency_graph)
                suspension_count = scheduler.add_dependencies(
                    claimed.object_id, exc.dependency_ids
                )
                if suspension_count > self.max_suspensions_per_job:
                    scheduler.fail_job(
                        claimed.object_id,
                        ExcessiveSuspensions(
                            object_id=claimed.object_id,
                            suspension_count=suspension_count,
                            max_suspensions=self.max_suspensions_per_job,
                            dependency_ids=exc.dependency_ids,
                        ),
                    )
            except BaseException as scheduling_exc:
                scheduler.fail_job(claimed.object_id, scheduling_exc)
        except BaseException as exc:
            scheduler.fail_job(claimed.object_id, exc)
        else:
            scheduler.complete_job(claimed.object_id)


@overload
def run_local[T](
    artifacts: Furu[T],
    *,
    max_workers: int | None = None,
    max_suspensions_per_job: int = 100,
) -> T: ...


@overload
def run_local[T](
    artifacts: Sequence[Furu[T]],
    *,
    max_workers: int | None = None,
    max_suspensions_per_job: int = 100,
) -> list[T]: ...


def run_local[T](
    artifacts: Furu[T] | Sequence[Furu[T]],
    *,
    max_workers: int | None = None,
    max_suspensions_per_job: int = 100,
) -> T | list[T]:
    executor = LocalExecutor(
        max_workers=max_workers,
        max_suspensions_per_job=max_suspensions_per_job,
    )
    return executor.run(artifacts)


def _normalize_artifacts(
    artifacts: Furu[Any] | Sequence[Furu[Any]],
) -> tuple[Furu[Any], ...]:
    if isinstance(artifacts, Furu):
        return (artifacts,)
    if not isinstance(artifacts, Sequence):
        raise TypeError("expected a Furu artifact or a sequence of Furu artifacts")
    normalized = tuple(artifacts)
    if any(not isinstance(artifact, Furu) for artifact in normalized):
        raise TypeError("expected Furu artifacts")
    return normalized


def _normalize_artifact_request[T](
    artifacts: Furu[T] | Sequence[Furu[T]],
) -> tuple[tuple[Furu[T], ...], bool]:
    if isinstance(artifacts, Furu):
        return (artifacts,), True
    if not isinstance(artifacts, Sequence):
        raise TypeError("expected a Furu artifact or a sequence of Furu artifacts")
    normalized = tuple(artifacts)
    if any(not isinstance(artifact, Furu) for artifact in normalized):
        raise TypeError("expected Furu artifacts")
    return normalized, False


def _load_completed_result[T](obj: Furu[T]) -> T:
    result_dir = result_dir_for_loading(obj)
    if result_dir is None:
        raise RuntimeError(f"executor completed without a result for {obj.object_id}")
    return cast(T, load_result_bundle(result_dir))
