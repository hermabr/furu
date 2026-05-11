from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence, cast

from furu.core import Furu
from furu.execution import BlockedOnDependencies, _execute_group, executor_job_context
from furu.locking import lock_many
from furu.migration import result_dir_for_loading
from furu.result import load_result_bundle

type JobState = Literal["queued", "running", "completed", "failed"]


class TooManySuspensionsError(RuntimeError):
    pass


@dataclass
class SchedulerJob:
    obj: Furu[Any]
    state: JobState = "queued"
    dependencies: set[str] = field(default_factory=set)
    suspensions: int = 0
    error: BaseException | None = None


class InMemoryScheduler:
    def __init__(self, *, max_suspensions_per_job: int = 100) -> None:
        self.jobs: dict[str, SchedulerJob] = {}
        self.max_suspensions_per_job = max_suspensions_per_job

    def submit(self, finals: Furu[Any] | Sequence[Furu[Any]]) -> None:
        if isinstance(finals, Furu):
            objs = (finals,)
        else:
            objs = tuple(finals)
        for obj in objs:
            self.add_artifact_recursive(obj)

    def add_artifact_recursive(self, obj: Furu[Any]) -> SchedulerJob:
        job = self._add_artifact(obj)
        deps = obj._declared_refs()
        self.add_dependencies(obj, deps, register_recursive=False)
        for dep in deps:
            self.add_artifact_recursive(dep)
        return job

    def add_dependencies(
        self,
        parent: Furu[Any],
        deps: Sequence[Furu[Any]],
        *,
        register_recursive: bool = True,
    ) -> None:
        parent_job = self._add_artifact(parent)
        for dep in deps:
            dep_job = (
                self.add_artifact_recursive(dep)
                if register_recursive
                else self._add_artifact(dep)
            )
            parent_job.dependencies.add(dep_job.obj.object_id)

    def claim_ready(self) -> Furu[Any] | None:
        completed_ids = self._completed_ids()
        for job_id in sorted(self.jobs):
            job = self.jobs[job_id]
            if job.state != "queued":
                continue
            if job.dependencies <= completed_ids:
                job.state = "running"
                return job.obj
        return None

    def complete_job(self, obj: Furu[Any]) -> None:
        job = self._add_artifact(obj)
        job.state = "completed"
        job.error = None

    def fail_job(self, obj: Furu[Any], exc: BaseException) -> None:
        job = self._add_artifact(obj)
        job.state = "failed"
        job.error = exc

    def suspend_job(self, obj: Furu[Any], deps: Sequence[Furu[Any]]) -> None:
        job = self._add_artifact(obj)
        job.suspensions += 1
        if job.suspensions > self.max_suspensions_per_job:
            self.fail_job(
                obj,
                TooManySuspensionsError(
                    f"{obj._log_label} suspended more than "
                    f"{self.max_suspensions_per_job} times while discovering dependencies"
                ),
            )
            return
        self.add_dependencies(obj, deps)
        job.state = "queued"

    def has_unfinished_jobs(self) -> bool:
        return any(job.state in {"queued", "running"} for job in self.jobs.values())

    def failed_jobs(self) -> tuple[SchedulerJob, ...]:
        return tuple(job for job in self.jobs.values() if job.state == "failed")

    def _add_artifact(self, obj: Furu[Any]) -> SchedulerJob:
        job = self.jobs.get(obj.object_id)
        if job is None:
            job = SchedulerJob(obj=obj)
            self.jobs[obj.object_id] = job
        return job

    def _completed_ids(self) -> set[str]:
        completed = {
            job_id for job_id, job in self.jobs.items() if job.state == "completed"
        }
        for job_id, job in self.jobs.items():
            if result_dir_for_loading(job.obj) is not None:
                completed.add(job_id)
        return completed


def run_worker(scheduler: InMemoryScheduler) -> None:
    while (obj := scheduler.claim_ready()) is not None:
        try:
            run_claimed_artifact(obj)
        except BlockedOnDependencies as exc:
            scheduler.suspend_job(obj, exc.deps)
        except BaseException as exc:
            scheduler.fail_job(obj, exc)
        else:
            scheduler.complete_job(obj)


def execute_local(
    finals: Furu[Any] | Sequence[Furu[Any]],
    *,
    max_suspensions_per_job: int = 100,
) -> InMemoryScheduler:
    scheduler = InMemoryScheduler(max_suspensions_per_job=max_suspensions_per_job)
    scheduler.submit(finals)
    run_worker(scheduler)
    return scheduler


def run_claimed_artifact[T](obj: Furu[T]) -> T:
    if (cached_result_dir := result_dir_for_loading(obj)) is not None:
        return cast(T, load_result_bundle(cached_result_dir))

    obj._internal_furu_dir.mkdir(parents=True, exist_ok=True)
    with lock_many([obj._lock_path]) as has_lock:
        if (cached_result_dir := result_dir_for_loading(obj)) is not None:
            return cast(T, load_result_bundle(cached_result_dir))

        results_by_dir: dict[Path, T] = {}
        with executor_job_context():
            _execute_group([obj], has_lock=has_lock, results_by_dir=results_by_dir)
        return results_by_dir[obj.data_dir]
