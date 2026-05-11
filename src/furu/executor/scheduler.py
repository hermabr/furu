from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from furu.core import Furu
from furu.migration import result_dir_for_loading

type JobState = Literal["queued", "running", "completed", "failed"]


class MaxSuspensionsExceeded(RuntimeError):
    def __init__(
        self,
        obj: Furu[Any],
        *,
        max_suspensions: int,
        deps: Sequence[Furu[Any]],
    ) -> None:
        labels = ", ".join(dep._log_label for dep in deps)
        super().__init__(
            f"{obj._log_label} suspended more than {max_suspensions} times "
            f"while rediscovering missing dependencies: {labels}"
        )


@dataclass(slots=True)
class SchedulerJob:
    artifact: Furu[Any]
    dependencies: set[str] = field(default_factory=set)
    state: JobState = "queued"
    suspensions: int = 0
    exception: BaseException | None = None
    declared_dependencies_registered: bool = False


class InMemoryScheduler:
    def __init__(self, *, max_suspensions_per_job: int = 10) -> None:
        if max_suspensions_per_job < 0:
            raise ValueError("max_suspensions_per_job must be non-negative")
        self.max_suspensions_per_job = max_suspensions_per_job
        self.jobs: dict[str, SchedulerJob] = {}
        self.final_ids: set[str] = set()

    def submit(
        self,
        finals: Furu[Any] | Sequence[Furu[Any]],
    ) -> tuple[SchedulerJob, ...]:
        from furu.executor.planner import submit

        return submit(self, finals)

    def add_artifact(self, obj: Furu[Any]) -> SchedulerJob:
        job = self.jobs.get(obj.object_id)
        if job is not None:
            return job

        job = SchedulerJob(artifact=obj)
        self.jobs[obj.object_id] = job
        return job

    def add_artifact_recursive(self, obj: Furu[Any]) -> SchedulerJob:
        from furu.executor.planner import add_artifact_recursive

        return add_artifact_recursive(self, obj)

    def add_dependencies(
        self,
        parent: Furu[Any],
        deps: Sequence[Furu[Any]],
    ) -> None:
        parent_job = self.add_artifact(parent)
        for dep in deps:
            dep_job = self.add_artifact_recursive(dep)
            parent_job.dependencies.add(dep_job.artifact.object_id)

    def claim_ready(self) -> Furu[Any] | None:
        self._refresh_cached_completions()
        for job in self.jobs.values():
            if job.state == "queued" and self._dependencies_satisfied(job):
                job.state = "running"
                return job.artifact
        return None

    def complete_job(self, obj: Furu[Any]) -> None:
        job = self.add_artifact(obj)
        job.state = "completed"
        job.exception = None

    def fail_job(self, obj: Furu[Any], exc: BaseException) -> None:
        job = self.add_artifact(obj)
        job.state = "failed"
        job.exception = exc

    def suspend_job(
        self,
        obj: Furu[Any],
        deps: Sequence[Furu[Any]],
    ) -> None:
        job = self.add_artifact(obj)
        job.suspensions += 1
        self.add_dependencies(obj, deps)

        if job.suspensions > self.max_suspensions_per_job:
            self.fail_job(
                obj,
                MaxSuspensionsExceeded(
                    obj,
                    max_suspensions=self.max_suspensions_per_job,
                    deps=deps,
                ),
            )
            return

        job.state = "queued"

    def failed_jobs(self) -> tuple[SchedulerJob, ...]:
        return tuple(job for job in self.jobs.values() if job.state == "failed")

    def unfinished_jobs(self) -> tuple[SchedulerJob, ...]:
        self._refresh_cached_completions()
        return tuple(
            job
            for job in self.jobs.values()
            if job.state not in {"completed", "failed"}
        )

    def finals_completed(self) -> bool:
        self._refresh_cached_completions()
        return all(
            self.jobs[final_id].state == "completed" for final_id in self.final_ids
        )

    def _dependencies_satisfied(self, job: SchedulerJob) -> bool:
        return all(self._dependency_satisfied(dep_id) for dep_id in job.dependencies)

    def _dependency_satisfied(self, dep_id: str) -> bool:
        dep_job = self.jobs.get(dep_id)
        if dep_job is None:
            return False
        if dep_job.state == "completed":
            return True
        return result_dir_for_loading(dep_job.artifact) is not None

    def _refresh_cached_completions(self) -> None:
        for job in self.jobs.values():
            if (
                job.state == "queued"
                and result_dir_for_loading(job.artifact) is not None
            ):
                job.state = "completed"
                job.exception = None
