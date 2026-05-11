from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from furu.migration import result_dir_for_loading

if TYPE_CHECKING:
    from furu.core import Furu


type JobState = Literal["queued", "running", "completed", "failed"]


DEFAULT_MAX_SUSPENSIONS_PER_JOB = 5


@dataclass
class Job:
    obj: Furu[Any]
    state: JobState = "queued"
    dependencies: set[str] = field(default_factory=set)
    suspensions: int = 0
    error: BaseException | None = None


class Scheduler:
    """In-memory scheduler for the local distributed executor.

    Tracks jobs keyed by ``object_id`` so that multiple finals or dynamic
    dependency discoveries referencing the same artifact converge to one job.
    There is no separate ``blocked`` state: a queued job is considered runnable
    when every ID in its ``dependencies`` set is either a completed job in this
    scheduler or already cached on disk.
    """

    def __init__(
        self,
        *,
        max_suspensions_per_job: int = DEFAULT_MAX_SUSPENSIONS_PER_JOB,
    ) -> None:
        self._jobs: dict[str, Job] = {}
        self.max_suspensions_per_job = max_suspensions_per_job

    def submit(self, finals: Iterable[Furu[Any]]) -> None:
        for final in finals:
            self.add_artifact_recursive(final)

    def add_artifact_recursive(self, obj: Furu[Any]) -> Job:
        existing = self._jobs.get(obj.object_id)
        if existing is not None:
            return existing

        job = Job(obj=obj)
        self._jobs[obj.object_id] = job

        for ref in obj._declared_refs():
            self.add_artifact_recursive(ref)
            job.dependencies.add(ref.object_id)
        return job

    def add_dependencies(self, parent: Furu[Any], deps: Iterable[Furu[Any]]) -> None:
        parent_job = self._jobs[parent.object_id]
        for dep in deps:
            self.add_artifact_recursive(dep)
            parent_job.dependencies.add(dep.object_id)

    def claim_ready(self) -> Job | None:
        for job in self._jobs.values():
            if job.state != "queued":
                continue
            if not self._dependencies_satisfied(job):
                continue
            job.state = "running"
            return job
        return None

    def _dependencies_satisfied(self, job: Job) -> bool:
        for dep_id in job.dependencies:
            dep_job = self._jobs.get(dep_id)
            if dep_job is None:
                return False
            if dep_job.state == "completed":
                continue
            if dep_job.state in ("queued", "running"):
                if result_dir_for_loading(dep_job.obj) is not None:
                    dep_job.state = "completed"
                    continue
                return False
            if dep_job.state == "failed":
                return False
        return True

    def complete_job(self, obj: Furu[Any]) -> None:
        job = self._jobs[obj.object_id]
        job.state = "completed"
        job.error = None

    def fail_job(self, obj: Furu[Any], exc: BaseException) -> None:
        job = self._jobs[obj.object_id]
        job.state = "failed"
        job.error = exc

    def suspend_job(self, obj: Furu[Any]) -> bool:
        """Move a running job back to queued after a BlockedOnDependencies.

        Returns ``True`` if the job remains retryable, ``False`` if the
        per-job suspension budget was exhausted (in which case the job has
        been marked failed).
        """
        job = self._jobs[obj.object_id]
        job.suspensions += 1
        if job.suspensions > self.max_suspensions_per_job:
            self.fail_job(
                obj,
                RuntimeError(
                    f"job {obj._log_label} exceeded max_suspensions_per_job="
                    f"{self.max_suspensions_per_job}; possible cycle or "
                    "non-converging dynamic dependencies"
                ),
            )
            return False
        job.state = "queued"
        return True

    def jobs(self) -> dict[str, Job]:
        return self._jobs

    def is_done(self) -> bool:
        return all(job.state in ("completed", "failed") for job in self._jobs.values())

    def has_failures(self) -> bool:
        return any(job.state == "failed" for job in self._jobs.values())
