from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from furu.core import Furu

type JobState = Literal["queued", "running", "completed", "failed"]


@dataclass(slots=True)
class Job:
    artifact: Furu[Any]
    state: JobState = "queued"
    dependencies: set[str] = field(default_factory=set)
    suspension_count: int = 0
    error: BaseException | None = None


class UpstreamFailure(RuntimeError):
    """Raised when a job is failed because one of its dependencies failed."""


class Scheduler:
    """In-memory job scheduler.

    States and transitions:
      - ``queued``: not yet running; claimable once every entry in
        ``dependencies`` is ``completed``.
      - ``running``: claimed by a worker.
      - ``completed``: finished successfully.
      - ``failed``: finished with an error, or transitively failed because a
        dependency failed.

    All methods are safe to call from any thread.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._jobs: dict[str, Job] = {}

    def submit(self, artifact: Furu[Any], dependencies: Iterable[str] = ()) -> bool:
        """Register a new job. Returns True if newly added, False if already present."""
        with self._cond:
            object_id = artifact.object_id
            if object_id in self._jobs:
                return False
            self._jobs[object_id] = Job(
                artifact=artifact,
                state="queued",
                dependencies=set(dependencies),
            )
            self._cond.notify_all()
            return True

    def claim_ready(self) -> Job | None:
        """Atomically claim a queued job whose every dependency is completed.

        Marks the returned job as ``running``. Returns ``None`` if no such job
        exists. Propagates upstream failures into dependent queued jobs first,
        so callers will see those as ``failed`` rather than stuck.
        """
        with self._cond:
            self._propagate_failures_locked()
            for job in self._jobs.values():
                if job.state != "queued":
                    continue
                if not all(
                    self._jobs[dep].state == "completed" for dep in job.dependencies
                ):
                    continue
                job.state = "running"
                return job
            return None

    def add_dependencies(self, object_id: str, new_dependencies: Iterable[str]) -> None:
        """Re-queue a running job and extend its dependency set.

        This is the suspension path: the worker caught
        :class:`furu.execution.BlockedOnDependencies` and is asking the
        scheduler to wait for the new dependencies before this job can run
        again. Increments the suspension counter so callers can enforce a cap.
        """
        with self._cond:
            job = self._jobs[object_id]
            job.dependencies.update(new_dependencies)
            job.state = "queued"
            job.suspension_count += 1
            self._cond.notify_all()

    def complete_job(self, object_id: str) -> None:
        with self._cond:
            self._jobs[object_id].state = "completed"
            self._cond.notify_all()

    def fail_job(self, object_id: str, error: BaseException) -> None:
        with self._cond:
            job = self._jobs[object_id]
            job.state = "failed"
            job.error = error
            self._cond.notify_all()

    def suspension_count(self, object_id: str) -> int:
        with self._cond:
            return self._jobs[object_id].suspension_count

    def is_finished(self) -> bool:
        with self._cond:
            self._propagate_failures_locked()
            return all(
                job.state in ("completed", "failed") for job in self._jobs.values()
            )

    def failures(self) -> dict[str, BaseException]:
        with self._cond:
            return {
                object_id: job.error
                for object_id, job in self._jobs.items()
                if job.state == "failed" and job.error is not None
            }

    def wait_for_change(self, timeout: float | None = None) -> None:
        with self._cond:
            self._cond.wait(timeout)

    def jobs_snapshot(self) -> dict[str, Job]:
        with self._cond:
            return {object_id: job for object_id, job in self._jobs.items()}

    def _propagate_failures_locked(self) -> None:
        progress = True
        while progress:
            progress = False
            for object_id, job in self._jobs.items():
                if job.state != "queued":
                    continue
                failed_deps = [
                    dep for dep in job.dependencies if self._jobs[dep].state == "failed"
                ]
                if failed_deps:
                    job.state = "failed"
                    job.error = UpstreamFailure(
                        f"{object_id} failed because dependencies failed: "
                        + ", ".join(sorted(failed_deps))
                    )
                    progress = True
