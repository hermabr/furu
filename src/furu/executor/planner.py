from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from furu.core import Furu

if TYPE_CHECKING:
    from furu.executor.scheduler import InMemoryScheduler, SchedulerJob


class DependencyPlanner:
    def __init__(self, scheduler: InMemoryScheduler) -> None:
        self.scheduler = scheduler

    def submit(
        self,
        finals: Furu[Any] | Sequence[Furu[Any]],
    ) -> tuple[SchedulerJob, ...]:
        return submit(self.scheduler, finals)

    def add_artifact_recursive(self, obj: Furu[Any]) -> SchedulerJob:
        return add_artifact_recursive(self.scheduler, obj)


def normalize_artifacts(
    artifact_or_artifacts: Furu[Any] | Sequence[Furu[Any]],
) -> tuple[Furu[Any], ...]:
    if isinstance(artifact_or_artifacts, Furu):
        return (artifact_or_artifacts,)
    if not isinstance(artifact_or_artifacts, Sequence):
        raise TypeError("expected a Furu object or a sequence of Furu objects")

    artifacts = tuple(artifact_or_artifacts)
    if any(not isinstance(artifact, Furu) for artifact in artifacts):
        raise TypeError("expected Furu objects")
    return artifacts


def submit(
    scheduler: InMemoryScheduler,
    finals: Furu[Any] | Sequence[Furu[Any]],
) -> tuple[SchedulerJob, ...]:
    jobs: list[SchedulerJob] = []
    for final in normalize_artifacts(finals):
        scheduler.final_ids.add(final.object_id)
        jobs.append(add_artifact_recursive(scheduler, final))
    return tuple(jobs)


def add_artifact_recursive(
    scheduler: InMemoryScheduler,
    obj: Furu[Any],
) -> SchedulerJob:
    job = scheduler.add_artifact(obj)
    if job.declared_dependencies_registered:
        return job

    job.declared_dependencies_registered = True
    scheduler.add_dependencies(obj, obj._declared_refs())
    return job
