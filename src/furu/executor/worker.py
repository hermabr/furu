from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast, overload

from furu.core import Furu
from furu.execution import BlockedOnDependencies, execute_artifact_direct
from furu.executor.planner import normalize_artifacts
from furu.executor.scheduler import InMemoryScheduler, SchedulerJob
from furu.migration import result_dir_for_loading
from furu.result import load_result_bundle


class SchedulerStalledError(RuntimeError):
    pass


class WorkerRunner:
    def __init__(self, scheduler: InMemoryScheduler) -> None:
        self.scheduler = scheduler

    def run_once(self) -> bool:
        obj = self.scheduler.claim_ready()
        if obj is None:
            return False

        try:
            execute_artifact_direct(obj, executor_mode=True)
        except BlockedOnDependencies as exc:
            self.scheduler.suspend_job(obj, exc.deps)
        except BaseException as exc:
            self.scheduler.fail_job(obj, exc)
        else:
            self.scheduler.complete_job(obj)
        return True

    def run_until_complete(self) -> None:
        while not self.scheduler.finals_completed():
            self._raise_if_failed()
            if self.run_once():
                continue

            self._raise_if_failed()
            if self.scheduler.finals_completed():
                return

            unfinished = self.scheduler.unfinished_jobs()
            labels = ", ".join(job.artifact._log_label for job in unfinished)
            raise SchedulerStalledError(
                "no runnable queued jobs remain before finals completed"
                + (f": {labels}" if labels else "")
            )

    def _raise_if_failed(self) -> None:
        failed = self.scheduler.failed_jobs()
        if failed:
            raise _failed_job_exception(failed[0])


def _failed_job_exception(job: SchedulerJob) -> BaseException:
    if job.exception is not None:
        return job.exception
    return RuntimeError(f"{job.artifact._log_label} failed without an exception")


@overload
def run_local[T](
    finals: Furu[T],
    *,
    max_suspensions_per_job: int = 10,
) -> T: ...


@overload
def run_local[T](
    finals: Sequence[Furu[T]],
    *,
    max_suspensions_per_job: int = 10,
) -> list[T]: ...


def run_local[T](
    finals: Furu[T] | Sequence[Furu[T]],
    *,
    max_suspensions_per_job: int = 10,
) -> T | list[T]:
    unwrap = isinstance(finals, Furu)
    artifacts = normalize_artifacts(cast(Furu[Any] | Sequence[Furu[Any]], finals))

    scheduler = InMemoryScheduler(max_suspensions_per_job=max_suspensions_per_job)
    scheduler.submit(artifacts)
    WorkerRunner(scheduler).run_until_complete()

    results = [_load_completed(artifact) for artifact in artifacts]
    return results[0] if unwrap else results


def _load_completed[T](obj: Furu[T]) -> T:
    result_dir = result_dir_for_loading(obj)
    if result_dir is None:
        raise RuntimeError(f"{obj._log_label} did not produce a cached result")
    return cast(T, load_result_bundle(result_dir))
