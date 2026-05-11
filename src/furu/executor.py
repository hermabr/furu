from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast, overload

from furu.core import Furu
from furu.execution import BlockedOnDependencies, _run_artifact_in_worker
from furu.migration import result_dir_for_loading
from furu.planner import build_plan
from furu.result import load_result_bundle
from furu.scheduler import Job, Scheduler

DEFAULT_MAX_SUSPENSIONS = 64
DEFAULT_WAIT_TIMEOUT_S = 0.05


class LocalExecutor:
    """Run a planned graph of Furu artifacts in a local thread pool.

    Each artifact in the plan becomes one job. Workers pick up runnable jobs
    (``queued`` with every dependency ``completed``) and execute them via
    :func:`furu.execution._run_artifact_in_worker`, which runs ``create()``
    under the artifact's own lock without going through
    :meth:`Furu.load_or_create`. ``load_or_create()`` calls made *inside*
    ``create()`` raise :class:`BlockedOnDependencies` if any requested artifact
    is missing; the worker treats that as a suspension, schedules the missing
    subgraphs, and re-queues the current job.

    Suspended jobs are re-run from the beginning. ``create()`` is therefore
    expected to be idempotent. ``max_suspensions`` caps how many times any
    single job may be suspended before it is failed with a clear error.
    """

    def __init__(
        self,
        *,
        max_workers: int = 4,
        max_suspensions: int = DEFAULT_MAX_SUSPENSIONS,
        wait_timeout_s: float = DEFAULT_WAIT_TIMEOUT_S,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if max_suspensions < 1:
            raise ValueError("max_suspensions must be >= 1")
        self._max_workers = max_workers
        self._max_suspensions = max_suspensions
        self._wait_timeout_s = wait_timeout_s
        self._scheduler = Scheduler()

    @property
    def scheduler(self) -> Scheduler:
        return self._scheduler

    def run(self, roots: Sequence[Furu[Any]]) -> None:
        plan = build_plan(roots)
        for object_id, artifact in plan.artifacts.items():
            self._scheduler.submit(artifact, dependencies=plan.edges[object_id])

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = [pool.submit(self._worker_loop) for _ in range(self._max_workers)]
            for future in futures:
                future.result()

        failures = self._scheduler.failures()
        if failures:
            first_id, first_error = next(iter(failures.items()))
            raise RuntimeError(
                f"executor finished with {len(failures)} failed job(s); "
                f"first failure on {first_id}: {first_error}"
            ) from first_error

    def _worker_loop(self) -> None:
        while True:
            if self._scheduler.is_finished():
                return
            job = self._scheduler.claim_ready()
            if job is None:
                self._scheduler.wait_for_change(timeout=self._wait_timeout_s)
                continue
            self._run_one(job)

    def _run_one(self, job: Job) -> None:
        artifact = job.artifact
        object_id = artifact.object_id

        if job.suspension_count >= self._max_suspensions:
            self._scheduler.fail_job(
                object_id,
                RuntimeError(
                    f"{object_id} suspended {job.suspension_count} times; "
                    f"exceeded max_suspensions={self._max_suspensions}"
                ),
            )
            return

        try:
            _run_artifact_in_worker(artifact)
        except BlockedOnDependencies as blocked:
            new_dep_ids: set[str] = set()
            for dep in blocked.deps:
                subplan = build_plan([dep])
                for sub_id, sub_artifact in subplan.artifacts.items():
                    self._scheduler.submit(
                        sub_artifact, dependencies=subplan.edges[sub_id]
                    )
                new_dep_ids.add(dep.object_id)
            self._scheduler.add_dependencies(object_id, new_dep_ids)
            return
        except BaseException as exc:
            self._scheduler.fail_job(object_id, exc)
            return

        self._scheduler.complete_job(object_id)


@overload
def run_local[T](
    roots: Furu[T],
    *,
    max_workers: int = 4,
    max_suspensions: int = DEFAULT_MAX_SUSPENSIONS,
) -> T: ...


@overload
def run_local[T](
    roots: Sequence[Furu[T]],
    *,
    max_workers: int = 4,
    max_suspensions: int = DEFAULT_MAX_SUSPENSIONS,
) -> list[T]: ...


def run_local[T](
    roots: Furu[T] | Sequence[Furu[T]],
    *,
    max_workers: int = 4,
    max_suspensions: int = DEFAULT_MAX_SUSPENSIONS,
) -> T | list[T]:
    """Plan and execute the dependency graph rooted at ``roots`` locally.

    Returns the cached result for each root. The roots argument shape (single
    artifact vs. sequence) determines whether the return value is unwrapped or
    a list, mirroring :func:`furu.load_or_create`.
    """
    if isinstance(roots, Furu):
        obj_list: list[Furu[T]] = [roots]
        unwrap = True
    else:
        obj_list = list(roots)
        unwrap = False
        if any(not isinstance(obj, Furu) for obj in obj_list):
            raise TypeError("run_local() expected Furu objects")

    executor = LocalExecutor(
        max_workers=max_workers,
        max_suspensions=max_suspensions,
    )
    executor.run(obj_list)

    results: list[T] = []
    for obj in obj_list:
        result_dir = result_dir_for_loading(obj)
        if result_dir is None:
            raise RuntimeError(
                f"executor finished without producing a result for {obj.object_id}"
            )
        results.append(cast(T, load_result_bundle(result_dir)))

    return results[0] if unwrap else results
